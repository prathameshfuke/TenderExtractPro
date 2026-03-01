"""
retrieval.py — Hybrid retrieval with cross-encoder reranking.

Uses pure-numpy brute-force cosine similarity for semantic search
(avoids FAISS binary compatibility issues on Python 3.14) overlaid
with BM25 keyword matching. A cross-encoder reranker refines results.

Architecture:
  1. NumPy brute-force cosine similarity with all-mpnet-base-v2 embeddings
  2. BM25Okapi for keyword-level matching (exact terms, standard codes)
  3. Weighted score fusion (0.4 BM25 + 0.6 embedding)
  4. Cross-encoder reranking on top-50 candidates → final top-k
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

from tender_extraction.config import config
from tender_extraction.schemas import Chunk

logger = logging.getLogger(__name__)

# ── Singleton caches ──────────────────────────────────────────────────────
_embed_model: Optional[SentenceTransformer] = None
_cross_encoder: Optional[CrossEncoder] = None


def _get_embed_model() -> SentenceTransformer:
    """Lazy-load and cache the sentence-transformer embedding model."""
    global _embed_model
    if _embed_model is None:
        model_name = config.retrieval.embedding_model
        logger.info("Loading embedding model: %s ...", model_name)
        _embed_model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded (dim=%d).",
                     _embed_model.get_sentence_embedding_dimension())
    return _embed_model


def _get_cross_encoder() -> CrossEncoder:
    """Lazy-load and cache the cross-encoder reranking model."""
    global _cross_encoder
    if _cross_encoder is None:
        model_name = config.retrieval.rerank_model
        logger.info("Loading cross-encoder reranker: %s ...", model_name)
        _cross_encoder = CrossEncoder(model_name, max_length=512)
        logger.info("Cross-encoder loaded.")
    return _cross_encoder


class HybridRetriever:
    """
    Numpy cosine-similarity + BM25 hybrid retriever with cross-encoder reranking.

    Usage:
        retriever = HybridRetriever()
        retriever.build_index(chunks)

        results = retriever.retrieve("steel reinforcement grade", top_k=10)
    """

    def __init__(self, persist_dir: Optional[str] = None):
        self._chunks: List[Chunk] = []
        self._bm25: Optional[BM25Okapi] = None
        self._tokenized_corpus: List[List[str]] = []
        # Numpy matrix: shape (n_chunks, embed_dim) — replaces FAISS
        self._embeddings: Optional[np.ndarray] = None
        self._persist_dir = persist_dir or "./_retrieval_index"

    def build_index(self, chunks: List[Chunk], force_rebuild: bool = False) -> None:
        """
        Build numpy embedding matrix and BM25 index from chunks.
        """
        if not chunks:
            raise ValueError("Cannot build index from empty chunk list.")

        self._chunks = chunks
        texts = [c.text for c in chunks]

        # ── BM25 index ──────────────────────────────────────────────
        logger.info("Building BM25 index for %d chunks ...", len(texts))
        self._tokenized_corpus = [t.lower().split() for t in texts]
        self._bm25 = BM25Okapi(self._tokenized_corpus)

        # ── Numpy embedding matrix ───────────────────────────────────
        logger.info("Generating embeddings (numpy brute-force cosine)...")
        embed_model = _get_embed_model()
        embeddings = embed_model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False, batch_size=32
        )
        embeddings = np.array(embeddings, dtype=np.float32)
        # L2-normalize so dot product == cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self._embeddings = embeddings / norms

        logger.info(
            "Index built: %d chunks, BM25 vocab=%d tokens, embed_dim=%d",
            len(chunks),
            sum(len(t) for t in self._tokenized_corpus),
            self._embeddings.shape[1],
        )

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval: numpy cosine similarity + BM25 + cross-encoder reranking.

        Pipeline:
          1. BM25 scores for all chunks (keyword matching)
          2. Numpy cosine similarity for semantic search (all chunks, brute-force)
          3. Weighted fusion of normalized scores
          4. Cross-encoder reranking of top candidates
          5. Return final top_k

        Returns list of dicts: {chunk, score, bm25_score, embedding_score}.
        """
        if self._bm25 is None or self._embeddings is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        n = len(self._chunks)
        rerank_k = config.retrieval.rerank_top_k

        # ── Step 1: BM25 scores ─────────────────────────────────────
        query_tokens = query.lower().split()
        bm25_raw = np.array(self._bm25.get_scores(query_tokens), dtype="float32")

        # ── Step 2: Numpy cosine similarity ─────────────────────────
        embed_model = _get_embed_model()
        qvec = embed_model.encode([query], convert_to_numpy=True)
        qvec = np.array(qvec, dtype=np.float32)
        # L2-normalize the query vector
        qnorm = np.linalg.norm(qvec)
        if qnorm > 0:
            qvec = qvec / qnorm
        # Cosine similarity = dot product (embeddings already normalized)
        emb_raw = (self._embeddings @ qvec.T).squeeze()  # shape (n,)
        # Shift from [-1, 1] to [0, 1]
        emb_raw = (emb_raw + 1.0) / 2.0

        # ── Step 3: Weighted fusion ─────────────────────────────────
        bm25_norm = _min_max_normalize(bm25_raw)
        emb_norm = _min_max_normalize(emb_raw)

        w_bm25 = config.retrieval.bm25_weight
        w_emb = config.retrieval.embedding_weight
        fused = w_bm25 * bm25_norm + w_emb * emb_norm

        # Take top candidates for reranking
        candidate_indices = np.argsort(fused)[::-1][:rerank_k]
        candidates = [
            (int(idx), float(fused[idx]))
            for idx in candidate_indices
            if fused[idx] > 0
        ]

        if not candidates:
            return []

        # ── Step 4: Cross-encoder reranking ─────────────────────────
        try:
            reranker = _get_cross_encoder()
            pairs = [(query, self._chunks[idx].text) for idx, _ in candidates]
            rerank_scores = reranker.predict(pairs)

            # Normalize rerank scores to [0, 1]
            rerank_norm = _min_max_normalize(np.array(rerank_scores, dtype="float32"))

            # Blend: 40% hybrid fusion + 60% cross-encoder
            final_scored = []
            for i, (idx, hybrid_score) in enumerate(candidates):
                final_score = 0.4 * hybrid_score + 0.6 * float(rerank_norm[i])
                final_scored.append((idx, final_score, hybrid_score))

            final_scored.sort(key=lambda x: x[1], reverse=True)
        except Exception as exc:
            logger.warning("Cross-encoder reranking failed: %s. Using hybrid scores.", exc)
            final_scored = [(idx, score, score) for idx, score in candidates]

        # ── Step 5: Build results ───────────────────────────────────
        results: List[Dict[str, Any]] = []
        for idx, final_score, hybrid_score in final_scored[:top_k]:
            results.append({
                "chunk": self._chunks[idx],
                "score": final_score,
                "bm25_score": float(bm25_norm[idx]),
                "embedding_score": float(emb_norm[idx]),
            })

        logger.info(
            "Retrieved %d chunks for '%s...' (top=%.3f, reranked)",
            len(results), query[:40], results[0]["score"] if results else 0.0,
        )
        return results

    def retrieve_spec_chunks(
        self, query: str, top_k: int = 15, prefer_pages_after: int = 15
    ) -> List[Dict[str, Any]]:
        """
        Spec-focused retrieval with page boosting for later pages
        where specs typically live in tender documents.
        """
        chunks = self.retrieve(query, top_k=top_k * 2)
        # Boost chunks from later pages where specs usually live
        for r in chunks:
            if r["chunk"].metadata.page > prefer_pages_after:
                r["score"] *= 1.2
            # Also boost table chunks — specs are often in tables
            if r["chunk"].metadata.chunk_type == "table":
                r["score"] *= 1.15
        chunks.sort(key=lambda x: x["score"], reverse=True)
        return chunks[:top_k]

    def retrieve_with_filter(
        self, query: str, top_k: int = 10,
        chunk_types: Optional[List[str]] = None,
        min_page: Optional[int] = None,
        max_page: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Filtered retrieval using numpy cosine similarity with post-filtering.
        """
        if self._embeddings is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        embed_model = _get_embed_model()
        qvec = embed_model.encode([query], convert_to_numpy=True)
        qvec = np.array(qvec, dtype=np.float32)
        qnorm = np.linalg.norm(qvec)
        if qnorm > 0:
            qvec = qvec / qnorm
        # All cosine similarities
        sims = (self._embeddings @ qvec.T).squeeze()
        # Sort descending
        sorted_indices = np.argsort(sims)[::-1]

        results = []
        for idx in sorted_indices:
            idx = int(idx)
            chunk = self._chunks[idx]

            if chunk_types and chunk.metadata.chunk_type not in chunk_types:
                continue
            if min_page is not None and chunk.metadata.page < min_page:
                continue
            if max_page is not None and chunk.metadata.page > max_page:
                continue

            score = float((sims[idx] + 1.0) / 2.0)
            results.append({
                "chunk": chunk,
                "score": score,
                "bm25_score": 0.0,
                "embedding_score": score,
            })

            if len(results) >= top_k:
                break

        return results

    def save(self, directory: str) -> None:
        """FAISS doesn't auto-persist, but we add dummy API for compatibility."""
        logger.info("FAISS index built in-memory. Persistence mocked.")

    @classmethod
    def load(cls, directory: str) -> "HybridRetriever":
        """
        Load is mocked for API compatibility. Index built fresh.
        """
        retriever = cls(persist_dir=directory)
        logger.info("HybridRetriever loaded (mock persistence).")
        return retriever


def _min_max_normalize(arr: np.ndarray) -> np.ndarray:
    """Normalize to [0, 1]. Returns zeros if all values identical."""
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-9:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


# ── Query expansion utilities ────────────────────────────────────────────

# Domain-specific synonym map for tender documents
_SYNONYMS = {
    "specification": ["specs", "requirements", "parameters", "standard", "criteria"],
    "material": ["grade", "composition", "alloy", "substance"],
    "dimension": ["size", "length", "width", "height", "thickness", "diameter"],
    "tolerance": ["deviation", "allowance", "accuracy", "precision", "variance"],
    "supply": ["procurement", "purchase", "delivery", "furnishing"],
    "scope": ["work", "deliverables", "obligations", "tasks", "responsibilities"],
    "standard": ["IS", "ASTM", "ISO", "BIS", "code", "norm"],
    "testing": ["inspection", "quality", "examination", "verification"],
    "warranty": ["guarantee", "maintenance", "defect liability"],
}


def expand_query(query: str) -> str:
    """
    Expand a query with domain-specific synonyms.
    Adds 1-2 relevant synonyms per matching keyword to improve recall
    without overwhelming BM25 with too many terms.
    """
    query_lower = query.lower()
    expansions = []
    for term, synonyms in _SYNONYMS.items():
        if term in query_lower:
            # Add top 2 synonyms that aren't already in the query
            added = 0
            for syn in synonyms:
                if syn.lower() not in query_lower and added < 2:
                    expansions.append(syn)
                    added += 1
    if expansions:
        return query + " " + " ".join(expansions)
    return query


# ── Smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tender_extraction.ingestion import ingest_document
    from tender_extraction.chunking import create_chunks
    from tender_extraction.table_extraction import extract_tables

    pdf = "dataset/globaltender1576.pdf"
    if not Path(pdf).exists():
        print(f"Dataset file not found: {pdf}")
        sys.exit(1)

    print(f"Ingesting {pdf} ...")
    pages = ingest_document(pdf)
    tables = extract_tables(pdf)
    chunks = create_chunks(pages, tables)
    print(f"Created {len(chunks)} chunks.")

    print("Building FAISS hybrid index ...")
    retriever = HybridRetriever()
    retriever.build_index(chunks)

    # Test retrieval with real queries
    queries = [
        "technical specifications requirements",
        "scope of work deliverables",
        "steel reinforcement grade specification",
    ]
    for q in queries:
        results = retriever.retrieve(q, top_k=5)
        print(f"\nQuery: '{q}'")
        for r in results:
            c = r["chunk"]
            print(f"  score={r['score']:.3f} page={c.metadata.page} "
                  f"type={c.metadata.chunk_type}: {c.text[:80]}...")

    # Test query expansion
    expanded = expand_query("specification tolerance material")
    print(f"\nExpanded query: '{expanded}'")

    print("\nRetrieval smoke test passed.")
