"""
retrieval.py — ChromaDB hybrid retrieval with cross-encoder reranking.

This module builds a ChromaDB persistent collection for semantic search
and overlays BM25 keyword matching for hybrid retrieval. A cross-encoder
reranker refines the final ranking for maximum precision.

Architecture:
  1. ChromaDB collection with all-mpnet-base-v2 embeddings (semantic)
  2. BM25Okapi for keyword-level matching (exact terms, standard codes)
  3. Weighted score fusion (0.4 BM25 + 0.6 embedding)
  4. Cross-encoder reranking on top-50 candidates → final top-k
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

import chromadb
from chromadb.utils import embedding_functions

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


def _collection_fingerprint(chunks: List[Chunk]) -> str:
    """
    Create a deterministic fingerprint from chunk contents.
    Used to detect when the document has changed and the collection
    needs to be rebuilt.
    """
    h = hashlib.sha256()
    for c in sorted(chunks, key=lambda x: x.chunk_id):
        h.update(c.chunk_id.encode())
        h.update(c.text[:200].encode())
    return h.hexdigest()[:16]


class HybridRetriever:
    """
    ChromaDB + BM25 hybrid retriever with cross-encoder reranking.

    Usage:
        retriever = HybridRetriever()
        retriever.build_index(chunks)

        results = retriever.retrieve("steel reinforcement grade", top_k=10)
    """

    def __init__(self, persist_dir: Optional[str] = None):
        self._chunks: List[Chunk] = []
        self._chunk_id_map: Dict[str, int] = {}  # chunk_id → index
        self._bm25: Optional[BM25Okapi] = None
        self._tokenized_corpus: List[List[str]] = []
        self._collection = None
        self._persist_dir = persist_dir or config.retrieval.chroma_persist_dir

    def build_index(self, chunks: List[Chunk], force_rebuild: bool = False) -> None:
        """
        Build ChromaDB collection and BM25 index from chunks.

        If a persistent collection already exists with the same fingerprint,
        skip re-indexing (saves ~8s encoding time on CPU).
        """
        if not chunks:
            raise ValueError("Cannot build index from empty chunk list.")

        self._chunks = chunks
        self._chunk_id_map = {c.chunk_id: i for i, c in enumerate(chunks)}
        texts = [c.text for c in chunks]

        # ── BM25 index ──────────────────────────────────────────────
        logger.info("Building BM25 index for %d chunks ...", len(texts))
        self._tokenized_corpus = [t.lower().split() for t in texts]
        self._bm25 = BM25Okapi(self._tokenized_corpus)

        # ── ChromaDB collection ─────────────────────────────────────
        fingerprint = _collection_fingerprint(chunks)
        collection_name = f"tender_{fingerprint}"

        # Use SentenceTransformer embedding function for ChromaDB
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.retrieval.embedding_model
        )

        client = chromadb.PersistentClient(
            path=self._persist_dir,
        )

        # Check if collection already exists with matching data
        existing_collections = [c.name for c in client.list_collections()]
        if collection_name in existing_collections and not force_rebuild:
            logger.info("ChromaDB collection '%s' exists — reusing.", collection_name)
            self._collection = client.get_collection(
                name=collection_name,
                embedding_function=ef,
            )
        else:
            # Clean up old collections
            for old_name in existing_collections:
                if old_name.startswith("tender_"):
                    try:
                        client.delete_collection(old_name)
                    except Exception:
                        pass

            logger.info("Building ChromaDB collection '%s' for %d chunks ...",
                        collection_name, len(chunks))
            self._collection = client.create_collection(
                name=collection_name,
                embedding_function=ef,
                metadata={"hnsw:space": "cosine"},
            )

            # Add chunks in batches (ChromaDB has a batch size limit)
            batch_size = 500
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                self._collection.add(
                    ids=[c.chunk_id for c in batch],
                    documents=[c.text for c in batch],
                    metadatas=[{
                        "page": c.metadata.page,
                        "section": c.metadata.section,
                        "chunk_type": c.metadata.chunk_type,
                        "parent_section": c.metadata.parent_section,
                    } for c in batch],
                )
            logger.info("ChromaDB collection built with %d chunks.", len(chunks))

        logger.info(
            "Index built: %d chunks, BM25 vocab=%d tokens, ChromaDB collection='%s'",
            len(chunks), sum(len(t) for t in self._tokenized_corpus), collection_name,
        )

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval: ChromaDB semantic + BM25 keyword with cross-encoder reranking.

        Pipeline:
          1. BM25 scores for all chunks (keyword matching)
          2. ChromaDB query for semantic similarity (top rerank_top_k)
          3. Weighted fusion of normalized scores
          4. Cross-encoder reranking of top candidates
          5. Return final top_k

        Returns list of dicts: {chunk, score, bm25_score, embedding_score}.
        """
        if self._bm25 is None or self._collection is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        n = len(self._chunks)
        rerank_k = config.retrieval.rerank_top_k

        # ── Step 1: BM25 scores ─────────────────────────────────────
        query_tokens = query.lower().split()
        bm25_raw = np.array(self._bm25.get_scores(query_tokens), dtype="float32")

        # ── Step 2: ChromaDB semantic search ────────────────────────
        chroma_results = self._collection.query(
            query_texts=[query],
            n_results=min(rerank_k, n),
            include=["distances"],
        )

        # ChromaDB cosine distance → similarity (1 - distance)
        emb_raw = np.zeros(n, dtype="float32")
        if chroma_results and chroma_results["ids"] and chroma_results["ids"][0]:
            for doc_id, distance in zip(
                chroma_results["ids"][0],
                chroma_results["distances"][0],
            ):
                if doc_id in self._chunk_id_map:
                    idx = self._chunk_id_map[doc_id]
                    emb_raw[idx] = max(0, 1.0 - distance)

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
        Filtered retrieval using ChromaDB metadata filters.
        Useful for targeting specific chunk types or page ranges.
        """
        if self._collection is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        where_filter = {}
        conditions = []

        if chunk_types:
            conditions.append({"chunk_type": {"$in": chunk_types}})
        if min_page is not None:
            conditions.append({"page": {"$gte": min_page}})
        if max_page is not None:
            conditions.append({"page": {"$lte": max_page}})

        if len(conditions) > 1:
            where_filter = {"$and": conditions}
        elif len(conditions) == 1:
            where_filter = conditions[0]

        try:
            chroma_results = self._collection.query(
                query_texts=[query],
                n_results=min(top_k, len(self._chunks)),
                where=where_filter if where_filter else None,
                include=["distances"],
            )
        except Exception as exc:
            logger.warning("Filtered query failed: %s. Falling back to unfiltered.", exc)
            return self.retrieve(query, top_k=top_k)

        results = []
        if chroma_results and chroma_results["ids"] and chroma_results["ids"][0]:
            for doc_id, distance in zip(
                chroma_results["ids"][0],
                chroma_results["distances"][0],
            ):
                if doc_id in self._chunk_id_map:
                    idx = self._chunk_id_map[doc_id]
                    score = max(0, 1.0 - distance)
                    results.append({
                        "chunk": self._chunks[idx],
                        "score": score,
                        "bm25_score": 0.0,
                        "embedding_score": score,
                    })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def save(self, directory: str) -> None:
        """ChromaDB auto-persists. This is a no-op for API compatibility."""
        logger.info("ChromaDB auto-persists to: %s", self._persist_dir)

    @classmethod
    def load(cls, directory: str) -> "HybridRetriever":
        """
        Load is not needed for ChromaDB (auto-persistent).
        Create a new retriever pointing to the persist directory.
        """
        retriever = cls(persist_dir=directory)
        logger.info("HybridRetriever configured with persist_dir: %s", directory)
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

    print("Building ChromaDB hybrid index ...")
    retriever = HybridRetriever(persist_dir="./_test_chroma_db")
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

    # Cleanup
    shutil.rmtree("./_test_chroma_db", ignore_errors=True)
    print("\nRetrieval smoke test passed.")
