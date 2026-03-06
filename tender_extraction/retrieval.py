"""
retrieval.py â€” Qdrant-backed hybrid retrieval with cross-encoder reranking.

Architecture
============
1.  BGE-large-en-v1.5 (1024-dim) dense embeddings stored in Qdrant.
    Asymmetric retrieval: documents indexed bare; queries prefixed with
    the BGE instruction "Represent this sentence for searching relevant
    passages: " which measurably improves recall for domain text.

2.  BM25Okapi in-memory index for exact/keyword matching (standard codes,
    part numbers, IS/ASTM references that semantic vectors miss).

3.  Weighted score fusion:  0.35 * BM25_norm + 0.65 * Qdrant_cosine

4.  Cross-encoder reranking (ms-marco-MiniLM-L-6-v2) on the top-40
    candidates before selecting final top-k.

5.  Optional Qdrant payload filter â€” pass section="Technical Specifications"
    to restrict search to spec chunks at the database level.

Qdrant runs fully in-process (no server) via the local-storage client mode.
One collection per document, named by the caller (typically the job_id).
"""

from __future__ import annotations

import logging
import io
import contextlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from tender_extraction.config import config
from tender_extraction.schemas import Chunk

logger = logging.getLogger(__name__)

# â”€â”€ Singleton model caches â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_embed_model: Optional[SentenceTransformer] = None
_cross_encoder: Optional[CrossEncoder] = None


def _resolve_torch_device() -> str:
    """Return 'cuda' when available, else 'cpu'."""
    try:
        import torch
        cuda_ok = bool(torch.cuda.is_available())
        if config.retrieval.require_gpu and not cuda_ok:
            raise RuntimeError(
                "GPU-only mode is enabled (REQUIRE_GPU=1), but PyTorch CUDA is not available. "
                "Install a CUDA-enabled torch build in the active environment."
            )
        return "cuda" if cuda_ok else "cpu"
    except Exception:
        if config.retrieval.require_gpu:
            raise
        return "cpu"

# BGE asymmetric instruction used at query time only (not at index time)
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _get_embed_model() -> SentenceTransformer:
    """Lazy-load and cache bge-large-en-v1.5."""
    global _embed_model
    if _embed_model is None:
        model_name = config.retrieval.embedding_model
        device = _resolve_torch_device()
        logger.info("Loading embedding model: %s ...", model_name)
        logger.info("Embedding model device: %s", device)
        # Some backend loaders print verbose diagnostics directly to stdout/stderr.
        # Silence those so the CLI progress bar stays readable.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            _embed_model = SentenceTransformer(model_name, device=device)
        logger.info(
            "Embedding model loaded (dim=%d).",
            _embed_model.get_sentence_embedding_dimension(),
        )
    return _embed_model


def _get_cross_encoder() -> CrossEncoder:
    """Lazy-load and cache the cross-encoder reranker."""
    global _cross_encoder
    if _cross_encoder is None:
        model_name = config.retrieval.rerank_model
        device = _resolve_torch_device()
        logger.info("Loading cross-encoder reranker: %s ...", model_name)
        logger.info("Cross-encoder device: %s", device)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            _cross_encoder = CrossEncoder(model_name, max_length=512, device=device)
        logger.info("Cross-encoder loaded.")
    return _cross_encoder


class HybridRetriever:
    """
    Qdrant-backed hybrid retriever.

    Usage::

        retriever = HybridRetriever()
        retriever.build_index(chunks, collection_name="job_abc123")
        results = retriever.retrieve("steel reinforcement grade", top_k=10)
        # filter to spec chunks only:
        results = retriever.retrieve_spec_chunks("yield strength", top_k=15)
    """

    def __init__(self, persist_dir: Optional[str] = None):
        self._chunks: List[Chunk] = []
        self._bm25: Optional[BM25Okapi] = None
        self._tokenized_corpus: List[List[str]] = []

        persist_path = str(persist_dir or config.retrieval.qdrant_path)
        Path(persist_path).mkdir(parents=True, exist_ok=True)

        # QdrantClient(path=...) uses the in-process local storage engine
        # (pure-Python, ships with qdrant-client) â€” no external server needed.
        self._qdrant = QdrantClient(path=persist_path)
        self._collection_name: Optional[str] = None

        logger.info("Qdrant local client initialised at: %s", persist_path)

    def close(self) -> None:
        """Explicitly close local Qdrant resources."""
        try:
            self._qdrant.close()
        except Exception:
            pass

    # â”€â”€ Index building â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def build_index(
        self,
        chunks: List[Chunk],
        collection_name: str = "tender",
        force_rebuild: bool = False,
    ) -> None:
        """
        Encode all chunks with BGE-large and upsert into a Qdrant collection.
        Also builds an in-memory BM25 index for hybrid scoring.

        If the collection already exists and force_rebuild=False, the Qdrant
        index is reused (BM25 is always rebuilt from the provided chunks).
        """
        if not chunks:
            raise ValueError("Cannot build index from empty chunk list.")

        self._chunks = chunks
        self._collection_name = collection_name
        texts = [c.text for c in chunks]

        # BM25 â€” always rebuilt (fast, in-memory)
        logger.info("Building BM25 index for %d chunks...", len(texts))
        self._tokenized_corpus = [t.lower().split() for t in texts]
        self._bm25 = BM25Okapi(self._tokenized_corpus)

        # Qdrant â€” rebuild only if needed
        existing = {c.name for c in self._qdrant.get_collections().collections}
        if collection_name in existing and not force_rebuild:
            logger.info(
                "Qdrant collection '%s' already exists â€” reusing (pass "
                "force_rebuild=True to re-embed).",
                collection_name,
            )
            return

        if collection_name in existing:
            self._qdrant.delete_collection(collection_name)
            logger.info("Deleted existing Qdrant collection '%s'.", collection_name)

        # Encode with BGE-large (documents: no instruction prefix)
        embed_model = _get_embed_model()
        embed_dim = embed_model.get_sentence_embedding_dimension()
        logger.info(
            "Encoding %d chunks with %s (dim=%d)...",
            len(texts), config.retrieval.embedding_model, embed_dim,
        )
        embeddings = embed_model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=16,
            normalize_embeddings=True,   # cosine = dot-product on unit vectors
        )

        # Create Qdrant collection
        self._qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embed_dim, distance=Distance.COSINE),
        )

        # Upsert in batches of 200 to stay memory-efficient
        batch_size = 200
        for start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[start: start + batch_size]
            batch_embs = embeddings[start: start + batch_size]
            points = [
                PointStruct(
                    id=start + i,
                    vector=emb.tolist(),
                    payload={
                        "chunk_id": ch.chunk_id,
                        "text": ch.text,
                        "section": ch.metadata.section,
                        "parent_section": ch.metadata.parent_section,
                        "chunk_type": ch.metadata.chunk_type,
                        "page": ch.metadata.page,
                        "table_id": ch.metadata.table_id or "",
                    },
                )
                for i, (ch, emb) in enumerate(zip(batch_chunks, batch_embs))
            ]
            self._qdrant.upsert(collection_name=collection_name, points=points)

        logger.info(
            "Qdrant index built: %d chunks, collection='%s', embed_dim=%d",
            len(chunks), collection_name, embed_dim,
        )

    # â”€â”€ Core retrieval â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        section_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval: Qdrant dense search + BM25 + cross-encoder reranking.

        Parameters
        ----------
        query          : Natural language query.
        top_k          : Number of results to return after reranking.
        section_filter : If set, restricts Qdrant search to chunks whose
                         `section` payload matches exactly (e.g.
                         "Technical Specifications").  BM25 candidates are
                         **not** filtered by section so there's always a
                         meaningful fusion pool.

        Returns
        -------
        List of dicts: {chunk, score, bm25_score, embedding_score}
        """
        if self._bm25 is None or self._collection_name is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        n = len(self._chunks)
        rerank_k = min(config.retrieval.rerank_top_k, n)

        # â”€â”€ Step 1: BM25 scores (all chunks) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        query_tokens = query.lower().split()
        bm25_raw = np.array(self._bm25.get_scores(query_tokens), dtype="float32")
        bm25_norm = _min_max_normalize(bm25_raw)

        # â”€â”€ Step 2: Qdrant dense search â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        embed_model = _get_embed_model()
        # BGE asymmetric: query gets an instruction prefix; documents do not.
        query_with_prefix = _BGE_QUERY_PREFIX + query
        qvec = embed_model.encode(
            [query_with_prefix],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]

        qdrant_filter: Optional[Filter] = None
        if section_filter:
            qdrant_filter = Filter(
                must=[
                    FieldCondition(
                        key="section",
                        match=MatchValue(value=section_filter),
                    )
                ]
            )

        qdrant_hits = self._qdrant.query_points(
            collection_name=self._collection_name,
            query=qvec.tolist(),
            query_filter=qdrant_filter,
            limit=rerank_k,
            with_payload=False,   # payloads already stored in self._chunks
        ).points

        # Build map: qdrant point_id (== chunk list index) â†’ cosine score
        hit_id_to_emb_score: Dict[int, float] = {
            int(h.id): float(h.score) for h in qdrant_hits
        }

        # â”€â”€ Step 3: Fuse BM25 + Qdrant scores â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Candidate pool = top-BM25 âˆª top-Qdrant
        top_bm25_ids = set(np.argsort(bm25_raw)[::-1][:rerank_k].tolist())
        top_qdrant_ids = set(hit_id_to_emb_score.keys())
        candidate_ids = list(top_bm25_ids | top_qdrant_ids)

        w_bm25 = config.retrieval.bm25_weight
        w_emb = config.retrieval.embedding_weight
        fused: List[tuple] = []   # (idx, fused_score, bm25_s, emb_s)
        for raw_idx in candidate_ids:
            idx = int(raw_idx)
            if idx >= n:
                continue
            bm25_s = float(bm25_norm[idx])
            emb_s = hit_id_to_emb_score.get(idx, 0.0)
            score = w_bm25 * bm25_s + w_emb * emb_s
            fused.append((idx, score, bm25_s, emb_s))

        fused.sort(key=lambda x: x[1], reverse=True)
        top_candidates = fused[:rerank_k]

        if not top_candidates:
            return []

        # â”€â”€ Step 4: Cross-encoder reranking â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        try:
            reranker = _get_cross_encoder()
            pairs = [(query, self._chunks[idx].text) for idx, *_ in top_candidates]
            ce_scores = reranker.predict(pairs)
            ce_norm = _min_max_normalize(np.array(ce_scores, dtype="float32"))

            final_scored: List[tuple] = []
            for i, (idx, hybrid_score, bm25_s, emb_s) in enumerate(top_candidates):
                # 35% hybrid + 65% cross-encoder for final rank
                final = 0.35 * hybrid_score + 0.65 * float(ce_norm[i])
                final_scored.append((idx, final, bm25_s, emb_s))

            final_scored.sort(key=lambda x: x[1], reverse=True)
        except Exception as exc:
            logger.warning(
                "Cross-encoder reranking failed: %s. Using hybrid scores.", exc
            )
            final_scored = list(top_candidates)

        # â”€â”€ Step 5: Build results â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        results: List[Dict[str, Any]] = []
        for idx, final_score, bm25_s, emb_s in final_scored[:top_k]:
            results.append(
                {
                    "chunk": self._chunks[idx],
                    "score": final_score,
                    "bm25_score": bm25_s,
                    "embedding_score": emb_s,
                }
            )

        logger.info(
            "Retrieved %d chunks for '%s...' (top=%.3f, section_filter=%s)",
            len(results),
            query[:40],
            results[0]["score"] if results else 0.0,
            section_filter,
        )
        return results

    # â”€â”€ Specialised retrieval variants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def retrieve_spec_chunks(
        self, query: str, top_k: int = 15
    ) -> List[Dict[str, Any]]:
        """
        Spec-focused retrieval with boosting for table chunks and later pages
        (where specs typically live in Indian government tenders).
        Also tries a section-filtered Qdrant pass and merges results.
        """
        # Broad pass (no filter) so we don't miss mislabelled sections
        results = self.retrieve(query, top_k=top_k * 2, section_filter=None)

        for r in results:
            ch = r["chunk"]
            if ch.metadata.chunk_type == "table":
                r["score"] = min(1.0, r["score"] * 1.25)
            if ch.metadata.page > 15:
                r["score"] = min(1.0, r["score"] * 1.10)
            sec = (ch.metadata.section or "").lower()
            if any(
                kw in sec
                for kw in ("specification", "technical", "parameter", "requirement")
            ):
                r["score"] = min(1.0, r["score"] * 1.15)

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def retrieve_with_filter(
        self,
        query: str,
        top_k: int = 10,
        chunk_types: Optional[List[str]] = None,
        min_page: Optional[int] = None,
        max_page: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Post-filtered retrieval by chunk type and page range."""
        results = self.retrieve(query, top_k=top_k * 4)
        filtered = []
        for r in results:
            ch = r["chunk"]
            if chunk_types and ch.metadata.chunk_type not in chunk_types:
                continue
            if min_page is not None and ch.metadata.page < min_page:
                continue
            if max_page is not None and ch.metadata.page > max_page:
                continue
            filtered.append(r)
            if len(filtered) >= top_k:
                break
        return filtered

    def delete_collection(self, collection_name: Optional[str] = None) -> None:
        """Delete a Qdrant collection (e.g. for cleanup after a job)."""
        name = collection_name or self._collection_name
        if name:
            try:
                self._qdrant.delete_collection(name)
                logger.info("Deleted Qdrant collection: %s", name)
            except Exception as exc:
                logger.warning("Could not delete collection %s: %s", name, exc)


# â”€â”€ Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _min_max_normalize(arr: np.ndarray) -> np.ndarray:
    """Normalise to [0, 1]. Returns zeros if all values identical."""
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-9:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


# â”€â”€ Query expansion â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_SYNONYMS: Dict[str, List[str]] = {
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
    """Expand query with up to 2 domain synonyms per matching keyword."""
    query_lower = query.lower()
    expansions = []
    for term, synonyms in _SYNONYMS.items():
        if term in query_lower:
            added = 0
            for syn in synonyms:
                if syn.lower() not in query_lower and added < 2:
                    expansions.append(syn)
                    added += 1
    return (query + " " + " ".join(expansions)).strip() if expansions else query

