"""
retrieval.py — Qdrant-backed hybrid retrieval with Parent-Child logic.

Architecture
============
1.  Parent-Child Strategy:
    - Parents: Semantic or section-aware chunks (500-1000 tokens).
    - Children: Smaller overlapping spans (200 tokens) indexed for precision.
    - Retrieval: Search hits children -> returns unique parents.

2.  BGE-large-en-v1.5 dense embeddings stored in Qdrant.
3.  BM25Okapi for exact/keyword matching on child chunks.
4.  Weighted score fusion + optional Cross-encoder reranking.
"""

from __future__ import annotations

import logging
import io
import contextlib
import re
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

_SPEC_SECTION_TERMS = (
    "specification", "technical", "parameter", "requirement", "compliance", "material", "performance",
)
_SCOPE_SECTION_TERMS = (
    "scope", "work", "deliverable", "timeline", "schedule", "responsibil", "obligation", "exclusion",
)
_SPEC_TEXT_TERMS = re.compile(
    r"\b(astm|iso|bis|is\s*[:\-]|minimum|maximum|tolerance|voltage|current|capacity|size|diameter|temperature|pressure|grade)\b",
    re.IGNORECASE,
)
_SCOPE_TEXT_TERMS = re.compile(
    r"\b(shall|must|deliver|supply|install|commission|timeline|schedule|exclude|location|site|responsib)\w*\b",
    re.IGNORECASE,
)

# Singleton model caches
_embed_model: Optional[SentenceTransformer] = None
_cross_encoder: Optional[CrossEncoder] = None


def _resolve_torch_device() -> str:
    """Return 'cuda' when available, else 'cpu'."""
    try:
        import torch
        cuda_ok = bool(torch.cuda.is_available())
        return "cuda" if cuda_ok else "cpu"
    except Exception:
        return "cpu"

_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _get_embed_model() -> SentenceTransformer:
    """Lazy-load and cache bge-large-en-v1.5."""
    global _embed_model
    if _embed_model is None:
        model_name = config.retrieval.embedding_model
        device = _resolve_torch_device()
        logger.info("Loading embedding model: %s on %s", model_name, device)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            _embed_model = SentenceTransformer(model_name, device=device)
    return _embed_model


def _get_cross_encoder() -> CrossEncoder:
    """Lazy-load and cache the cross-encoder reranker."""
    global _cross_encoder
    if _cross_encoder is None:
        model_name = config.retrieval.rerank_model
        device = _resolve_torch_device()
        logger.info("Loading cross-encoder: %s on %s", model_name, device)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            _cross_encoder = CrossEncoder(model_name, max_length=512, device=device)
    return _cross_encoder


class HybridRetriever:
    """
    Qdrant-backed hybrid retriever with Parent-Child logic.
    """

    def __init__(self, persist_dir: Optional[str] = None):
        self._parents: Dict[str, Chunk] = {}
        self._child_chunks: List[Dict[str, Any]] = []
        self._bm25: Optional[BM25Okapi] = None

        persist_path = str(persist_dir or config.retrieval.qdrant_path)
        Path(persist_path).mkdir(parents=True, exist_ok=True)

        self._qdrant = QdrantClient(path=persist_path)
        self._collection_name: Optional[str] = None

        logger.info("Qdrant local client initialised at: %s", persist_path)

    def close(self) -> None:
        try:
            self._qdrant.close()
        except Exception:
            pass

    def build_index(
        self,
        chunks: List[Chunk],
        collection_name: str = "tender",
        force_rebuild: bool = False,
    ) -> None:
        """
        Index children but keep parents for retrieval.
        """
        if not chunks:
            raise ValueError("Cannot build index from empty chunk list.")

        self._collection_name = collection_name
        self._parents = {c.chunk_id: c for c in chunks}
        
        # Create child chunks
        child_chunks: List[Dict[str, Any]] = []
        for parent in chunks:
            children_texts = self._create_child_texts(parent.text)
            for i, child_text in enumerate(children_texts):
                child_chunks.append({
                    "text": child_text,
                    "parent_id": parent.chunk_id,
                })
        
        self._child_chunks = child_chunks
        texts_to_index = [c["text"] for c in child_chunks]

        # BM25 on children
        logger.info("Building BM25 for %d children...", len(texts_to_index))
        self._bm25 = BM25Okapi([t.lower().split() for t in texts_to_index])

        # Qdrant
        existing = {c.name for c in self._qdrant.get_collections().collections}
        if collection_name in existing and not force_rebuild:
            logger.info("Reusing Qdrant collection '%s'.", collection_name)
            return

        if collection_name in existing:
            self._qdrant.delete_collection(collection_name)

        embed_model = _get_embed_model()
        embed_dim = embed_model.get_sentence_embedding_dimension()
        
        logger.info("Encoding %d children...", len(texts_to_index))
        embeddings = embed_model.encode(
            texts_to_index,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=32,
            normalize_embeddings=True,
        )

        self._qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embed_dim, distance=Distance.COSINE),
        )

        points = [
            PointStruct(
                id=i,
                vector=emb.tolist(),
                payload={"parent_id": item["parent_id"], "text": item["text"]},
            )
            for i, (item, emb) in enumerate(zip(child_chunks, embeddings))
        ]
        
        # Upsert in chunks to avoid large request payload issues
        for i in range(0, len(points), 1000):
            self._qdrant.upsert(collection_name=collection_name, points=points[i:i+1000])

        logger.info("Index built: %d children from %d parents.", len(child_chunks), len(chunks))

    def _create_child_texts(self, text: str, size: int = 200, overlap: int = 50) -> List[str]:
        words = text.split()
        if len(words) <= size:
            return [text]
        children = []
        for i in range(0, len(words), size - overlap):
            span = " ".join(words[i : i + size])
            children.append(span)
            if i + size >= len(words):
                break
        return children

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        section_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval on children -> return parents.
        """
        if self._bm25 is None or self._collection_name is None:
            raise RuntimeError("Index not built.")

        # BM25
        query_tokens = query.lower().split()
        bm25_raw = np.array(self._bm25.get_scores(query_tokens), dtype="float32")
        bm25_norm = _min_max_normalize(bm25_raw)

        # Qdrant
        embed_model = _get_embed_model()
        qvec = embed_model.encode([_BGE_QUERY_PREFIX + query], normalize_embeddings=True)[0]
        
        qdrant_hits = self._qdrant.query_points(
            collection_name=self._collection_name,
            query=qvec.tolist(),
            limit=top_k * 5,
        ).points

        hit_id_to_emb_score = {int(h.id): float(h.score) for h in qdrant_hits}

        # Fuse
        rerank_k = min(top_k * 5, len(self._child_chunks))
        top_bm25_ids = set(np.argsort(bm25_raw)[::-1][:rerank_k].tolist())
        top_qdrant_ids = set(hit_id_to_emb_score.keys())
        candidate_ids = list(top_bm25_ids | top_qdrant_ids)

        w_bm25, w_emb = config.retrieval.bm25_weight, config.retrieval.embedding_weight
        child_scored = []
        for idx in candidate_ids:
            if idx >= len(self._child_chunks): continue
            score = w_bm25 * float(bm25_norm[idx]) + w_emb * hit_id_to_emb_score.get(idx, 0.0)
            child_scored.append((idx, score))

        child_scored.sort(key=lambda x: x[1], reverse=True)

        # Map to parents
        seen_parents = set()
        final_results = []
        for idx, score in child_scored:
            pid = self._child_chunks[idx]["parent_id"]
            if pid not in seen_parents:
                seen_parents.add(pid)
                final_results.append({
                    "chunk": self._parents[pid],
                    "score": score,
                    "bm25_score": float(bm25_norm[idx]),
                    "embedding_score": hit_id_to_emb_score.get(idx, 0.0)
                })
            if len(final_results) >= top_k:
                break
        
        return final_results

    def retrieve_spec_chunks(self, query: str, top_k: int = 15) -> List[Dict[str, Any]]:
        results = self.retrieve(query, top_k=top_k * 2)
        for r in results:
            ch = r["chunk"]
            if ch.metadata.chunk_type == "table":
                r["score"] *= 1.15
            if any(term in (ch.metadata.section or "").lower() for term in _SPEC_SECTION_TERMS):
                r["score"] *= 1.15
            if _SPEC_TEXT_TERMS.search(ch.text):
                r["score"] *= 1.10
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def retrieve_scope_chunks(self, query: str, top_k: int = 12) -> List[Dict[str, Any]]:
        results = self.retrieve(query, top_k=top_k * 2)
        for r in results:
            ch = r["chunk"]
            if ch.metadata.chunk_type in {"paragraph", "list"}:
                r["score"] *= 1.10
            if any(term in (ch.metadata.section or "").lower() for term in _SCOPE_SECTION_TERMS):
                r["score"] *= 1.20
            if _SCOPE_TEXT_TERMS.search(ch.text):
                r["score"] *= 1.10
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def retrieve_question_chunks(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Specific retrieval for QA queries."""
        # We could add specific logic here, like emphasizing paragraph chunks
        return self.retrieve(query, top_k=top_k)

    def delete_collection(self, collection_name: Optional[str] = None) -> None:
        name = collection_name or self._collection_name
        if name:
            try:
                self._qdrant.delete_collection(name)
            except Exception:
                pass


def _min_max_normalize(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-9:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def expand_query(query: str) -> str:
    # (Simple synonym expansion from original)
    _SYNONYMS = {"specification": ["specs", "standard"], "material": ["grade", "alloy"], "dimension": ["size", "diameter"]}
    query_lower = query.lower()
    expansions = []
    for term, syns in _SYNONYMS.items():
        if term in query_lower:
            expansions.extend([s for s in syns if s.lower() not in query_lower][:2])
    return (query + " " + " ".join(expansions)).strip() if expansions else query
