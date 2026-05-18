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
import json
import re
import hashlib
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
from tender_extraction.model_runtime import get_embedding_dimension, get_embedding_model, resolve_embedding_device
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

_cross_encoder: Optional[CrossEncoder] = None


_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _get_cross_encoder() -> CrossEncoder:
    """Lazy-load and cache the cross-encoder reranker."""
    global _cross_encoder
    if _cross_encoder is None:
        model_name = config.retrieval.rerank_model
        device = resolve_embedding_device()
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

        raw_dir = str(persist_dir or config.retrieval.qdrant_path)
        self._in_memory = (raw_dir == ":memory:")

        if self._in_memory:
            # No file-locking, no disk I/O — ideal for ephemeral QA sessions.
            self._qdrant = QdrantClient(location=":memory:")
            self._persist_path = None
            self._manifest_path = None
            logger.info("Qdrant in-memory client initialised (QA session mode).")
        else:
            self._persist_path = Path(raw_dir)
            self._persist_path.mkdir(parents=True, exist_ok=True)
            self._manifest_path = self._persist_path / "_hybrid_manifest.json"
            self._qdrant = QdrantClient(path=raw_dir)
            logger.info("Qdrant local client initialised at: %s", raw_dir)

        self._collection_name: Optional[str] = None

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
        chunk_fingerprint = self._fingerprint_chunks(chunks)
        
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
        manifest = self._load_manifest()
        if (
            collection_name in existing
            and not force_rebuild
            and manifest.get(collection_name, {}).get("fingerprint") == chunk_fingerprint
        ):
            logger.info("Reusing Qdrant collection '%s'.", collection_name)
            return

        if collection_name in existing:
            self._qdrant.delete_collection(collection_name)

        embed_model = get_embedding_model()
        embed_dim = get_embedding_dimension(embed_model)
        
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

        manifest[collection_name] = {
            "fingerprint": chunk_fingerprint,
            "child_count": len(child_chunks),
        }
        self._write_manifest(manifest)
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
        embed_model = get_embedding_model()
        qvec = embed_model.encode([_BGE_QUERY_PREFIX + query], normalize_embeddings=True)[0]
        
        qdrant_hits = self._qdrant.query_points(
            collection_name=self._collection_name,
            query=qvec.tolist(),
            limit=top_k * 5,
        ).points

        hit_id_to_emb_score = {int(h.id): float(h.score) for h in qdrant_hits}

        # Fuse with RRF (Reciprocal Rank Fusion)
        RRF_K = 60
        bm25_ranked_indices = np.argsort(bm25_raw)[::-1]
        bm25_ranks = {int(idx): r + 1 for r, idx in enumerate(bm25_ranked_indices)}
        
        qdrant_ranked_indices = [int(h.id) for h in qdrant_hits]
        qdrant_ranks = {idx: r + 1 for r, idx in enumerate(qdrant_ranked_indices)}

        rerank_k = min(top_k * 5, len(self._child_chunks))
        top_bm25_ids = set(bm25_ranked_indices[:rerank_k].tolist())
        top_qdrant_ids = set(qdrant_ranked_indices)
        candidate_ids = list(top_bm25_ids | top_qdrant_ids)

        w_bm25, w_emb = config.retrieval.bm25_weight, config.retrieval.embedding_weight
        child_scored = []
        for idx in candidate_ids:
            if idx >= len(self._child_chunks): continue
            
            bm25_rank = bm25_ranks.get(idx)
            bm25_rrf = 1.0 / (RRF_K + bm25_rank) if bm25_rank is not None else 0.0
            
            qdrant_rank = qdrant_ranks.get(idx)
            qdrant_rrf = 1.0 / (RRF_K + qdrant_rank) if qdrant_rank is not None else 0.0
            
            score = w_bm25 * bm25_rrf + w_emb * qdrant_rrf
            child_scored.append((idx, score))

        child_scored.sort(key=lambda x: x[1], reverse=True)

        # Map to parents
        seen_parents = set()
        parent_candidates = []
        rerank_window = max(top_k, config.retrieval.rerank_top_k)
        candidate_parent_limit = max(top_k * 4, rerank_window)
        for idx, score in child_scored:
            pid = self._child_chunks[idx]["parent_id"]
            if pid not in seen_parents:
                chunk = self._parents[pid]
                if section_filter and section_filter.lower() not in (chunk.metadata.section or "").lower():
                    continue
                seen_parents.add(pid)
                parent_candidates.append({
                    "chunk": chunk,
                    "score": score,
                    "bm25_score": float(bm25_norm[idx]),
                    "embedding_score": hit_id_to_emb_score.get(idx, 0.0)
                })
            if len(parent_candidates) >= candidate_parent_limit:
                break

        final_results = self._rerank_parent_candidates(query, parent_candidates, top_k)
        return final_results[:top_k]

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
            manifest = self._load_manifest()
            if name in manifest:
                del manifest[name]
                self._write_manifest(manifest)

    def _rerank_parent_candidates(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        rerank_window = min(len(candidates), max(top_k, config.retrieval.rerank_top_k))
        if rerank_window <= 0:
            return candidates[:top_k]

        try:
            cross_encoder = _get_cross_encoder()
            pairs = [(query, item["chunk"].text) for item in candidates[:rerank_window]]
            rerank_scores = cross_encoder.predict(pairs)
            for item, rerank_score in zip(candidates[:rerank_window], rerank_scores):
                item["rerank_score"] = float(rerank_score)
            candidates[:rerank_window] = sorted(
                candidates[:rerank_window],
                key=lambda item: (item.get("rerank_score", float("-inf")), item["score"]),
                reverse=True,
            )
        except Exception as exc:
            logger.warning("Cross-encoder reranking failed (%s). Using fused retrieval scores.", exc)

        return candidates[:top_k]

    def _load_manifest(self) -> Dict[str, Dict[str, Any]]:
        if self._in_memory or self._manifest_path is None:
            return {}
        if not self._manifest_path.exists():
            return {}
        try:
            return json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _write_manifest(self, manifest: Dict[str, Dict[str, Any]]) -> None:
        if self._in_memory or self._manifest_path is None:
            return   # no disk persistence in memory mode
        self._manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _fingerprint_chunks(self, chunks: List[Chunk]) -> str:
        payload = [
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "metadata": chunk.metadata.model_dump(),
            }
            for chunk in chunks
        ]
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()


def _min_max_normalize(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-9:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def expand_query(query: str) -> str:
    """
    Expand a query with domain-specific synonyms for government tender documents.

    Covers the primary terminology families found in Indian and international
    public-sector procurement: specifications, scope, commercial/legal terms,
    and common abbreviations used by CPWD, NHAI, MoRTH, PWD, etc.
    """
    _SYNONYMS: Dict[str, List[str]] = {
        # Technical specs
        "specification":    ["specs", "standard", "requirement", "parameter", "technical requirement"],
        "material":         ["grade", "alloy", "composition", "substance", "item"],
        "dimension":        ["size", "diameter", "length", "width", "thickness", "height"],
        "tolerance":        ["allowance", "deviation", "variation", "accuracy"],
        "capacity":         ["rating", "output", "throughput", "flow rate", "load"],
        "performance":      ["efficiency", "output", "yield", "productivity", "benchmark"],
        "standard":         ["IS", "BIS", "ASTM", "ISO", "IEC", "BS", "DIN", "code", "norm"],
        "test":             ["testing", "inspection", "quality check", "QC", "trial", "evaluation"],
        "installation":     ["erection", "commissioning", "setting up", "mounting", "fixing"],

        # Scope of work
        "scope":            ["scope of work", "work description", "SOW", "activities", "task", "job"],
        "deliverable":      ["output", "supply", "product", "milestone", "submission"],
        "timeline":         ["schedule", "completion date", "milestone", "period", "deadline", "duration"],
        "exclusion":        ["excluded", "not in scope", "outside scope", "not included", "exceptions"],
        "location":         ["site", "project site", "place", "premises", "area", "zone"],
        "obligation":       ["responsibility", "duty", "commitment", "liable", "shall", "must"],

        # Commercial / legal
        "tender":           ["bid", "NIT", "proposal", "RFP", "RFQ", "enquiry", "notice", "invitation"],
        "contract":         ["agreement", "work order", "purchase order", "PO", "LOI", "LOA"],
        "guarantee":        ["EMD", "earnest money", "security deposit", "bank guarantee", "performance guarantee", "PBG"],
        "penalty":          ["liquidated damages", "LD", "delay penalty", "deduction", "forfeiture"],
        "price":            ["rate", "cost", "amount", "quote", "BOQ", "bill of quantities", "tariff"],
        "payment":          ["billing", "invoice", "milestone payment", "advance", "retention", "mobilisation"],
        "eligibility":      ["qualification", "criteria", "pre-qualification", "PQ", "turnover", "net worth", "financial capacity"],
        "experience":       ["past work", "track record", "similar work", "credentials", "portfolio", "completed project"],
        "validity":         ["bid validity", "offer validity", "period", "duration"],
        "submission":       ["submission date", "last date", "due date", "closing date"],

        # Common abbreviations
        "quantity":         ["qty", "nos", "numbers", "units", "count"],
        "drawing":          ["DWG", "layout", "plan", "blueprint", "sketch"],
    }

    query_lower = query.lower()
    expansions: List[str] = []
    for term, syns in _SYNONYMS.items():
        if term in query_lower:
            for s in syns:
                if s.lower() not in query_lower and s not in expansions:
                    expansions.append(s)
                    if len(expansions) >= 4:   # cap additions per query to avoid context bloat
                        break
        if len(expansions) >= 6:
            break
    return (query + " " + " ".join(expansions)).strip() if expansions else query
