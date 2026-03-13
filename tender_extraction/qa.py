from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from tender_extraction.chunking import create_chunks
from tender_extraction.extraction import answer_question
from tender_extraction.ingestion import ingest_document
from tender_extraction.main import discover_document_topic
from tender_extraction.retrieval import HybridRetriever, expand_query
from tender_extraction.table_extraction import extract_tables


class DocumentChatSession:
    """Cached retrieval session for grounded document Q&A."""

    def __init__(self, file_path: str, persist_dir: Optional[str] = None, force_reindex: bool = False):
        self.file_path = str(file_path)
        self.persist_dir = persist_dir
        self.force_reindex = force_reindex
        self.topic = ""
        self._retriever: Optional[HybridRetriever] = None
        self._ready = False

    def build(self) -> "DocumentChatSession":
        if self._ready:
            return self

        pages = ingest_document(self.file_path)
        tables = extract_tables(self.file_path) if Path(self.file_path).suffix.lower() == ".pdf" else []
        chunks = create_chunks(pages, tables)
        topic = discover_document_topic(pages)

        retriever = HybridRetriever(persist_dir=self.persist_dir)
        digest = hashlib.md5(Path(self.file_path).resolve().as_posix().encode("utf-8")).hexdigest()[:10]
        collection_name = f"qa_{Path(self.file_path).stem[:20]}_{digest}"
        retriever.build_index(chunks, collection_name=collection_name, force_rebuild=self.force_reindex)

        self.topic = topic
        self._retriever = retriever
        self._ready = True
        return self

    def ask(self, question: str) -> Dict[str, Any]:
        self.build()
        assert self._retriever is not None

        queries: List[str] = [question, expand_query(question)]
        if self.topic:
            queries.append(f"{self.topic} {question}")

        seen = set()
        merged: List[Dict[str, Any]] = []
        for query in queries:
            for hit in self._retriever.retrieve_question_chunks(query, top_k=8):
                chunk_id = hit["chunk"].chunk_id
                if chunk_id in seen:
                    continue
                seen.add(chunk_id)
                merged.append(hit)

        merged.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        return answer_question(merged[:8], question, self.topic)

    def close(self) -> None:
        if self._retriever is not None:
            self._retriever.close()
            self._retriever = None
        self._ready = False