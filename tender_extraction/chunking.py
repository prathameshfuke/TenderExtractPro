"""
chunking.py — Hierarchical chunking with rich metadata.

Implements section-aware paragraph chunking where each chunk 
retains its section, parent section, and type (table/paragraph/list).
This preserves context for the LLM.

Tables are never chunked as text. Each table row becomes its own chunk
with headers repeated to preserve column-value relationships.
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from tender_extraction.config import config
from tender_extraction.schemas import Chunk, ChunkMetadata

logger = logging.getLogger(__name__)

# Regex for numbered section headers commonly found in Indian government tenders.
_SECTION_RE = re.compile(r"^(\d+(?:\.\d+)*\.?)\s+(.+)$")


def _count_tokens(text: str) -> int:
    """
    Count tokens using tiktoken for accurate LLM budget estimation.
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding(config.chunking.tiktoken_model)
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def create_chunks(
    pages: List[Dict[str, Any]],
    tables: Optional[List[Dict[str, Any]]] = None,
    use_semantic: bool = True,
) -> List[Chunk]:
    """
    Create hierarchical chunks from page text and extracted tables.
    """
    chunks: List[Chunk] = []

    # Tables first — they're the most valuable data source
    if tables:
        table_chunks = _chunk_tables(tables)
        chunks.extend(table_chunks)

    # Then text from each page
    for page_data in pages:
        page_num = page_data["page"]
        text = page_data["text"]
        is_ocr = page_data.get("is_ocr", False)
        chunk_type = "image_ocr" if is_ocr else "paragraph"
        heading_hints = page_data.get("headings", [])

        page_chunks = _chunk_text(
            text, page_num, chunk_type, 
            heading_hints=heading_hints,
            use_semantic=use_semantic
        )
        chunks.extend(page_chunks)

    logger.info(
        "Created %d total chunks (%d table, %d text/ocr)",
        len(chunks),
        sum(1 for c in chunks if c.metadata.chunk_type == "table"),
        sum(1 for c in chunks if c.metadata.chunk_type != "table"),
    )
    return chunks


def _chunk_tables(tables: List[Dict[str, Any]]) -> List[Chunk]:
    """
    Each table row becomes a separate chunk with the full header row
    prepended for context.
    """
    chunks: List[Chunk] = []

    for table in tables:
        headers = table.get("headers", [])
        rows = table.get("rows", [])
        table_id = table.get("table_id", "table_unknown")
        page = table.get("page", 0)
        bbox = table.get("bbox")

        header_line = " | ".join(headers)

        for row_idx, row in enumerate(rows):
            if not any(cell.strip() for cell in row):
                continue

            row_line = " | ".join(row)
            text = f"[Table Headers]: {header_line}\n[Row {row_idx + 1}]: {row_line}"

            chunk_id = f"{table_id}_row_{row_idx + 1}_{_short_uuid()}"
            metadata = ChunkMetadata(
                section="Table",
                parent_section="Document Tables",
                chunk_type="table",
                page=page,
                table_id=table_id,
                bbox=bbox,
            )
            chunks.append(Chunk(chunk_id=chunk_id, text=text, metadata=metadata))

    logger.info("Created %d table-row chunks from %d tables", len(chunks), len(tables))
    return chunks


def _chunk_text(
    text: str,
    page: int,
    chunk_type: str,
    heading_hints: Optional[List[str]] = None,
    use_semantic: bool = True,
) -> List[Chunk]:
    """
    Split page text into paragraph-level chunks with section awareness.
    """
    chunks: List[Chunk] = []
    lines = text.split("\n")

    current_section = "Unknown"
    parent_section = "Unknown"
    paragraph_buffer: List[str] = []

    def _flush_paragraph():
        nonlocal current_section, parent_section
        if not paragraph_buffer:
            return
        para_text = "\n".join(paragraph_buffer).strip()
        if not para_text:
            paragraph_buffer.clear()
            return

        detected_type = chunk_type
        list_lines = [l for l in paragraph_buffer if re.match(r"^\s*[-•*]\s", l)]
        if len(list_lines) > len(paragraph_buffer) * 0.5 and len(paragraph_buffer) >= 2:
            detected_type = "list"

        if use_semantic and detected_type in ("paragraph", "list", "image_ocr"):
            sub_texts = _semantic_split_text(para_text)
        else:
            sub_texts = _split_by_tokens(para_text)

        for sub_text in sub_texts:
            chunk_id = f"chunk_{page}_{_short_uuid()}"
            metadata = ChunkMetadata(
                section=current_section,
                parent_section=parent_section,
                chunk_type=detected_type,  # type: ignore[arg-type]
                page=page,
            )
            chunks.append(Chunk(chunk_id=chunk_id, text=sub_text, metadata=metadata))

        paragraph_buffer.clear()

    heading_set = set(heading_hints or [])

    for line in lines:
        stripped = line.strip()
        if not stripped:
            _flush_paragraph()
            continue

        header_match = _SECTION_RE.match(stripped)
        is_pymupdf_heading = (not header_match) and (stripped in heading_set)

        if header_match:
            _flush_paragraph()
            section_number = header_match.group(1).rstrip(".")
            section_title = header_match.group(2).strip()
            full_section = f"{section_number} {section_title}"
            parent_section, current_section = _resolve_hierarchy(
                section_number, full_section, current_section, parent_section
            )
            paragraph_buffer.append(stripped)
            continue

        if is_pymupdf_heading:
            _flush_paragraph()
            current_section = stripped
            parent_section = stripped
            paragraph_buffer.append(stripped)
            continue

        paragraph_buffer.append(stripped)

    _flush_paragraph()
    return chunks


# Singleton model cache for semantic chunking
_semantic_embed_model = None

def _get_semantic_embed_model():
    global _semantic_embed_model
    if _semantic_embed_model is None:
        from langchain_community.embeddings import HuggingFaceBgeEmbeddings
        logger.info("Loading embedding model for semantic chunking: %s", config.retrieval.embedding_model)
        _semantic_embed_model = HuggingFaceBgeEmbeddings(
            model_name=config.retrieval.embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    return _semantic_embed_model


def _semantic_split_text(text: str) -> List[str]:
    """
    Use LangChain's SemanticChunker to split text based on embedding distances.
    """
    try:
        from langchain_experimental.text_splitter import SemanticChunker
        
        embed_model = _get_semantic_embed_model()
        
        splitter = SemanticChunker(
            embed_model, 
            breakpoint_threshold_type="gradient",
        )
        
        docs = splitter.create_documents([text])
        sub_texts = [doc.page_content for doc in docs]
        
        final_texts = []
        for t in sub_texts:
            if _count_tokens(t) > config.chunking.max_chunk_tokens * 1.5:
                final_texts.extend(_split_by_tokens(t))
            else:
                final_texts.append(t)
        return final_texts
        
    except Exception as exc:
        logger.warning("Semantic chunking failed (%s). Falling back to token-based splitting.", exc)
        return _split_by_tokens(text)


def _resolve_hierarchy(
    section_number: str,
    full_section: str,
    current_section: str,
    parent_section: str,
) -> Tuple[str, str]:
    parts = section_number.split(".")
    if len(parts) == 1:
        return full_section, full_section
    else:
        return current_section, full_section


def _split_by_tokens(text: str) -> List[str]:
    max_tokens = config.chunking.max_chunk_tokens
    overlap_tokens = config.chunking.overlap_tokens
    token_count = _count_tokens(text)

    if token_count <= max_tokens:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    sub_chunks: List[str] = []
    current: List[str] = []
    current_count = 0

    for sentence in sentences:
        sent_tokens = _count_tokens(sentence)
        if current_count + sent_tokens > max_tokens and current:
            sub_chunks.append(" ".join(current))
            overlap_sents: List[str] = []
            overlap_count = 0
            for s in reversed(current):
                s_toks = _count_tokens(s)
                if overlap_count + s_toks > overlap_tokens and overlap_sents:
                    break
                overlap_sents.insert(0, s)
                overlap_count += s_toks
            current = overlap_sents + [sentence]
            current_count = overlap_count + sent_tokens
        else:
            current.append(sentence)
            current_count += sent_tokens

    if current:
        sub_chunks.append(" ".join(current))

    return sub_chunks if sub_chunks else [text]


def _short_uuid() -> str:
    return uuid.uuid4().hex[:8]


if __name__ == "__main__":
    import sys
    from pathlib import Path
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tender_extraction.ingestion import ingest_document
    from tender_extraction.table_extraction import extract_tables

    pdf = "dataset/globaltender1576.pdf"
    if not Path(pdf).exists():
        print(f"Dataset file not found: {pdf}")
        sys.exit(1)

    print(f"Ingesting {pdf} ...")
    pages = ingest_document(pdf)
    tables = extract_tables(pdf)

    print(f"Chunking {len(pages)} pages + {len(tables)} tables ...")
    chunks = create_chunks(pages, tables)
    print(f"Total chunks: {len(chunks)}")
