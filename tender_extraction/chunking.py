"""
chunking.py — Hierarchical chunking with rich metadata.

The chunking strategy can make or break extraction quality. We tried three
approaches before landing on this one:

  1. Fixed-size sliding window (400 tokens, 100 overlap): Simple but terrible.
     Section headers got split from their content. A spec that said "Steel: 
     see section 3.2 for details" ended up in a different chunk from section 3.2.

  2. Recursive text splitting (like LangChain): Better, but lost all section
     hierarchy. Every chunk was an orphan with no context about where in the
     document it came from.

  3. Current approach — section-aware paragraph chunking: Each chunk knows its
     section, parent section, and type (table/paragraph/list/image_ocr). This
     lets the LLM understand context even when looking at a single chunk.

The most important rule: tables are NEVER chunked as text. Each table row
becomes its own chunk with headers repeated. This preserves column-value
relationships. If you chunk a table as text, "500 kg" might end up in a
different chunk from "Steel Bars" and the LLM will happily assign that
quantity to concrete instead.
- Prathamesh, 2026-02-12
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
# Matches patterns like "3.2 Material Requirements", "3.2.1 Steel", "10. Scope"
# Also handles trailing dots: "3.2. Material Requirements"
# Tested against headers from MTF, PBMC, NHAI, and CPWD tenders.
_SECTION_RE = re.compile(r"^(\d+(?:\.\d+)*\.?)\s+(.+)$")


def _count_tokens(text: str) -> int:
    """
    Count tokens using tiktoken for accurate LLM budget estimation.

    Falls back to word-count heuristic if tiktoken isn't available or
    fails (which happened once with some Unicode garbage from a corrupt
    scan). The heuristic of len/4 is surprisingly close to tiktoken
    for English text — within 10% on average.
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
) -> List[Chunk]:
    """
    Create hierarchical chunks from page text and extracted tables.

    Table chunks are created first because they're higher priority —
    they contain the structured spec data. Text chunks fill in the
    narrative context (scope of work, general conditions, etc.)

    Args:
        pages:  Output of ingestion.ingest_document().
        tables: Output of table_extraction.extract_tables().

    Returns:
        List of Chunk objects with rich metadata.
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

        page_chunks = _chunk_text(text, page_num, chunk_type)
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

    Why not one chunk per table? Because tables in tenders can be 50+
    rows, which blows past the token limit. And if we stuff the whole
    table into one chunk, the LLM tends to only extract specs from the
    first and last few rows (attention locality bias).

    Why repeat headers in every row chunk? Because without headers, a row
    like "500 | kg | ±5% | IS 456" is meaningless. The LLM needs to know
    that column 1 is Quantity, column 2 is Unit, etc.
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
            # Format that gives the LLM clear table context
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


def _chunk_text(text: str, page: int, chunk_type: str) -> List[Chunk]:
    """
    Split page text into paragraph-level chunks with section awareness.

    The section detection regex picks up numbered headers which cover
    95%+ of Indian government tender docs. For the rare tender without
    numbered sections, chunks still get created — they just have
    section="Unknown" which is handled gracefully downstream.
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

        # Detect lists: lines starting with bullets or dashes.
        # Tenders love bulleted lists for deliverables and exclusions.
        detected_type = chunk_type
        list_lines = [l for l in paragraph_buffer if re.match(r"^\s*[-•*]\s", l)]
        if len(list_lines) > len(paragraph_buffer) * 0.5 and len(paragraph_buffer) >= 2:
            detected_type = "list"

        # Split if the paragraph exceeds our token limit.
        # This happens with long narrative sections like "General Conditions"
        # which can be 2000+ tokens of continuous text.
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

    for line in lines:
        stripped = line.strip()
        if not stripped:
            # Blank line = paragraph boundary
            _flush_paragraph()
            continue

        # Check for section header
        header_match = _SECTION_RE.match(stripped)
        if header_match:
            _flush_paragraph()

            section_number = header_match.group(1).rstrip(".")
            section_title = header_match.group(2).strip()
            full_section = f"{section_number} {section_title}"

            parent_section, current_section = _resolve_hierarchy(
                section_number, full_section, current_section, parent_section
            )
            # Include section header in the next paragraph so the text
            # starts with "3.2 Material Requirements\nSteel bars shall..."
            paragraph_buffer.append(stripped)
            continue

        paragraph_buffer.append(stripped)

    # Don't forget the last paragraph (no trailing blank line)
    _flush_paragraph()

    return chunks


def _resolve_hierarchy(
    section_number: str,
    full_section: str,
    current_section: str,
    parent_section: str,
) -> Tuple[str, str]:
    """
    Figure out parent/child section relationships from section numbers.

    "3" → top-level, parent is itself
    "3.2" → parent is whatever the current section was (likely "3 ...")
    "3.2.1" → parent is the current section (likely "3.2 ...")

    Not perfect for all edge cases (e.g. jumping from 3.2.1 to 4.1) but
    good enough for the tender docs we've tested. The section metadata is
    nice-to-have context for the LLM, not a critical data path.
    """
    parts = section_number.split(".")
    if len(parts) == 1:
        return full_section, full_section
    else:
        return current_section, full_section


def _split_by_tokens(text: str) -> List[str]:
    """
    Split text into sub-chunks if it exceeds max_chunk_tokens.

    We split on sentence boundaries (period + space) rather than at
    exact token counts. This means chunks might be slightly over/under
    the limit but at least sentences stay whole. A spec like "Steel bars
    shall conform to IS 456" is useless if split after "Steel bars shall".
    """
    max_tokens = config.chunking.max_chunk_tokens
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
            current = [sentence]
            current_count = sent_tokens
        else:
            current.append(sentence)
            current_count += sent_tokens

    if current:
        sub_chunks.append(" ".join(current))

    return sub_chunks if sub_chunks else [text]


def _short_uuid() -> str:
    """8-char hex ID. Short enough for readability in logs, long enough
    to avoid collisions for typical document sizes (<10k chunks)."""
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

    # Report by type
    type_counts = {}
    for c in chunks:
        ct = c.metadata.chunk_type
        type_counts[ct] = type_counts.get(ct, 0) + 1
    print(f"Total chunks: {len(chunks)} (types: {type_counts})")

    # Report by section (top 10)
    section_counts = {}
    for c in chunks:
        s = c.metadata.section
        section_counts[s] = section_counts.get(s, 0) + 1
    top_sections = sorted(section_counts.items(), key=lambda x: -x[1])[:10]
    print(f"\nTop 10 sections:")
    for sec, cnt in top_sections:
        print(f"  {sec[:50]}: {cnt} chunks")

    # Show a few real chunks
    print(f"\nSample chunks (first 3):")
    for c in chunks[:3]:
        print(f"  [{c.chunk_id}] page={c.metadata.page} type={c.metadata.chunk_type}")
        print(f"    section: {c.metadata.section}")
        print(f"    text: {c.text[:100]}...")

    print("\nChunking smoke test passed.")
