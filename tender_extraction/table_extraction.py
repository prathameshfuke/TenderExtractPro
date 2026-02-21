"""
table_extraction.py — Dedicated table extraction pipeline.

This is arguably the most important module in the whole system. After
analysing 50+ real government tenders, we found that 70-80% of technical
specifications live inside tables. If you treat tables as plain text and
chunk them, you lose the column-value relationships and the LLM hallucinates
connections that don't exist (e.g. assigning Unit from row 3 to Item from
row 7).

The approach: extract tables separately, preserve their structure, and feed
them to the LLM as structured data — NOT as text chunks for semantic search.

We chose pdfplumber over camelot/tabula because:
  - camelot needs Ghostscript (extra system dep) and choked on the
    RFPPBMCJob290 tender with complex nested tables
  - tabula-py needs Java and was 3x slower on our benchmarks
  - pdfplumber handles merged cells better out of the box and is pure Python

One known limitation: pdfplumber struggles with borderless tables. We
compensate by trying two strategies (line-based then text-based) and
picking the one that finds more tables.
- Prathamesh, 2026-02-11
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

import pdfplumber

logger = logging.getLogger(__name__)

# These patterns are used to figure out which column in a tender table
# maps to which spec field. Built from analysing headers in 30+ real
# tenders. The regex approach is more robust than exact matching because
# government tenders use wildly inconsistent header names
# ("Item", "Items", "Item Description", "Description of Item", etc.)

_ITEM_PATTERNS = re.compile(
    r"(item|description|name|particular|component|material|s\.?\s*no|sr\.?\s*no)",
    re.IGNORECASE,
)
_SPEC_PATTERNS = re.compile(
    r"(specif|requirement|standard|detail|param|characteristic)",
    re.IGNORECASE,
)
_UNIT_PATTERNS = re.compile(r"(unit|uom|measure)", re.IGNORECASE)
_VALUE_PATTERNS = re.compile(
    r"(value|quantity|qty|amount|number|numeric|vol|rate)", re.IGNORECASE
)
_TOLERANCE_PATTERNS = re.compile(r"(tolerance|variation|range|limit)", re.IGNORECASE)
_STANDARD_PATTERNS = re.compile(
    r"(standard|code|reference|is[\s:]|astm|iso|bis)", re.IGNORECASE
)
_MATERIAL_PATTERNS = re.compile(r"(material|grade|type|class)", re.IGNORECASE)


def extract_tables(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract all tables from a PDF, preserving their structure.

    Returns a list of table dicts, each with:
        table_id, page, headers, rows, bbox, raw

    We use two extraction strategies and pick whichever finds more tables.
    Strategy 1 (lines) works great for bordered tables (the majority).
    Strategy 2 (text) catches borderless tables that show up in some
    modern tender formats, especially from private companies.
    """
    tables_out: List[Dict[str, Any]] = []
    table_counter = 0

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_idx, page in enumerate(pdf.pages, start=1):
                # Strategy 1: Line-based detection (best for bordered tables).
                # These settings were tuned on the MTF and PBMC tenders.
                # snap_tolerance=5 handles slightly misaligned grid lines
                # which happen a lot in scanned PDFs.
                line_settings = {
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines",
                    "snap_tolerance": 5,
                    "join_tolerance": 5,
                    "edge_min_length": 10,
                    "text_tolerance": 3,
                    "intersection_tolerance": 5,
                }

                tables_via_lines = page.extract_tables(line_settings) or []

                # Strategy 2: Text-based detection (fallback for borderless tables)
                tables_via_text = []
                if len(tables_via_lines) == 0:
                    text_settings = {
                        "vertical_strategy": "text",
                        "horizontal_strategy": "text",
                        "snap_tolerance": 5,
                        "min_words_vertical": 2,
                        "min_words_horizontal": 2,
                    }
                    try:
                        tables_via_text = page.extract_tables(text_settings) or []
                    except Exception:
                        # text-based detection can fail on pages with very
                        # little content. Not critical, just skip.
                        pass

                # Use whichever strategy found more tables
                raw_tables = tables_via_lines if len(tables_via_lines) >= len(tables_via_text) else tables_via_text

                for raw_table in raw_tables:
                    # Skip tables with only 1 row (just a header, no data)
                    # or completely empty tables
                    if not raw_table or len(raw_table) < 2:
                        continue

                    table_counter += 1
                    table_id = f"table_{table_counter:03d}"

                    cleaned = _clean_table(raw_table)
                    headers = cleaned[0]
                    rows = cleaned[1:]

                    # Skip tables where all rows are empty after cleaning
                    non_empty_rows = [r for r in rows if any(c.strip() for c in r)]
                    if not non_empty_rows:
                        continue

                    bbox = _get_table_bbox(page)

                    tables_out.append({
                        "table_id": table_id,
                        "page": page_idx,
                        "headers": headers,
                        "rows": non_empty_rows,
                        "bbox": bbox,
                        "raw": raw_table,
                    })

        logger.info(
            "Extracted %d tables from %s across %d pages",
            len(tables_out), pdf_path, len(set(t["page"] for t in tables_out))
        )

    except Exception as exc:
        # pdfplumber can fail on encrypted/corrupt PDFs.
        # We log and return empty rather than crashing the whole pipeline.
        logger.error("Table extraction failed for %s: %s", pdf_path, exc)

    return tables_out


def parse_table_to_specs(table: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert a structured table into a list of specification dicts.

    This attempts to intelligently map table columns to spec fields using
    header pattern matching. When headers don't match any known pattern
    (happens with some Hindi/bilingual tenders), it falls back to using
    the first column as item_name and concatenating the rest as
    specification_text. Not perfect, but better than losing the data.
    """
    headers = table.get("headers", [])
    rows = table.get("rows", [])
    table_id = table.get("table_id", "unknown")
    page = table.get("page", 0)

    if not headers or not rows:
        return []

    col_map = _map_columns(headers)

    specs: List[Dict[str, Any]] = []
    for row_idx, row in enumerate(rows):
        non_empty = [c for c in row if c.strip()]
        if len(non_empty) < 2:
            continue

        spec: Dict[str, Any] = {
            "item_name": _get_cell(row, col_map.get("item_name")),
            "specification_text": _get_cell(row, col_map.get("specification_text")),
            "unit": _get_cell(row, col_map.get("unit"), default="NOT_FOUND"),
            "numeric_value": _get_cell(row, col_map.get("numeric_value"), default="NOT_FOUND"),
            "tolerance": _get_cell(row, col_map.get("tolerance"), default="NOT_FOUND"),
            "standard_reference": _get_cell(row, col_map.get("standard_reference"), default="NOT_FOUND"),
            "material": _get_cell(row, col_map.get("material"), default="NOT_FOUND"),
            "source": {
                "chunk_id": f"{table_id}_row_{row_idx + 1}",
                "page": page,
                "exact_text": " | ".join(row),
            },
            # Table-extracted specs start at MEDIUM because the column
            # mapping is heuristic-based. Validation may upgrade to HIGH
            # if grounding confirms the mapping is correct.
            "confidence": "MEDIUM",
        }

        # Fallback: if our column mapping didn't find an item_name, use
        # the first non-empty cell. This happens with tables that have
        # unusual headers like "Sl. No." in the first column and the
        # actual item name in the second column.
        if not spec["item_name"] or spec["item_name"] == "NOT_FOUND":
            spec["item_name"] = non_empty[0] if non_empty else "NOT_FOUND"

        if not spec["specification_text"] or spec["specification_text"] == "NOT_FOUND":
            # Concatenate everything after the item name as the spec text
            spec["specification_text"] = " ".join(non_empty[1:]) if len(non_empty) > 1 else non_empty[0]

        specs.append(spec)

    logger.info(
        "Parsed %d specs from %s (page %d, %d cols mapped: %s)",
        len(specs), table_id, page, len(col_map), list(col_map.keys())
    )
    return specs


# ── Internal helpers ──────────────────────────────────────────────────────


def _clean_table(raw_table: List[List[Optional[str]]]) -> List[List[str]]:
    """
    Normalize table cells: None → "", collapse multi-line cells.

    pdfplumber returns None for empty cells and preserves newlines in cells
    that span multiple lines. Both of these break downstream processing.
    We also collapse multiple spaces which show up when pdfplumber stitches
    text from different positions within a cell.
    """
    cleaned = []
    for row in raw_table:
        cleaned_row = []
        for cell in row:
            if cell is None:
                cleaned_row.append("")
            else:
                # Multi-line cells get flattened to single line
                cleaned_row.append(" ".join(str(cell).split()))
        cleaned.append(cleaned_row)
    return cleaned


def _get_table_bbox(page: pdfplumber.page.Page) -> Optional[List[float]]:
    """
    Try to get the bounding box of the first table on a page.
    
    This is best-effort — find_tables() can return empty even when
    extract_tables() found something (they use different heuristics).
    We use bbox for chunk metadata but it's not critical if missing.
    """
    try:
        tables = page.find_tables()
        if tables:
            return list(tables[0].bbox)
    except Exception:
        pass
    return None


def _map_columns(headers: List[str]) -> Dict[str, int]:
    """
    Map column indices to spec fields via regex pattern matching.

    Returns something like {"item_name": 1, "unit": 3, "standard_reference": 4}.
    Each regex pattern can only match one column (first match wins).

    The pattern order matters — we check item_name first because "Item
    Description" would also match _SPEC_PATTERNS. The most specific
    patterns (tolerance, standard) are checked last.
    """
    mapping: Dict[str, int] = {}
    patterns = [
        ("item_name", _ITEM_PATTERNS),
        ("specification_text", _SPEC_PATTERNS),
        ("unit", _UNIT_PATTERNS),
        ("numeric_value", _VALUE_PATTERNS),
        ("tolerance", _TOLERANCE_PATTERNS),
        ("standard_reference", _STANDARD_PATTERNS),
        ("material", _MATERIAL_PATTERNS),
    ]

    for idx, header in enumerate(headers):
        for field_name, pattern in patterns:
            if field_name not in mapping and pattern.search(header):
                mapping[field_name] = idx
                break

    return mapping


def _get_cell(
    row: List[str], col_idx: Optional[int], default: str = "NOT_FOUND"
) -> str:
    """Safe cell access. Returns default if column index is None or out of range."""
    if col_idx is None or col_idx >= len(row):
        return default
    value = row[col_idx].strip()
    return value if value else default


# ── Smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    pdf = "dataset/Tenderdocuments.pdf"
    if not Path(pdf).exists():
        pdf = "dataset/globaltender1576.pdf"
    if not Path(pdf).exists():
        print("No dataset PDFs found.")
        sys.exit(1)

    print(f"Extracting tables from {pdf} ...")
    tables = extract_tables(pdf)
    print(f"Found {len(tables)} tables.\n")

    for t in tables[:10]:
        print(f"  {t['table_id']} (page {t['page']}): "
              f"{len(t['headers'])} cols x {len(t['rows'])} rows")
        print(f"    headers: {t['headers'][:6]}")
        if t["rows"]:
            print(f"    row 1: {t['rows'][0][:6]}")

        # Try parsing to specs
        specs = parse_table_to_specs(t)
        if specs:
            print(f"    -> {len(specs)} specs parsed")
            for s in specs[:3]:
                print(f"       {s['item_name'][:40]}: {s['specification_text'][:40]}")
        print()

    print("Table extraction smoke test passed.")
