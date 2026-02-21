"""
test_pipeline.py — Tests for TenderExtractPro.

Covers core logic that doesn't need the LLM model:
  - Pydantic schema enforcement and defaults
  - Chunking logic with synthetic and real data
  - Table column mapping against common tender header formats
  - Grounding verification and confidence scoring
  - Real dataset PDF ingestion and table extraction

Run with:
    python tests/test_pipeline.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tender_extraction.schemas import (
    Chunk, ChunkMetadata, ExtractionResult,
    TechnicalSpecification, SourceCitation,
    ScopeTask, ScopeOfWork,
)
from tender_extraction.chunking import create_chunks
from tender_extraction.table_extraction import _map_columns, _clean_table, extract_tables
from tender_extraction.validation import (
    verify_grounding, assign_confidence,
    validate_extractions, _enforce_not_found,
)
from tender_extraction.ingestion import ingest_document

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"


# -- Schema tests --

def test_spec_schema_defaults():
    spec = TechnicalSpecification(
        item_name="Steel Bars",
        specification_text="Grade 60 steel bars",
        source=SourceCitation(chunk_id="chunk_001", page=5),
    )
    assert spec.unit == "NOT_FOUND"
    assert spec.tolerance == "NOT_FOUND"
    assert spec.confidence == "LOW"
    print("  PASS: test_spec_schema_defaults")


def test_empty_spec_text_rejected():
    try:
        TechnicalSpecification(
            item_name="Bad", specification_text="  ",
            source=SourceCitation(chunk_id="x", page=1),
        )
        print("  FAIL: test_empty_spec_text_rejected (should have raised)")
        return False
    except Exception:
        print("  PASS: test_empty_spec_text_rejected")
    return True


def test_extraction_result_roundtrip():
    result = ExtractionResult(
        technical_specifications=[
            TechnicalSpecification(
                item_name="Cement", specification_text="OPC Grade 53",
                unit="MT", standard_reference="IS 12269",
                source=SourceCitation(chunk_id="c1", page=15),
                confidence="HIGH",
            )
        ],
        scope_of_work=ScopeOfWork(
            tasks=[ScopeTask(
                task_description="Site prep",
                source=SourceCitation(chunk_id="c2", page=8),
            )],
        ),
    )
    data = result.model_dump()
    assert len(data["technical_specifications"]) == 1
    assert data["scope_of_work"]["tasks"][0]["timeline"] == "NOT_FOUND"
    print("  PASS: test_extraction_result_roundtrip")


# -- Chunking tests --

def test_chunking_with_sections():
    pages = [{
        "page": 1,
        "text": "1 Introduction\nThis tender is for highway construction.\n\n"
                "2 Technical Specifications\n2.1 Material Requirements\n"
                "Steel bars shall conform to ASTM A615 Grade 60.\n",
        "is_ocr": False,
    }]
    chunks = create_chunks(pages, tables=None)
    assert len(chunks) >= 2
    sections = {c.metadata.section for c in chunks}
    assert any("Introduction" in s or "Technical" in s or "Material" in s for s in sections)
    print(f"  PASS: test_chunking_with_sections ({len(chunks)} chunks)")


def test_chunking_table_rows():
    tables = [{
        "table_id": "table_001", "page": 5,
        "headers": ["Item", "Specification", "Unit"],
        "rows": [["Steel Bars", "ASTM A615", "kg"], ["Cement", "IS 12269", "MT"]],
        "bbox": [100, 200, 500, 400],
    }]
    chunks = create_chunks(pages=[], tables=tables)
    assert len(chunks) == 2
    for c in chunks:
        assert c.metadata.chunk_type == "table"
        assert "[Table Headers]" in c.text
    print(f"  PASS: test_chunking_table_rows ({len(chunks)} chunks)")


# -- Table column mapping tests --

def test_column_mapping_standard():
    headers = ["Sr. No.", "Item Description", "Specification Details", "Unit of Measure", "Quantity"]
    mapping = _map_columns(headers)
    assert "item_name" in mapping
    assert "specification_text" in mapping
    assert "unit" in mapping
    print(f"  PASS: test_column_mapping_standard: {mapping}")


def test_column_mapping_alternate():
    headers = ["Sl No", "Name of Material", "Required Quantity", "Rate"]
    mapping = _map_columns(headers)
    assert "item_name" in mapping
    print(f"  PASS: test_column_mapping_alternate: {mapping}")


def test_clean_table():
    raw = [["Header 1", None, "Header 3"], ["Cell\nwith\nnewlines", "", "  Value  "]]
    cleaned = _clean_table(raw)
    assert cleaned[0][1] == ""
    assert "\n" not in cleaned[1][0]
    print("  PASS: test_clean_table")


# -- Grounding tests --

def test_grounding_exact_match():
    chunk = Chunk(
        chunk_id="c1",
        text="Steel bars Grade 60 conforming to ASTM A615.",
        metadata=ChunkMetadata(page=15),
    )
    spec = {
        "source": {"exact_text": "Steel bars Grade 60 conforming to ASTM A615"}
    }
    score = verify_grounding(spec, [{"chunk": chunk, "score": 0.9}])
    assert score >= 0.95
    print(f"  PASS: test_grounding_exact_match (score={score:.3f})")


def test_grounding_rejects_hallucination():
    chunk = Chunk(
        chunk_id="c1",
        text="The deadline for submission of tenders is January 15th.",
        metadata=ChunkMetadata(page=1),
    )
    spec = {
        "source": {"exact_text": "Concrete pump rental agreement terms and conditions"}
    }
    score = verify_grounding(spec, [{"chunk": chunk, "score": 0.9}])
    assert score < 0.4
    print(f"  PASS: test_grounding_rejects_hallucination (score={score:.3f})")


def test_confidence_mapping():
    assert assign_confidence(0.95) == "HIGH"
    assert assign_confidence(0.75) == "MEDIUM"
    assert assign_confidence(0.30) == "LOW"
    print("  PASS: test_confidence_mapping")


def test_enforce_not_found():
    data = {"unit": "", "tolerance": None, "material": "Steel"}
    result = _enforce_not_found(data)
    assert result["unit"] == "NOT_FOUND"
    assert result["tolerance"] == "NOT_FOUND"
    assert result["material"] == "Steel"
    print("  PASS: test_enforce_not_found")


def test_validation_rejects_hallucinated():
    source_chunks = [{
        "chunk": Chunk(
            chunk_id="c1",
            text="Steel reinforcement bars shall be Grade 60 conforming to ASTM A615",
            metadata=ChunkMetadata(page=15),
        ),
        "score": 0.9,
    }]
    extraction = {
        "technical_specifications": [
            {
                "item_name": "Steel Bars",
                "specification_text": "Grade 60 ASTM A615",
                "source": {"chunk_id": "c1", "page": 15,
                           "exact_text": "Steel reinforcement bars shall be Grade 60 conforming to ASTM A615"},
            },
            {
                "item_name": "Hallucinated",
                "specification_text": "This was invented",
                "source": {"chunk_id": "fake", "page": 99,
                           "exact_text": "Completely fabricated text"},
            },
        ],
        "scope_of_work": {"tasks": [], "exclusions": []},
    }
    result = validate_extractions(extraction, source_chunks)
    specs = result["technical_specifications"]
    assert len(specs) == 1
    assert specs[0]["item_name"] == "Steel Bars"
    print(f"  PASS: test_validation_rejects_hallucinated (1 accepted, 1 rejected)")


# -- Real dataset tests --

def test_real_pdf_ingestion():
    pdf_path = DATASET_DIR / "globaltender1576.pdf"
    if not pdf_path.exists():
        print("  SKIP: test_real_pdf_ingestion (dataset not found)")
        return
    pages = ingest_document(str(pdf_path))
    assert len(pages) > 0
    pages_with_text = [p for p in pages if len(p["text"].strip()) > 50]
    assert len(pages_with_text) > 0
    total_chars = sum(len(p["text"]) for p in pages)
    print(f"  PASS: test_real_pdf_ingestion ({len(pages)} pages, {total_chars:,} chars)")


def test_real_pdf_table_extraction():
    pdf_path = DATASET_DIR / "Tenderdocuments.pdf"
    if not pdf_path.exists():
        print("  SKIP: test_real_pdf_table_extraction (dataset not found)")
        return
    tables = extract_tables(str(pdf_path))
    print(f"  PASS: test_real_pdf_table_extraction ({len(tables)} tables)")


def test_real_pdf_chunking():
    pdf_path = DATASET_DIR / "globaltender1576.pdf"
    if not pdf_path.exists():
        print("  SKIP: test_real_pdf_chunking (dataset not found)")
        return
    pages = ingest_document(str(pdf_path))
    tables = extract_tables(str(pdf_path))
    chunks = create_chunks(pages, tables)
    assert len(chunks) > 0
    types = {}
    for c in chunks:
        types[c.metadata.chunk_type] = types.get(c.metadata.chunk_type, 0) + 1
    print(f"  PASS: test_real_pdf_chunking ({len(chunks)} chunks, types: {types})")


# -- Runner --

def run_all_tests():
    print("\n" + "=" * 60)
    print("  TenderExtractPro -- Test Suite")
    print("=" * 60 + "\n")

    tests = [
        test_spec_schema_defaults,
        test_empty_spec_text_rejected,
        test_extraction_result_roundtrip,
        test_chunking_with_sections,
        test_chunking_table_rows,
        test_column_mapping_standard,
        test_column_mapping_alternate,
        test_clean_table,
        test_grounding_exact_match,
        test_grounding_rejects_hallucination,
        test_confidence_mapping,
        test_enforce_not_found,
        test_validation_rejects_hallucinated,
        test_real_pdf_ingestion,
        test_real_pdf_table_extraction,
        test_real_pdf_chunking,
    ]

    passed = failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as exc:
            failed += 1
            print(f"  FAIL: {test_fn.__name__}: {exc}")

    print(f"\n{'=' * 60}")
    print(f"  Results: {passed} passed, {failed} failed, {len(tests)} total")
    print(f"{'=' * 60}\n")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
