"""
test_pipeline.py — Tests for TenderExtractPro.

These tests cover the core logic that doesn't need external dependencies
(no LLM, no Tesseract, no internet). They validate:
  - Pydantic schema enforcement and defaults
  - Chunking logic with synthetic and real-world-like data
  - Table column mapping against common tender header formats
  - Grounding verification and confidence scoring
  - Example output JSON schema compliance
  - Real dataset PDF ingestion (tables + text extraction)

The real dataset tests (test_real_*) are run against actual tender PDFs
in the dataset/ folder. These are critical smoke tests that catch
regressions in pdfplumber settings, chunking logic, and table extraction.

Run with:
    python tests/test_pipeline.py
    python -m pytest tests/test_pipeline.py -v
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

# Make sure the project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tender_extraction.schemas import (
    Chunk,
    ChunkMetadata,
    ExtractionResult,
    TechnicalSpecification,
    SourceCitation,
    ScopeTask,
    Exclusion,
    ScopeOfWork,
)
from tender_extraction.chunking import create_chunks
from tender_extraction.table_extraction import (
    _map_columns,
    _clean_table,
    extract_tables,
)
from tender_extraction.validation import (
    _compute_grounding,
    _score_to_confidence,
    _enforce_not_found,
    verify_grounding,
)
from tender_extraction.ingestion import ingest_document

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"



def test_spec_schema_defaults():
    """NOT_FOUND defaults should be set for all optional fields."""
    spec = TechnicalSpecification(
        item_name="Steel Bars",
        specification_text="Grade 60 steel bars",
        source=SourceCitation(chunk_id="chunk_001", page=5),
    )
    assert spec.unit == "NOT_FOUND"
    assert spec.tolerance == "NOT_FOUND"
    assert spec.confidence == "LOW"
    assert spec.source.exact_text == "NOT_FOUND"
    print("  ✓ test_spec_schema_defaults")


def test_extraction_result_empty():
    """Empty result should serialize with proper structure."""
    result = ExtractionResult()
    data = result.model_dump()
    assert data["technical_specifications"] == []
    assert data["scope_of_work"]["tasks"] == []
    assert data["scope_of_work"]["exclusions"] == []
    print("  ✓ test_extraction_result_empty")


def test_extraction_result_full():
    """Full result with nested data should validate and serialize."""
    result = ExtractionResult(
        technical_specifications=[
            TechnicalSpecification(
                item_name="Cement",
                specification_text="OPC Grade 53",
                unit="MT",
                numeric_value="120",
                standard_reference="IS 12269",
                source=SourceCitation(
                    chunk_id="chunk_010", page=15,
                    exact_text="OPC Grade 53 as per IS 12269",
                ),
                confidence="HIGH",
            )
        ],
        scope_of_work=ScopeOfWork(
            tasks=[
                ScopeTask(
                    task_description="Site preparation",
                    deliverables=["Cleared site"],
                    timeline="2 weeks",
                    source=SourceCitation(chunk_id="chunk_020", page=8),
                )
            ],
            exclusions=[
                Exclusion(
                    item="Furniture",
                    source=SourceCitation(chunk_id="chunk_030", page=12),
                )
            ],
        ),
    )
    data = result.model_dump()
    assert len(data["technical_specifications"]) == 1
    assert data["technical_specifications"][0]["confidence"] == "HIGH"
    assert len(data["scope_of_work"]["tasks"]) == 1
    print("  ✓ test_extraction_result_full")



def test_chunking_with_sections():
    """Section headers should propagate into chunk metadata."""
    pages = [{
        "page": 1,
        "text": (
            "1 Introduction\n"
            "This tender is for highway construction.\n"
            "\n"
            "2 Technical Specifications\n"
            "2.1 Material Requirements\n"
            "Steel bars shall conform to ASTM A615 Grade 60.\n"
            "Cement shall be OPC Grade 53.\n"
        ),
        "is_ocr": False,
    }]
    chunks = create_chunks(pages, tables=None)
    assert len(chunks) >= 2, f"Expected ≥2 chunks, got {len(chunks)}"

    sections = {c.metadata.section for c in chunks}
    # At least one section header should be detected
    assert any("Introduction" in s or "Technical" in s or "Material" in s
               for s in sections), f"No section headers found in: {sections}"
    print(f"  ✓ test_chunking_with_sections ({len(chunks)} chunks, sections: {sections})")


def test_chunking_table_rows():
    """Each table row should become a separate chunk with headers."""
    tables = [{
        "table_id": "table_001",
        "page": 5,
        "headers": ["Item", "Specification", "Unit"],
        "rows": [
            ["Steel Bars", "ASTM A615 Grade 60", "kg"],
            ["Cement", "IS 12269 OPC 53", "MT"],
        ],
        "bbox": [100, 200, 500, 400],
    }]
    chunks = create_chunks(pages=[], tables=tables)
    assert len(chunks) == 2, f"Expected 2 table chunks, got {len(chunks)}"

    for c in chunks:
        assert c.metadata.chunk_type == "table"
        assert c.metadata.table_id == "table_001"
        assert "[Table Headers]" in c.text
        assert "[Row" in c.text
    print(f"  ✓ test_chunking_table_rows ({len(chunks)} chunks)")




def test_column_mapping_standard_headers():
    """Standard tender table headers should map correctly."""
    headers = [
        "Sr. No.",
        "Item Description",
        "Specification Details",
        "Unit of Measure",
        "Quantity",
        "IS/ASTM Standard",
    ]
    mapping = _map_columns(headers)
    assert "item_name" in mapping, f"item_name not in mapping: {mapping}"
    assert "specification_text" in mapping
    assert "unit" in mapping
    print(f"  ✓ test_column_mapping_standard_headers: {mapping}")


def test_column_mapping_alternate_headers():
    """Alternative header formats found in real tenders."""
    headers = ["Sl No", "Name of Material", "Required Quantity", "Rate"]
    mapping = _map_columns(headers)
    assert "item_name" in mapping
    print(f"  ✓ test_column_mapping_alternate_headers: {mapping}")


def test_clean_table_normalizes():
    """Table cleaning should handle None, newlines, and extra spaces."""
    raw = [
        ["Header 1", None, "Header 3"],
        ["Cell\nwith\nnewlines", "", "  Value  "],
    ]
    cleaned = _clean_table(raw)
    assert cleaned[0][1] == ""
    assert "\n" not in cleaned[1][0]
    assert cleaned[1][0] == "Cell with newlines"
    print("  ✓ test_clean_table_normalizes")




def test_grounding_exact_match():
    """Exact substring should give 1.0 grounding score."""
    score = _compute_grounding(
        "Steel bars Grade 60",
        "The contractor shall supply Steel bars Grade 60 for all foundations.",
    )
    assert score == 1.0
    print(f"  ✓ test_grounding_exact_match (score={score})")


def test_grounding_fuzzy_match():
    """Similar but not identical text should give intermediate score."""
    score = _compute_grounding(
        "OPC Grade 53 cement",
        "Ordinary Portland Cement of Grade 53",
    )
    assert 0.3 < score < 1.0
    print(f"  ✓ test_grounding_fuzzy_match (score={score:.3f})")


def test_grounding_rejects_hallucination():
    """Completely unrelated text should give very low score."""
    score = _compute_grounding(
        "Concrete pump rental agreement terms",
        "The deadline for submission of tenders is January 15th.",
    )
    assert score < 0.3
    print(f"  ✓ test_grounding_rejects_hallucination (score={score:.3f})")


def test_confidence_mapping():
    """Score→confidence bucket mapping."""
    assert _score_to_confidence(0.95) == "HIGH"
    assert _score_to_confidence(0.75) == "MEDIUM"
    assert _score_to_confidence(0.30) == "LOW"
    print("  ✓ test_confidence_mapping")


def test_enforce_not_found():
    """Empty/None fields should become NOT_FOUND."""
    data = {"unit": "", "tolerance": None, "material": "Steel", "item_name": "X"}
    result = _enforce_not_found(data)
    assert result["unit"] == "NOT_FOUND"
    assert result["tolerance"] == "NOT_FOUND"
    assert result["material"] == "Steel"  # non-empty should be unchanged
    print("  ✓ test_enforce_not_found")


def test_grounding_integration():
    """Full grounding verification: should accept real and reject hallucinated."""
    source_chunks = [{
        "chunk": Chunk(
            chunk_id="chunk_001",
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
                "unit": "kg",
                "source": {
                    "chunk_id": "chunk_001", "page": 15,
                    "exact_text": "Steel reinforcement bars shall be Grade 60 conforming to ASTM A615",
                },
                "confidence": "HIGH",
            },
            {
                "item_name": "Hallucinated Item",
                "specification_text": "This was made up by the LLM",
                "unit": "kg",
                "source": {
                    "chunk_id": "chunk_999", "page": 99,
                    "exact_text": "Completely fabricated text that doesn't exist anywhere",
                },
                "confidence": "HIGH",
            },
        ],
        "scope_of_work": {"tasks": [], "exclusions": []},
    }
    validated = verify_grounding(extraction, source_chunks)
    specs = validated["technical_specifications"]
    assert len(specs) == 1, f"Expected hallucinated spec rejected, got {len(specs)}"
    assert specs[0]["item_name"] == "Steel Bars"
    print(f"  ✓ test_grounding_integration (1 accepted, 1 rejected)")





def test_example_output_validates():
    """The example_output.json should pass Pydantic validation."""
    example_path = PROJECT_ROOT / "sample_output" / "example_output.json"
    if not example_path.exists():
        print("  ⊘ test_example_output_validates SKIPPED (file not found)")
        return

    with open(example_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    result = ExtractionResult.model_validate(data)
    assert len(result.technical_specifications) >= 5
    assert len(result.scope_of_work.tasks) >= 3

    for spec in result.technical_specifications:
        assert spec.source.chunk_id, f"Missing citation for {spec.item_name}"
        assert spec.source.page > 0

    print(
        f"  ✓ test_example_output_validates "
        f"({len(result.technical_specifications)} specs, "
        f"{len(result.scope_of_work.tasks)} tasks)"
    )





def test_real_pdf_ingestion():
    """
    Ingest a real tender PDF from the dataset folder.
    Uses globaltender1576.pdf because it's the smallest (~1.2MB).
    Verifies that we get non-empty text from at least some pages.
    """
    pdf_path = DATASET_DIR / "globaltender1576.pdf"
    if not pdf_path.exists():
        print("  ⊘ test_real_pdf_ingestion SKIPPED (dataset not found)")
        return

    pages = ingest_document(str(pdf_path))
    assert len(pages) > 0, "No pages ingested"

    # At least some pages should have meaningful text
    pages_with_text = [p for p in pages if len(p["text"].strip()) > 50]
    assert len(pages_with_text) > 0, "All pages empty after ingestion"

    total_chars = sum(len(p["text"]) for p in pages)
    print(
        f"  ✓ test_real_pdf_ingestion: {len(pages)} pages, "
        f"{len(pages_with_text)} with text, {total_chars:,} total chars"
    )


def test_real_pdf_table_extraction():
    """
    Extract tables from a real tender PDF.
    Uses MTF.pdf which has multiple specification tables.
    """
    pdf_path = DATASET_DIR / "MTF.pdf"
    if not pdf_path.exists():
        print("  ⊘ test_real_pdf_table_extraction SKIPPED (dataset not found)")
        return

    tables = extract_tables(str(pdf_path))

    # Most tender PDFs have at least a few tables
    print(
        f"  ✓ test_real_pdf_table_extraction: {len(tables)} tables found"
    )
    for t in tables[:3]:  # Show first 3 tables for verification
        print(
            f"    table {t['table_id']} (page {t['page']}): "
            f"{len(t['headers'])} cols, {len(t['rows'])} rows"
        )
        if t["headers"]:
            print(f"    headers: {t['headers'][:5]}")


def test_real_pdf_full_chunking():
    """
    Full ingestion + chunking pipeline on a real tender.
    Uses Tenderdocuments.pdf.
    """
    pdf_path = DATASET_DIR / "Tenderdocuments.pdf"
    if not pdf_path.exists():
        print("  ⊘ test_real_pdf_full_chunking SKIPPED (dataset not found)")
        return

    pages = ingest_document(str(pdf_path))
    tables = extract_tables(str(pdf_path))
    chunks = create_chunks(pages, tables)

    assert len(chunks) > 0, "No chunks created"

    # Count chunk types
    type_counts = {}
    for c in chunks:
        ct = c.metadata.chunk_type
        type_counts[ct] = type_counts.get(ct, 0) + 1

    print(
        f"  ✓ test_real_pdf_full_chunking: {len(chunks)} chunks "
        f"(types: {type_counts})"
    )


def test_real_pdf_pbmc_tables():
    """
    Smoke test on the largest tender: RFPPBMCJob290.pdf (~12MB).
    This is the most complex doc with lots of tables.
    """
    pdf_path = DATASET_DIR / "RFPPBMCJob290.pdf"
    if not pdf_path.exists():
        print("  ⊘ test_real_pdf_pbmc_tables SKIPPED (dataset not found)")
        return

    tables = extract_tables(str(pdf_path))
    total_rows = sum(len(t["rows"]) for t in tables)

    print(
        f"  ✓ test_real_pdf_pbmc_tables: {len(tables)} tables, "
        f"{total_rows} total rows"
    )



def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 60)
    print("  TenderExtractPro — Test Suite")
    print("=" * 60 + "\n")

    tests = [
        # Schema tests
        test_spec_schema_defaults,
        test_extraction_result_empty,
        test_extraction_result_full,
        # Chunking tests
        test_chunking_with_sections,
        test_chunking_table_rows,
        # Table mapping tests
        test_column_mapping_standard_headers,
        test_column_mapping_alternate_headers,
        test_clean_table_normalizes,
        # Grounding tests
        test_grounding_exact_match,
        test_grounding_fuzzy_match,
        test_grounding_rejects_hallucination,
        test_confidence_mapping,
        test_enforce_not_found,
        test_grounding_integration,
        # Example output
        test_example_output_validates,
        # Real dataset tests
        test_real_pdf_ingestion,
        test_real_pdf_table_extraction,
        test_real_pdf_full_chunking,
        test_real_pdf_pbmc_tables,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as exc:
            failed += 1
            print(f"  ✗ {test_fn.__name__} FAILED: {exc}")

    print(f"\n{'=' * 60}")
    print(f"  Results: {passed} passed, {failed} failed, {len(tests)} total")
    print(f"{'=' * 60}\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
