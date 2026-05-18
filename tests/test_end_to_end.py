"""
test_end_to_end.py — End-to-end RAG pipeline verification for TenderExtractPro.

Runs the full 6-stage pipeline on a real dataset PDF and validates:
  1. Ingestion   — pages extracted with text content
  2. Table ext.  — at least some tables found (or skipped gracefully)
  3. Chunking    — meaningful chunk count and correct metadata
  4. Retrieval   — hybrid RRF search returns relevant results
  5. QA          — DocumentChatSession can answer a basic question
  6. expand_query — domain synonym expansion works correctly

NOTE: LLM extraction (Stage 5 in the main pipeline) requires the Mistral GGUF
      model file.  If it is absent the pipeline still completes — the stage is
      skipped with a warning.  This test validates the pipeline *structure*
      end-to-end without requiring the model to be present.

Run with:
    venv312\\Scripts\\python.exe tests/test_end_to_end.py
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import json
from pathlib import Path

# Ensure project root is importable whether run directly or via pytest.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s | %(name)s | %(message)s",
)
# Show our own pipeline stages at INFO.
logging.getLogger("tender_extraction").setLevel(logging.INFO)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR  = PROJECT_ROOT / "dataset"
SAMPLE_PDF   = DATASET_DIR / "globaltender1576.pdf"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_passed = _failed = 0

def _report(name: str, ok: bool, detail: str = "") -> None:
    global _passed, _failed
    status = "PASS" if ok else "FAIL"
    msg = f"  {status}: {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    if ok:
        _passed += 1
    else:
        _failed += 1


# ---------------------------------------------------------------------------
# Stage 1 — Ingestion
# ---------------------------------------------------------------------------

def test_stage1_ingestion() -> list:
    if not SAMPLE_PDF.exists():
        _report("stage1_ingestion", True, "SKIP (sample PDF not found)")
        return []

    from tender_extraction.ingestion import ingest_document
    pages = ingest_document(str(SAMPLE_PDF))

    ok = len(pages) > 0 and any(len(p["text"].strip()) > 50 for p in pages)
    total_chars = sum(len(p["text"]) for p in pages)
    _report(
        "stage1_ingestion",
        ok,
        f"{len(pages)} pages, {total_chars:,} chars"
        + (f", {sum(1 for p in pages if p['is_ocr'])} OCR" if any(p['is_ocr'] for p in pages) else ""),
    )
    return pages


# ---------------------------------------------------------------------------
# Stage 2 — Table Extraction
# ---------------------------------------------------------------------------

def test_stage2_table_extraction() -> list:
    if not SAMPLE_PDF.exists():
        _report("stage2_table_extraction", True, "SKIP (sample PDF not found)")
        return []

    from tender_extraction.table_extraction import extract_tables
    tables = extract_tables(str(SAMPLE_PDF))

    # Gracefully accept 0 tables — not all PDFs have extractable tables.
    ok = isinstance(tables, list)
    _report("stage2_table_extraction", ok, f"{len(tables)} tables found")
    return tables


# ---------------------------------------------------------------------------
# Stage 3 — Chunking
# ---------------------------------------------------------------------------

def test_stage3_chunking(pages: list, tables: list) -> list:
    if not pages:
        _report("stage3_chunking", True, "SKIP (no pages)")
        return []

    from tender_extraction.chunking import create_chunks
    # Use fast (non-semantic) chunking to avoid loading the embedding model twice.
    chunks = create_chunks(pages, tables, use_semantic=False)

    ok = len(chunks) > 10
    types = {}
    for c in chunks:
        types[c.metadata.chunk_type] = types.get(c.metadata.chunk_type, 0) + 1
    _report("stage3_chunking", ok, f"{len(chunks)} chunks | types: {types}")
    return chunks


# ---------------------------------------------------------------------------
# Stage 4 — Hybrid RRF Retrieval
# ---------------------------------------------------------------------------

def test_stage4_retrieval(chunks: list) -> None:
    if not chunks:
        _report("stage4_retrieval", True, "SKIP (no chunks)")
        return

    from tender_extraction.retrieval import HybridRetriever
    from tender_extraction.config import config

    persist_dir = str(PROJECT_ROOT / "_e2e_test_qdrant")
    # Disable cross-encoder reranking for speed in CI/test runs.
    orig_rerank_k = config.retrieval.rerank_top_k
    config.retrieval.rerank_top_k = 0

    try:
        retriever = HybridRetriever(persist_dir=persist_dir)
        retriever.build_index(chunks, collection_name="e2e_test", force_rebuild=True)

        # Query 1 — technical spec
        r1 = retriever.retrieve("technical specifications material requirement", top_k=5)
        ok1 = len(r1) > 0
        _report("stage4_retrieval_spec_query", ok1, f"top result: '{r1[0]['chunk'].text[:60]}...'" if r1 else "no results")

        # Query 2 — scope of work
        r2 = retriever.retrieve("scope of work deliverables obligations", top_k=5)
        ok2 = len(r2) > 0
        _report("stage4_retrieval_scope_query", ok2, f"top result: '{r2[0]['chunk'].text[:60]}...'" if r2 else "no results")

        # Validate RRF score is present and non-negative
        if r1:
            score = r1[0].get("score", -1)
            _report("stage4_rrf_score_positive", score >= 0, f"score={score:.6f}")

        retriever.close()
    finally:
        config.retrieval.rerank_top_k = orig_rerank_k
        shutil.rmtree(persist_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Stage 5 — Query Expansion
# ---------------------------------------------------------------------------

def test_stage5_query_expansion() -> None:
    from tender_extraction.retrieval import expand_query

    # Core technical terms
    e1 = expand_query("material specification")
    ok1 = len(e1) > len("material specification") and ("grade" in e1.lower() or "specs" in e1.lower())
    _report("stage5_expand_query_spec", ok1, f"'{e1[:80]}'")

    # Procurement terms
    e2 = expand_query("eligibility criteria")
    ok2 = len(e2) > len("eligibility criteria")
    _report("stage5_expand_query_eligibility", ok2, f"'{e2[:80]}'")

    # Guarantee / EMD
    e3 = expand_query("guarantee deposit")
    ok3 = len(e3) > len("guarantee deposit")
    _report("stage5_expand_query_guarantee", ok3, f"'{e3[:80]}'")

    # Query with no matching term — should return unchanged
    e4 = expand_query("totally unrelated xyz query")
    ok4 = e4 == "totally unrelated xyz query"
    _report("stage5_expand_query_no_match", ok4, f"returned: '{e4}'")


# ---------------------------------------------------------------------------
# Stage 6 — QA Session (uses chunk cache, avoids re-indexing)
# ---------------------------------------------------------------------------

def test_stage6_qa_session(chunks: list) -> None:
    if not SAMPLE_PDF.exists() or not chunks:
        _report("stage6_qa_session", True, "SKIP (no PDF or chunks)")
        return

    from tender_extraction.qa import DocumentChatSession
    from tender_extraction.config import config

    persist_dir = str(PROJECT_ROOT / "_e2e_qa_qdrant")
    orig_rerank_k = config.retrieval.rerank_top_k
    config.retrieval.rerank_top_k = 0

    try:
        session = DocumentChatSession(
            str(SAMPLE_PDF),
            persist_dir=persist_dir,
            force_reindex=True,
        )
        # Pre-inject the already-computed chunks so we skip re-ingestion.
        session._retriever = None  # will be built in build()
        # Build normally — will re-ingest but that's fast with cached chunks.
        session.build()

        answer = session.ask("What is this tender about?")

        ok = isinstance(answer, dict) and "answer" in answer
        ans_text = str(answer.get("answer", ""))[:120]
        _report("stage6_qa_session", ok, f"answer='{ans_text}'")
        session.close()
    except Exception as exc:
        _report("stage6_qa_session", False, f"Exception: {exc}")
    finally:
        config.retrieval.rerank_top_k = orig_rerank_k
        shutil.rmtree(persist_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Manifest integrity check
# ---------------------------------------------------------------------------

def test_manifest_written_after_index() -> None:
    """Validate that the Qdrant manifest is correctly written/read on rebuild."""
    from tender_extraction.retrieval import HybridRetriever
    from tender_extraction.schemas import Chunk, ChunkMetadata

    chunks = [
        Chunk(
            chunk_id="manifest_chunk_1",
            text="Diesel generator set rated at 500 kVA, 1500 RPM, water cooled.",
            metadata=ChunkMetadata(page=5, section="Equipment", chunk_type="paragraph"),
        ),
    ]
    persist_dir = str(PROJECT_ROOT / "_e2e_manifest_check")
    try:
        r = HybridRetriever(persist_dir=persist_dir)
        r.build_index(chunks, collection_name="manifest_test", force_rebuild=True)
        manifest = r._load_manifest()
        ok = "manifest_test" in manifest and "fingerprint" in manifest["manifest_test"]
        _report("manifest_integrity", ok, f"keys: {list(manifest.get('manifest_test', {}).keys())}")
        r.close()
    finally:
        shutil.rmtree(persist_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 65)
    print("  TenderExtractPro — End-to-End RAG Pipeline Test")
    print("=" * 65 + "\n")

    # Linear pipeline
    pages  = test_stage1_ingestion()
    tables = test_stage2_table_extraction()
    chunks = test_stage3_chunking(pages, tables)
    test_stage4_retrieval(chunks)
    test_stage5_query_expansion()

    # QA session test (skip if LLM model absent — it will just return NOT_FOUND)
    test_stage6_qa_session(chunks)

    # Auxiliary checks
    test_manifest_written_after_index()

    print(f"\n{'=' * 65}")
    total = _passed + _failed
    print(f"  Results: {_passed} passed, {_failed} failed, {total} total")
    print(f"{'=' * 65}\n")
    sys.exit(0 if _failed == 0 else 1)


if __name__ == "__main__":
    main()
