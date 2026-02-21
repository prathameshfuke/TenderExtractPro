"""
main.py — Pipeline orchestration for TenderExtractPro.

This is the entry point that ties everything together. The pipeline
runs in 6 stages with timing and error handling at each stage.

We originally had this as a simple script but refactored to a class
because:
  1. The retriever index needs to persist across multiple queries
  2. Users wanted to call the pipeline from their own code without CLI
  3. We needed to support batch processing (iterate over a directory)

The multi-query retrieval strategy (querying 4 different phrasings per
extraction type) improved recall by ~15% compared to a single query.
Different phrasings hit different chunks via BM25 — "technical
specifications" finds spec sections, "material grade quality" finds
material tables, etc. We deduplicate by chunk_id so there's no
redundancy in the final context.
- Prathamesh, 2026-02-17
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from tender_extraction.config import config
from tender_extraction.ingestion import ingest_document
from tender_extraction.table_extraction import extract_tables, parse_table_to_specs
from tender_extraction.chunking import create_chunks
from tender_extraction.retrieval import HybridRetriever
from tender_extraction.extraction import extract_specifications, extract_scope_of_work
from tender_extraction.validation import verify_grounding, validate_extraction_result

logger = logging.getLogger("tender_extraction")


class TenderExtractionPipeline:
    """
    End-to-end extraction pipeline.

    Usage:
        pipeline = TenderExtractionPipeline()
        result = pipeline.run("dataset/MTF.pdf")
        print(json.dumps(result, indent=2))
    """

    def __init__(self):
        self._retriever: Optional[HybridRetriever] = None

    def run(
        self,
        file_path: str,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the full 6-stage pipeline on a single document.

        Returns a validated ExtractionResult dict. Optionally writes
        the result to a JSON file (human-readable with indentation).
        """
        overall_start = time.time()
        path = Path(file_path)
        logger.info("=" * 60)
        logger.info("TenderExtractPro — Processing: %s (%.1f MB)",
                     path.name, path.stat().st_size / (1024 * 1024))
        logger.info("=" * 60)

        # ── Stage 1: Ingestion ────────────────────────────────────
        t0 = time.time()
        logger.info("[1/6] Ingesting document ...")
        pages = ingest_document(file_path)
        logger.info("  ✓ %d pages in %.1fs", len(pages), time.time() - t0)

        # ── Stage 2: Table extraction (PDF only) ──────────────────
        t0 = time.time()
        logger.info("[2/6] Extracting tables ...")
        tables = []
        table_specs = []
        if path.suffix.lower() == ".pdf":
            tables = extract_tables(file_path)
            for table in tables:
                table_specs.extend(parse_table_to_specs(table))
            logger.info(
                "  ✓ %d tables, %d table-specs in %.1fs",
                len(tables), len(table_specs), time.time() - t0
            )
        else:
            logger.info("  ⊘ Skipped (non-PDF)")

        # ── Stage 3: Chunking ────────────────────────────────────
        t0 = time.time()
        logger.info("[3/6] Creating chunks ...")
        chunks = create_chunks(pages, tables)
        logger.info("  ✓ %d chunks in %.1fs", len(chunks), time.time() - t0)

        if not chunks:
            logger.warning("No chunks — returning empty result.")
            return _empty_result()

        # ── Stage 4: Retrieval ───────────────────────────────────
        t0 = time.time()
        logger.info("[4/6] Building retrieval index + querying ...")
        self._retriever = HybridRetriever(chunks)

        # Multiple queries per extraction type to maximize recall.
        # Each query catches different aspects of the specs/scope.
        spec_queries = [
            "technical specifications requirements standards",
            "material specifications grade quality",
            "dimensions measurements tolerances",
            "equipment machinery specifications capacity",
            "bill of quantities item description rate",
        ]
        scope_queries = [
            "scope of work tasks deliverables",
            "project timeline schedule milestones completion",
            "exclusions not included out of scope",
            "contractor responsibilities obligations",
        ]

        spec_chunks = self._multi_query_retrieve(spec_queries, top_k=15)
        scope_chunks = self._multi_query_retrieve(scope_queries, top_k=10)
        logger.info(
            "  ✓ %d spec chunks, %d scope chunks in %.1fs",
            len(spec_chunks), len(scope_chunks), time.time() - t0
        )

        # ── Stage 5: LLM Extraction ─────────────────────────────
        t0 = time.time()
        logger.info("[5/6] Running LLM extraction ...")

        llm_specs = extract_specifications(spec_chunks)
        llm_scope = extract_scope_of_work(scope_chunks)

        # Merge table-extracted specs with LLM-extracted specs.
        # Table specs come first because they're typically more structured
        # and reliable than LLM-extracted ones.
        all_specs = table_specs + llm_specs
        logger.info(
            "  ✓ %d specs (%d table + %d LLM), %d tasks in %.1fs",
            len(all_specs), len(table_specs), len(llm_specs),
            len(llm_scope.get("tasks", [])), time.time() - t0
        )

        # ── Stage 6: Validation & Grounding ─────────────────────
        t0 = time.time()
        logger.info("[6/6] Validating and grounding ...")

        raw_result = {
            "technical_specifications": all_specs,
            "scope_of_work": llm_scope,
        }

        all_retrieval_results = spec_chunks + scope_chunks
        validated = verify_grounding(raw_result, all_retrieval_results)

        # Pydantic validation pass for schema enforcement
        try:
            result_model = validate_extraction_result(validated)
            final_result = result_model.model_dump()
        except Exception as exc:
            # If Pydantic fails, we still return the raw validated data.
            # This usually happens when the LLM outputs a weird type
            # (e.g. page as string "15" instead of int 15).
            logger.warning("Pydantic validation issue: %s. Using raw result.", exc)
            final_result = validated

        logger.info("  ✓ Validation in %.1fs", time.time() - t0)

        # ── Summary ──────────────────────────────────────────────
        elapsed = time.time() - overall_start
        n_specs = len(final_result.get("technical_specifications", []))
        n_tasks = len(final_result.get("scope_of_work", {}).get("tasks", []))
        n_excl = len(final_result.get("scope_of_work", {}).get("exclusions", []))

        logger.info("=" * 60)
        logger.info("DONE in %.1fs | %d specs | %d tasks | %d exclusions",
                     elapsed, n_specs, n_tasks, n_excl)
        logger.info("=" * 60)

        # Write output if path specified
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(final_result, f, indent=2, ensure_ascii=False)
            logger.info("Output written to: %s", output_path)

        return final_result

    def _multi_query_retrieve(
        self,
        queries: List[str],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Run multiple retrieval queries and deduplicate by chunk_id.

        This is better than a single query because BM25 is very sensitive
        to exact wording. "steel specification" and "material grade" might
        hit completely different chunks even though they're both relevant
        to technical specs.
        """
        seen_ids = set()
        results = []

        for query in queries:
            hits = self._retriever.hybrid_retrieve(query, top_k=top_k)
            for hit in hits:
                cid = hit["chunk"].chunk_id
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    results.append(hit)

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]


def _empty_result() -> Dict[str, Any]:
    """Schema-compliant empty result for when processing finds nothing."""
    return {
        "technical_specifications": [],
        "scope_of_work": {"tasks": [], "exclusions": []},
    }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="tender_extraction",
        description="TenderExtractPro — Extract technical specs and scope from tender documents",
    )
    parser.add_argument("file", help="Path to tender document (PDF, DOCX, JPG, PNG)")
    parser.add_argument("--output", "-o", default=None, help="JSON output path (default: stdout)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    pipeline = TenderExtractionPipeline()

    try:
        result = pipeline.run(args.file, args.output)
        if args.output is None:
            print(json.dumps(result, indent=2, ensure_ascii=False))
    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        sys.exit(1)
    except ValueError as exc:
        logger.error("Invalid input: %s", exc)
        sys.exit(1)
    except RuntimeError as exc:
        logger.error("Runtime error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
