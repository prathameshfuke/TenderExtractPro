"""
main.py — End-to-end pipeline orchestration.

Runs the system in a 6-stage sequence:
  1. Ingestion
  2. Table extraction
  3. Chunking
  4. Retrieval
  5. LLM Extraction
  6. Validation

Produces structured XML/JSON output.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from tender_extraction.config import config
from tender_extraction.ingestion import ingest_document
from tender_extraction.table_extraction import extract_tables, parse_table_to_specs
from tender_extraction.chunking import create_chunks
from tender_extraction.retrieval import HybridRetriever
from tender_extraction.extraction import extract_specifications, extract_scope_of_work
from tender_extraction.validation import validate_extractions, validate_schema

logger = logging.getLogger("tender_extraction")


def discover_document_topic(pages: list[dict]) -> str:
    """
    Read pages 1-5 to find the tender subject.
    Returns a topic string for targeted retrieval.
    """
    early_text = " ".join(p["text"] for p in pages[:5])
    # Extract the subject line — look for patterns like "for Procurement of X"
    patterns = [
        r"(?:procurement|supply|purchase|tender for)[^\n]{0,200}",
        r"(?:name of work|subject)[:\s]+([^\n]{10,150})",
        r"CHAPTER[\s\-]+\d+[^\n]*\n([^\n]{20,200})"
    ]
    for pat in patterns:
        m = re.search(pat, early_text, re.IGNORECASE)
        if m:
            return m.group(0)[:200]
    return early_text[:300]


def build_targeted_queries(topic: str) -> list[str]:
    """Build retrieval queries specific to what this tender is actually about."""
    return [
        topic,                                    # exact topic
        f"{topic} technical specifications",
        f"{topic} requirements performance",
        f"{topic} minimum maximum parameters",
        "schedule of requirements specifications technical details",
        "chapter 4 specifications",               # common tender section name
        "scope of supply deliverables",
        "warranty maintenance period",
        "compliance statement specifications",
    ]


def _progress(step: int, total: int, label: str, detail: str = ""):
    bar_len = 30
    filled = int(bar_len * step / total)
    bar = "█" * filled + "░" * (bar_len - filled)
    pct = int(100 * step / total)
    print(f"\r  [{bar}] {pct:3d}%  Step {step}/{total}: {label}"
          + (f" — {detail}" if detail else ""), 
          end="" if step < total else "\n", flush=True)


class TenderExtractionPipeline:
    """
    Full 6-stage extraction pipeline.

    Usage:
        pipeline = TenderExtractionPipeline()
        result = pipeline.run("dataset/globaltender1576.pdf", output_path="out.json")
    """

    def __init__(self):
        self._retriever: Optional[HybridRetriever] = None

    def run(
        self,
        file_path: str,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run all 6 stages on a real document. Produces real JSON output.
        Raises RuntimeError if the LLM model is not available.
        """
        overall_start = time.time()
        path = Path(file_path)
        logger.info("=" * 60)
        logger.info("TenderExtractPro -- Processing: %s (%.1f MB)",
                     path.name, path.stat().st_size / (1024 * 1024))
        logger.info("=" * 60)

        # -- Stage 1: Ingestion ------------------------------------------------
        t0 = time.time()
        _progress(1, 6, "Ingesting document...")
        logger.info("[1/6] Ingesting document ...")
        pages = ingest_document(file_path)
        logger.info(
            "  Ingested: %d pages (%d text, %d OCR) in %.1fs",
            len(pages),
            sum(1 for p in pages if not p["is_ocr"]),
            sum(1 for p in pages if p["is_ocr"]),
            time.time() - t0,
        )

        # -- Stage 2: Table Extraction -----------------------------------------
        t0 = time.time()
        _progress(2, 6, "Extracting tables...")
        logger.info("[2/6] Extracting tables ...")
        tables = []
        table_specs = []
        if path.suffix.lower() == ".pdf":
            tables = extract_tables(file_path)
            for table in tables:
                table_specs.extend(parse_table_to_specs(table))
            logger.info("  Found: %d tables, %d table-specs in %.1fs",
                        len(tables), len(table_specs), time.time() - t0)
        else:
            logger.info("  Skipped (non-PDF)")

        # -- Stage 3: Chunking -------------------------------------------------
        t0 = time.time()
        _progress(3, 6, "Creating chunks...")
        logger.info("[3/6] Creating chunks ...")
        chunks = create_chunks(pages, tables)
        logger.info("  Created: %d chunks in %.1fs", len(chunks), time.time() - t0)

        if not chunks:
            logger.warning("No chunks created. Returning empty result.")
            return _empty_result(output_path)

        # -- Stage 4: Retrieval ------------------------------------------------
        t0 = time.time()
        _progress(4, 6, "Building retrieval index + querying...")
        logger.info("[4/6] Building retrieval index + querying ...")
        self._retriever = HybridRetriever()
        self._retriever.build_index(chunks)

        topic = discover_document_topic(pages)
        spec_queries = build_targeted_queries(topic)
        scope_queries = [
            "scope of work tasks deliverables",
            "project timeline schedule milestones completion",
            "exclusions not included out of scope",
            "contractor responsibilities obligations",
        ]

        spec_chunks = self._multi_query_retrieve(spec_queries, top_k=15, is_spec=True)
        scope_chunks = self._multi_query_retrieve(scope_queries, top_k=10)
        logger.info(
            "  Retrieved: %d spec chunks, %d scope chunks in %.1fs",
            len(spec_chunks), len(scope_chunks), time.time() - t0,
        )

        # -- Stage 5: LLM Extraction ------------------------------------------
        t0 = time.time()
        _progress(5, 6, "Running LLM extraction (1-3m)...")
        logger.info("[5/6] Running LLM extraction ...")

        llm_specs = extract_specifications(spec_chunks, topic=topic)
        llm_scope = extract_scope_of_work(scope_chunks, topic=topic)

        # Table-extracted specs come first because they're more reliable
        # (structured column mapping vs. LLM generation)
        all_specs = table_specs + llm_specs
        logger.info(
            "  Extracted: %d specs (%d table + %d LLM), %d tasks in %.1fs",
            len(all_specs), len(table_specs), len(llm_specs),
            len(llm_scope.get("tasks", [])), time.time() - t0,
        )

        # -- Stage 6: Validation -----------------------------------------------
        t0 = time.time()
        _progress(6, 6, "Validating and grounding extractions...")
        logger.info("[6/6] Validating and grounding ...")

        raw_result = {
            "technical_specifications": all_specs,
            "scope_of_work": llm_scope,
        }

        all_source_chunks = spec_chunks + scope_chunks
        validated = validate_extractions(raw_result, all_source_chunks)

        # Pydantic schema enforcement
        try:
            result_model = validate_schema(validated)
            final_result = result_model.dict()
        except Exception as exc:
            logger.warning("Pydantic validation issue: %s. Using raw result.", exc)
            final_result = validated

        logger.info("  Validation complete in %.1fs", time.time() - t0)

        # -- Summary -----------------------------------------------------------
        elapsed = time.time() - overall_start
        n_specs = len(final_result.get("technical_specifications", []))
        n_tasks = len(final_result.get("scope_of_work", {}).get("tasks", []))
        n_excl = len(final_result.get("scope_of_work", {}).get("exclusions", []))
        acc_score = final_result.get("accuracy_score", 0.0)

        logger.info("=" * 60)
        logger.info("DONE in %.1fs", elapsed)
        logger.info("  Specs:      %d", n_specs)
        logger.info("  Tasks:      %d", n_tasks)
        logger.info("  Exclusions: %d", n_excl)
        logger.info("  Accuracy:   %.2f%%", acc_score)
        logger.info("=" * 60)

        # Write output
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
        is_spec: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Run multiple retrieval queries and deduplicate by chunk_id.
        Different query phrasings hit different BM25 keyword matches.
        """
        seen_ids = set()
        results = []

        for query in queries:
            if is_spec:
                hits = self._retriever.retrieve_spec_chunks(query, top_k=top_k)
            else:
                hits = self._retriever.retrieve(query, top_k=top_k)
            for hit in hits:
                cid = hit["chunk"].chunk_id
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    results.append(hit)

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]


def _empty_result(output_path: Optional[str] = None) -> Dict[str, Any]:
    """Schema-compliant empty result."""
    result = {
        "technical_specifications": [],
        "scope_of_work": {"tasks": [], "exclusions": []},
        "accuracy_score": 0.0,
    }
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    return result


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="tender_extraction",
        description="TenderExtractPro -- Extract technical specs and scope from tender documents",
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
