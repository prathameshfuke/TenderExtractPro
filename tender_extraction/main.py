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
import contextlib
import json
import logging
import os
import sys
import time
import re
import warnings
import concurrent.futures
from pathlib import Path
from typing import Any, Dict, List, Optional

from tender_extraction.config import config
from tender_extraction.ingestion import ingest_document
from tender_extraction.table_extraction import extract_tables, parse_table_to_specs
from tender_extraction.chunking import create_chunks
from tender_extraction.retrieval import HybridRetriever, expand_query
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
        topic,
        # Retrieval template focused on scope/spec sections.
        (
            "Find sections related to: Scope of Work, Technical Specifications, "
            "Equipment Specifications, System Requirements, Annexures containing technical data"
        ),
        f"{topic} scope of work work description project scope deliverables",
        f"{topic} technical specifications technical requirements system requirements",
        f"{topic} equipment specifications annexure specifications parameter value unit",
        # Keyword-heavy hybrid booster for BM25
        "scope work technical specifications requirements equipment system annexure",
        "schedule of requirements technical details features parameter value unit tolerance",
        "minimum maximum operating parameters compliance standard IS ASTM",
    ]


# _progress() removed — main() drives tqdm directly via progress_callback


class TenderExtractionPipeline:
    """
    Full 6-stage extraction pipeline.

    Usage:
        pipeline = TenderExtractionPipeline()
        result = pipeline.run("dataset/globaltender1576.pdf", output_path="out.json")
    """

    def __init__(self, persist_dir: Optional[str] = None, force_reindex: bool = False):
        self._retriever: Optional[HybridRetriever] = None
        self._persist_dir = persist_dir
        self._force_reindex = force_reindex

    def run(
        self,
        file_path: str,
        output_path: Optional[str] = None,
        progress_callback=None,
    ) -> Dict[str, Any]:
        """
        Run all 6 stages on a real document. Produces real JSON output.
        Raises RuntimeError if the LLM model is not available.
        
        Args:
            progress_callback: Optional callable(progress: int, message: str).
                               Called at each stage with 0-100 progress and a message.
        """
        def _update(pct: int, msg: str):
            if progress_callback:
                try:
                    progress_callback(pct, msg)
                except Exception:
                    pass
        overall_start = time.time()
        path = Path(file_path)
        logger.info("=" * 60)
        logger.info("TenderExtractPro -- Processing: %s (%.1f MB)",
                 path.name, path.stat().st_size / (1024 * 1024))
        logger.info("=" * 60)

        try:

            # -- Stage 1: Ingestion --------------------------------------------
            t0 = time.time()
            _update(5, "Ingesting document pages...")
            logger.info("[1/6] Ingesting document ...")
            pages = ingest_document(file_path)
            logger.info(
                "  Ingested: %d pages (%d text, %d OCR) in %.1fs",
                len(pages),
                sum(1 for p in pages if not p["is_ocr"]),
                sum(1 for p in pages if p["is_ocr"]),
                time.time() - t0,
            )

            # -- Stage 2: Table Extraction -------------------------------------
            t0 = time.time()
            _update(20, "Extracting tables...")
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

            # -- Stage 3: Chunking ---------------------------------------------
            t0 = time.time()
            _update(35, "Chunking document text (Semantic)...")
            logger.info("[3/6] Creating chunks ...")
            # Enabling semantic chunking by default as it yields much better boundaries
            chunks = create_chunks(pages, tables, use_semantic=True)
            logger.info("  Created: %d chunks in %.1fs", len(chunks), time.time() - t0)

            if not chunks:
                logger.warning("No chunks created. Returning empty result.")
                return _empty_result(output_path)

            # -- Stage 4: Retrieval --------------------------------------------
            t0 = time.time()
            _update(45, "Building retrieval index (Parent-Child)...")
            logger.info("[4/6] Building retrieval index + querying ...")
            self._retriever = HybridRetriever(persist_dir=self._persist_dir)
            collection_name = path.stem
            self._retriever.build_index(chunks, collection_name=collection_name, force_rebuild=self._force_reindex)

            topic = discover_document_topic(pages)
            spec_queries = [expand_query(q) for q in build_targeted_queries(topic)]
            scope_queries = [
                expand_query("detailed scope of work tasks deliverables"),
                expand_query("project completion schedule milestones timeline"),
                expand_query("items excluded from scope boundary limits"),
                expand_query("technical requirements and contractor obligations"),
            ]

            # Since we now use Parent-Child, top_k 15-20 gives us rich context
            spec_chunks = self._multi_query_retrieve(spec_queries, top_k=15, mode="spec")
            scope_chunks = self._multi_query_retrieve(scope_queries, top_k=10, mode="scope")

            logger.info(
                "  Retrieved: %d spec chunks, %d scope chunks in %.1fs",
                len(spec_chunks), len(scope_chunks), time.time() - t0,
            )

            # -- Stage 5: LLM Extraction --------------------------------------
            t0 = time.time()
            _update(60, "Running LLM extraction (this takes 1-3 min)...")
            logger.info("[5/6] Running LLM extraction concurrently ...")

            llm_specs = []
            llm_scope = {"summary": "NOT_FOUND", "deliverables": [], "exclusions": [], "locations": [], "references": []}
        
            # NOTE: LLM inference is serialized by _llm_lock in extraction.py.
            # Running in threads just adds overhead without true parallelism,
            # so we run them sequentially for clarity and reliability.
            try:
                _update(62, "LLM extracting technical specifications...")
                llm_specs = extract_specifications(spec_chunks, topic)
            except Exception as e:
                if config.llm.require_gpu:
                    raise
                logger.warning("Failed to extract specifications: %s", e)

            try:
                _update(78, "LLM extracting scope of work...")
                llm_scope = extract_scope_of_work(scope_chunks, topic)
            except Exception as e:
                if config.llm.require_gpu:
                    raise
                logger.warning("Failed to extract scope of work: %s", e)

            # Table-extracted specs come first because they're more reliable
            # (structured column mapping vs. LLM generation)
            all_specs = table_specs + llm_specs
            logger.info(
                "  Extracted: %d specs (%d table + %d LLM), %d deliverables in %.1fs",
                len(all_specs), len(table_specs), len(llm_specs),
                len(llm_scope.get("deliverables", [])), time.time() - t0,
            )

            # -- Stage 6: Validation -------------------------------------------
            t0 = time.time()
            _update(90, "Validating and grounding extractions...")
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
                if hasattr(result_model, 'model_dump'):
                    final_result = result_model.model_dump()
                else:
                    final_result = result_model.dict()
            except Exception as exc:
                logger.warning("Pydantic validation issue: %s. Using raw result.", exc)
                final_result = validated

            logger.info("  Validation complete in %.1fs", time.time() - t0)

            # -- Summary -------------------------------------------------------
            elapsed = time.time() - overall_start
            n_specs = len(final_result.get("technical_specifications", []))
            n_deliverables = len(final_result.get("scope_of_work", {}).get("deliverables", []))
            n_excl = len(final_result.get("scope_of_work", {}).get("exclusions", []))
            acc_score = final_result.get("accuracy_score", 0.0)

            logger.info("=" * 60)
            logger.info("DONE in %.1fs", elapsed)
            logger.info("  Specs:       %d", n_specs)
            logger.info("  Deliverables:%d", n_deliverables)
            logger.info("  Exclusions:  %d", n_excl)
            logger.info("  Accuracy:    %.2f%%", acc_score)
            logger.info("=" * 60)

            # Write output
            if output_path:
                os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(final_result, f, indent=2, ensure_ascii=False)
                logger.info("Output written to: %s", output_path)

            return final_result
        finally:
            if self._retriever is not None:
                self._retriever.close()
                self._retriever = None

    def _multi_query_retrieve(
        self,
        queries: List[str],
        top_k: int = 10,
        is_spec: bool = False,
        mode: str = "generic",
    ) -> List[Dict[str, Any]]:
        """
        Run multiple retrieval queries and deduplicate by chunk_id.
        Different query phrasings hit different BM25 keyword matches.
        """
        seen_ids = set()
        results = []

        for query in queries:
            if is_spec or mode == "spec":
                hits = self._retriever.retrieve_spec_chunks(query, top_k=top_k)
            elif mode == "scope":
                hits = self._retriever.retrieve_scope_chunks(query, top_k=top_k)
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
        "scope_of_work": {"summary": "NOT_FOUND", "deliverables": [], "exclusions": [], "locations": [], "references": []},
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
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show pipeline log messages alongside the progress bar")
    parser.add_argument("--reindex", action="store_true", help="Force re-indexing of the vector store")

    args = parser.parse_args()

    # Keep CLI output focused on the main progress bar.
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # With -v show INFO from our own loggers; otherwise only hard errors.
    app_level = logging.INFO if args.verbose else logging.ERROR
    logging.basicConfig(
        level=logging.WARNING,          # root logger quiet by default
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    logging.getLogger("tender_extraction").setLevel(app_level)

    # Always silence noisy third-party debug loggers.
    _SUPPRESS = ("pdfminer", "pdfplumber", "PIL", "urllib3", "httpx",
                 "sentence_transformers", "transformers", "torch",
                 "huggingface_hub", "filelock")
    for _name in list(logging.Logger.manager.loggerDict.keys()):
        if any(_name.startswith(p) for p in _SUPPRESS):
            logging.getLogger(_name).setLevel(logging.ERROR)

    try:
        from transformers.utils import logging as _hf_logging
        _hf_logging.set_verbosity_error()
    except Exception:
        pass

    # Silence common HF warning spam so tqdm remains readable.
    warnings.filterwarnings(
        "ignore",
        message=r"Warning: You are sending unauthenticated requests to the HF Hub.*",
    )
    warnings.filterwarnings("ignore", module=r"huggingface_hub\.utils\._http")

    pipeline = TenderExtractionPipeline(force_reindex=args.reindex)

    try:
        from tqdm import tqdm as _tqdm
        _has_tqdm = True
    except ImportError:
        _has_tqdm = False

    try:
        if _has_tqdm:
            bar = _tqdm(
                total=100,
                desc="Starting",
                unit="%",
                ncols=80,
                bar_format="{desc:<35} {percentage:3.0f}%|{bar}| {elapsed}<{remaining}",
                file=sys.stderr,
                colour="green",
            )
            _last = [0]

            def _progress_cb(pct: int, msg: str):
                inc = max(0, pct - _last[0])
                if inc:
                    bar.update(inc)
                    _last[0] = pct
                bar.set_description(msg[:33])

            with open(os.devnull, "w", encoding="utf-8") as _sink:
                with contextlib.redirect_stdout(_sink):
                    result = pipeline.run(args.file, args.output, progress_callback=_progress_cb)

            # Finish bar to 100%.
            remaining = 100 - _last[0]
            if remaining:
                bar.update(remaining)
            bar.set_description("Done")
            bar.close()
        else:
            with open(os.devnull, "w", encoding="utf-8") as _sink:
                with contextlib.redirect_stdout(_sink):
                    result = pipeline.run(args.file, args.output)

        # Print final summary to stderr so it appears after the bar.
        final = result
        n_specs = len(final.get("technical_specifications", []))
        scope = final.get("scope_of_work", {})
        n_del = len(scope.get("deliverables", []))
        n_excl = len(scope.get("exclusions", []))
        acc = final.get("accuracy_score", 0.0)
        print(
            f"\n  Specs: {n_specs}   Deliverables: {n_del}   Exclusions: {n_excl}   "
            f"Accuracy: {acc:.1f}%",
            file=sys.stderr,
        )
        if args.output:
            print(f"  Output: {args.output}", file=sys.stderr)
        else:
            print(json.dumps(result, indent=2, ensure_ascii=False))

    except FileNotFoundError as exc:
        print(f"Error: File not found — {exc}", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
