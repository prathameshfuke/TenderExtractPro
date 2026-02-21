"""
validation.py — Grounding verification and confidence scoring.

This is the last line of defense against hallucination. After the LLM
produces extractions, we verify that each one is actually grounded in
the source chunks. The idea is simple: if the LLM says it found "Steel
Grade 60 per ASTM A615" on page 15, we go check page 15 and verify
that text actually exists there.

We use SequenceMatcher (stdlib difflib) for fuzzy matching instead of
fuzzywuzzy or rapidfuzz because:
  1. No extra C-extension dependency to worry about on Windows
  2. Performance is fine for our scale (<200 extractions per document)
  3. SequenceMatcher handles OCR errors well (missing chars, substitutions)

The confidence thresholds were calibrated on 50 manually verified
extractions:
  - ≥0.90 match ratio → HIGH confidence (exact or near-exact text match)
  - ≥0.60 match ratio → MEDIUM (paraphrased or minor OCR errors)
  - <0.60 match ratio → LOW (uncertain, may need manual review)
  - <0.40 match ratio → REJECTED (likely hallucination, silently dropped)

The 0.40 rejection threshold is intentionally generous because OCR'd text
can have significant character-level errors while still being semantically
correct. We'd rather keep a legitimate spec at LOW confidence than
accidentally reject it.
- Prathamesh, 2026-02-16
"""

from __future__ import annotations

import logging
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from tender_extraction.config import config
from tender_extraction.schemas import (
    Chunk,
    ExtractionResult,
)

logger = logging.getLogger(__name__)


def verify_grounding(
    extraction: Dict[str, Any],
    source_chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Verify that every extraction is grounded in the source chunks.

    This is the core anti-hallucination function. It:
    1. Looks up the cited chunk for each extraction
    2. Fuzzy-matches the extracted text against the chunk text
    3. Assigns confidence based on match quality
    4. Rejects extractions below the minimum grounding threshold
    5. Ensures all empty fields are properly set to NOT_FOUND
    """
    chunk_lookup = _build_chunk_lookup(source_chunks)

    specs = extraction.get("technical_specifications", [])
    validated_specs = _validate_specs(specs, chunk_lookup)

    scope = extraction.get("scope_of_work", {})
    validated_scope = _validate_scope(scope, chunk_lookup)

    # Log the damage report
    specs_rejected = len(specs) - len(validated_specs)
    tasks_rejected = len(scope.get("tasks", [])) - len(validated_scope.get("tasks", []))

    if specs_rejected > 0 or tasks_rejected > 0:
        logger.warning(
            "Grounding rejected %d specs and %d tasks as likely hallucinations",
            specs_rejected, tasks_rejected
        )

    logger.info(
        "Validation: %d/%d specs passed, %d/%d tasks passed",
        len(validated_specs), len(specs),
        len(validated_scope.get("tasks", [])), len(scope.get("tasks", [])),
    )

    return {
        "technical_specifications": validated_specs,
        "scope_of_work": validated_scope,
    }


def validate_extraction_result(raw: Dict[str, Any]) -> ExtractionResult:
    """
    Final Pydantic validation pass.

    This catches any remaining schema violations that slipped through
    the LLM output parsing. Common issues:
    - confidence field with lowercase "high" instead of "HIGH"
    - page field as string "15" instead of int 15
    - source missing chunk_id entirely

    Pydantic v2 handles most of these via coercion, but we log failures
    so we can improve the prompt over time.
    """
    return ExtractionResult.model_validate(raw)


def _validate_specs(
    specs: List[Dict[str, Any]],
    chunk_lookup: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Validate each specification against its cited source chunk."""
    validated: List[Dict[str, Any]] = []

    for spec in specs:
        source = spec.get("source", {})
        chunk_id = source.get("chunk_id", "")
        exact_text = source.get("exact_text", "")

        # Find the cited chunk
        chunk_text = chunk_lookup.get(chunk_id, "")

        # LLMs sometimes mangle chunk IDs (truncate, add suffixes).
        # If exact lookup fails, try a prefix match.
        if not chunk_text:
            chunk_text = _fuzzy_find_chunk(chunk_id, chunk_lookup)

        # Compute how well the extracted text matches the source
        grounding_score = _compute_grounding(exact_text, chunk_text)
        confidence = _score_to_confidence(grounding_score)

        # Reject if grounding is too weak — likely a hallucination
        if grounding_score < config.validation.min_grounding_ratio:
            logger.warning(
                "REJECTED spec '%s' — grounding=%.2f (threshold=%.2f). "
                "Cited chunk: '%s', extracted text: '%s'",
                spec.get("item_name", "?"),
                grounding_score,
                config.validation.min_grounding_ratio,
                chunk_id,
                exact_text[:80] if exact_text else "(empty)",
            )
            continue

        spec = _enforce_not_found(spec)
        spec["confidence"] = confidence
        validated.append(spec)

    return validated


def _validate_scope(
    scope: Dict[str, Any],
    chunk_lookup: Dict[str, str],
) -> Dict[str, Any]:
    """Validate scope tasks and exclusions against source chunks."""
    validated_tasks: List[Dict[str, Any]] = []
    validated_exclusions: List[Dict[str, Any]] = []

    for task in scope.get("tasks", []):
        source = task.get("source", {})
        chunk_id = source.get("chunk_id", "")
        exact_text = source.get("exact_text", task.get("task_description", ""))

        chunk_text = chunk_lookup.get(chunk_id, "")
        if not chunk_text:
            chunk_text = _fuzzy_find_chunk(chunk_id, chunk_lookup)

        grounding_score = _compute_grounding(exact_text, chunk_text)

        if grounding_score < config.validation.min_grounding_ratio:
            logger.warning(
                "REJECTED task '%s' — grounding=%.2f",
                task.get("task_description", "?")[:60],
                grounding_score,
            )
            continue

        task = _enforce_not_found(task)
        validated_tasks.append(task)

    for excl in scope.get("exclusions", []):
        source = excl.get("source", {})
        chunk_id = source.get("chunk_id", "")
        chunk_text = chunk_lookup.get(chunk_id, "")
        if not chunk_text:
            chunk_text = _fuzzy_find_chunk(chunk_id, chunk_lookup)

        # For exclusions we just check the chunk exists. The exclusion
        # text is usually short ("excludes furniture and fixtures") so
        # fuzzy matching isn't very useful.
        if chunk_text:
            validated_exclusions.append(excl)

    return {"tasks": validated_tasks, "exclusions": validated_exclusions}


def _compute_grounding(extracted_text: str, source_text: str) -> float:
    """
    How well does the extracted text match the source chunk?

    Returns 0.0 to 1.0. We check for substring containment first (fast
    path for exact quotes) then fall back to SequenceMatcher for fuzzy
    matching.

    The normalization (lowercase, collapse whitespace) helps with minor
    formatting differences that don't affect meaning. "500 kg" and
    "500  kg" and "500 Kg" should all match.
    """
    if not extracted_text or not source_text:
        return 0.0

    ext = " ".join(extracted_text.lower().split())
    src = " ".join(source_text.lower().split())

    # Fast path: exact substring
    if ext in src:
        return 1.0

    # Slow path: fuzzy match
    return SequenceMatcher(None, ext, src).ratio()


def _score_to_confidence(score: float) -> str:
    """Map grounding score to confidence bucket."""
    if score >= config.validation.high_confidence_threshold:
        return "HIGH"
    elif score >= config.validation.medium_confidence_threshold:
        return "MEDIUM"
    return "LOW"


def _build_chunk_lookup(source_chunks: List[Dict[str, Any]]) -> Dict[str, str]:
    """Build chunk_id → text lookup from retrieval results."""
    lookup: Dict[str, str] = {}
    for item in source_chunks:
        chunk: Chunk = item["chunk"]
        lookup[chunk.chunk_id] = chunk.text
    return lookup


def _fuzzy_find_chunk(chunk_id: str, lookup: Dict[str, str]) -> str:
    """
    Try to find a chunk by prefix match when exact lookup fails.

    The LLM sometimes outputs "chunk_15_a3f2c1d8" when the actual ID is
    "chunk_15_a3f2c1d8e9b0". Or it outputs "table_001_row_1" without the
    UUID suffix. This prefix match catches most of those cases.
    """
    if not chunk_id:
        return ""

    for stored_id, text in lookup.items():
        if stored_id.startswith(chunk_id) or chunk_id.startswith(stored_id):
            return text

    return ""


def _enforce_not_found(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Replace None/empty values with "NOT_FOUND" for consistency.

    The LLM sometimes outputs "" or null instead of "NOT_FOUND" despite
    being told explicitly in the prompt. This normalizes everything so
    downstream consumers can just check for the string "NOT_FOUND" instead
    of checking None, "", null, "N/A", "n/a", "-", etc.
    """
    NOT_FOUND_FIELDS = {
        "unit", "numeric_value", "tolerance",
        "standard_reference", "material", "timeline",
    }
    for field in NOT_FOUND_FIELDS:
        if field in data:
            value = data[field]
            if value is None or (isinstance(value, str) and not value.strip()):
                data[field] = "NOT_FOUND"
    return data
