"""
validation.py — Grounding verification using rapidfuzz.

Verifies each extraction is grounded in the source document by performing
fuzzy substring matching against the original chunk text.

Confidence thresholds map matching scores to HIGH, MEDIUM, and LOW, 
rejecting extractions that fall below the minimum threshold.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from rapidfuzz import fuzz

from tender_extraction.config import config
from tender_extraction.schemas import Chunk, ExtractionResult

logger = logging.getLogger(__name__)


def verify_grounding(
    spec: Dict[str, Any],
    source_chunks: List[Dict[str, Any]],
) -> float:
    """
    Check how well a single spec's citation matches the source chunks.

    Uses rapidfuzz.fuzz.partial_ratio to find the best fuzzy match of
    the spec's cited exact_text against all source chunk texts.

    Returns a float 0.0 to 1.0 representing match quality.
    """
    exact_text = spec.get("source", {}).get("exact_text", "")
    if not exact_text or exact_text == "NOT_FOUND":
        return 0.0

    best_score = 0.0
    for item in source_chunks:
        chunk: Chunk = item["chunk"]
        ratio = fuzz.token_sort_ratio(
            exact_text.lower().strip(),
            chunk.text.lower().strip(),
        )
        score = ratio / 100.0
        if score > best_score:
            best_score = score

    return best_score


def assign_confidence(score: float) -> str:
    """Map grounding score to confidence bucket."""
    if score >= config.validation.high_confidence_threshold:
        return "HIGH"
    elif score >= config.validation.medium_confidence_threshold:
        return "MEDIUM"
    return "LOW"


def validate_extractions(
    extraction: Dict[str, Any],
    source_chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Full validation pass: grounding + confidence + NOT_FOUND enforcement.

    Drops specs with grounding score < 0.40 (logged as warning).
    Updates confidence field based on actual grounding score.
    Replaces empty/None fields with "NOT_FOUND".
    """
    # Validate specs
    raw_specs = extraction.get("technical_specifications", [])
    validated_specs: List[Dict[str, Any]] = []

    total_score = 0.0
    accepted_count = 0

    for spec in raw_specs:
        score = verify_grounding(spec, source_chunks)
        confidence = assign_confidence(score)

        if score < config.validation.min_grounding_ratio - 1e-9:
            logger.warning(
                "REJECTED spec '%s' (grounding=%.2f < threshold=%.2f). "
                "Likely hallucination.",
                spec.get("item_name", "?"), score, config.validation.min_grounding_ratio,
            )
            continue

        total_score += score
        accepted_count += 1
        spec["confidence"] = confidence
        spec = _enforce_not_found(spec)
        validated_specs.append(spec)

    # Validate scope tasks (lighter check — just verify the chunk exists)
    scope = extraction.get("scope_of_work", {})
    validated_tasks: List[Dict[str, Any]] = []
    for task in scope.get("tasks", []):
        task_score = _verify_task_grounding(task, source_chunks)
        if task_score < config.validation.min_grounding_ratio - 1e-9:
            logger.warning(
                "REJECTED task '%s' (grounding=%.2f)",
                task.get("task_description", "?")[:60], task_score,
            )
            continue
        total_score += task_score
        accepted_count += 1
        task = _enforce_not_found(task)
        validated_tasks.append(task)

    validated_exclusions = scope.get("exclusions", [])

    # Stats
    specs_rejected = len(raw_specs) - len(validated_specs)
    tasks_rejected = len(scope.get("tasks", [])) - len(validated_tasks)
    high_count = sum(1 for s in validated_specs if s.get("confidence") == "HIGH")
    med_count = sum(1 for s in validated_specs if s.get("confidence") == "MEDIUM")
    low_count = sum(1 for s in validated_specs if s.get("confidence") == "LOW")

    logger.info(
        "Validation: %d/%d specs passed (HIGH=%d, MEDIUM=%d, LOW=%d), "
        "%d/%d tasks passed, %d rejected",
        len(validated_specs), len(raw_specs), high_count, med_count, low_count,
        len(validated_tasks), len(scope.get("tasks", [])),
        specs_rejected + tasks_rejected,
    )

    overall_accuracy = (total_score / accepted_count) if accepted_count > 0 else 0.0

    return {
        "technical_specifications": validated_specs,
        "scope_of_work": {
            "tasks": validated_tasks,
            "exclusions": validated_exclusions,
        },
        "accuracy_score": round(overall_accuracy * 100, 2),
    }


def validate_schema(raw: Dict[str, Any]) -> ExtractionResult:
    """Pydantic validation — catches type mismatches and missing fields."""
    try:
        from pydantic import BaseModel
        _pydantic_v2 = hasattr(BaseModel, 'model_validate')
    except ImportError:
        _pydantic_v2 = False

    if _pydantic_v2:
        return ExtractionResult.model_validate(raw)
    else:
        return ExtractionResult.parse_obj(raw)


def _verify_task_grounding(
    task: Dict[str, Any], source_chunks: List[Dict[str, Any]]
) -> float:
    """Lighter grounding check for scope tasks."""
    desc = task.get("task_description", "")
    exact = task.get("source", {}).get("exact_text", desc)
    text_to_check = exact if exact and exact != "NOT_FOUND" else desc

    if not text_to_check:
        return 0.0

    best = 0.0
    for item in source_chunks:
        chunk: Chunk = item["chunk"]
        ratio = fuzz.token_sort_ratio(
            text_to_check.lower().strip(),
            chunk.text.lower().strip(),
        ) / 100.0
        if ratio > best:
            best = ratio
    return best


def _enforce_not_found(data: Dict[str, Any]) -> Dict[str, Any]:
    """Replace None/empty string values with 'NOT_FOUND' for consistency."""
    FIELDS = {"unit", "numeric_value", "tolerance", "standard_reference", "material", "timeline"}
    for field in FIELDS:
        if field in data:
            val = data[field]
            if val is None or (isinstance(val, str) and not val.strip()):
                data[field] = "NOT_FOUND"
    return data


# ── Smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    # Test grounding with real data
    from tender_extraction.schemas import ChunkMetadata

    # Simulate a real chunk and a real extraction
    real_chunk = Chunk(
        chunk_id="chunk_test_001",
        text="Steel reinforcement bars shall be Grade 60 conforming to ASTM A615 standard.",
        metadata=ChunkMetadata(page=15, section="3.1 Materials"),
    )
    source_chunks = [{"chunk": real_chunk, "score": 0.9}]

    # This spec should PASS grounding (text exists in source)
    good_spec = {
        "item_name": "Steel Bars",
        "specification_text": "Grade 60 conforming to ASTM A615",
        "source": {
            "chunk_id": "chunk_test_001",
            "page": 15,
            "exact_text": "Steel reinforcement bars shall be Grade 60 conforming to ASTM A615",
        },
    }

    # This spec should FAIL grounding (hallucinated)
    bad_spec = {
        "item_name": "Copper Wire",
        "specification_text": "Pure copper 99.9% purity",
        "source": {
            "chunk_id": "chunk_fake",
            "page": 99,
            "exact_text": "Copper wire meeting international standards",
        },
    }

    extraction = {
        "technical_specifications": [good_spec, bad_spec],
        "scope_of_work": {"tasks": [], "exclusions": []},
    }

    result = validate_extractions(extraction, source_chunks)
    specs = result["technical_specifications"]
    print(f"Input: 2 specs, Output: {len(specs)} specs (1 should be rejected)")
    for s in specs:
        print(f"  - {s['item_name']}: confidence={s['confidence']}")

    assert len(specs) == 1, f"Expected 1 spec (bad should be rejected), got {len(specs)}"
    assert specs[0]["item_name"] == "Steel Bars"
    print("\nValidation smoke test passed.")
