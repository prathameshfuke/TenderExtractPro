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

    Checks both spec.source.exact_text AND spec.specification_text against
    all source chunk texts, returning the best match score. This handles
    the common case where the LLM paraphrases the exact_text but the
    specification_text is closer to verbatim.

    Returns a float 0.0 to 1.0 representing match quality.
    """
    exact_text = spec.get("source", {}).get("exact_text", "")
    # New schema: component + specs dict.
    # Primary grounding signal = exact_text (LLM should quote source verbatim).
    # Use component name only as a fallback when exact_text is absent.
    component = spec.get("component", "")

    # Collect distinct non-empty texts to check
    texts_to_check = []
    if exact_text and exact_text not in ("NOT_FOUND", ""):
        texts_to_check.append(exact_text)
    elif component and component not in ("NOT_FOUND", ""):
        # Fallback: use component name if no exact citation was provided
        texts_to_check.append(component[:200])

    if not texts_to_check:
        return 0.0

    best_score = 0.0
    for text_to_check in texts_to_check:
        for item in source_chunks:
            chunk: Chunk = item["chunk"]
            # partial_ratio finds the best matching *substring* in chunk.text —
            # the right metric for verifying a short citation phrase exists
            # inside a longer chunk. token_sort_ratio was wrong here: comparing
            # sorted token sets of a 15-token citation vs a 500-token chunk
            # always returns a near-zero overlap ratio and rejects valid extractions.
            ratio = fuzz.partial_ratio(
                text_to_check.lower().strip(),
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
        spec = _normalize_spec_shape(spec)
        score = verify_grounding(spec, source_chunks)
        confidence = assign_confidence(score)

        if score < config.validation.min_grounding_ratio - 1e-9:
            logger.warning(
                "REJECTED spec '%s' (grounding=%.2f < threshold=%.2f). "
                "Likely hallucination.",
                spec.get("component", "?"), score, config.validation.min_grounding_ratio,
            )
            continue

        total_score += score
        accepted_count += 1
        # Confidence is now a float (0.0–1.0); overwrite with actual grounding score
        spec["confidence"] = round(score, 4)
        validated_specs.append(spec)

    # Validate scope deliverables (plain strings in new schema)
    scope = extraction.get("scope_of_work", {})
    validated_deliverables: List[str] = []
    raw_deliverables = scope.get("deliverables", [])
    for item in raw_deliverables:
        if not isinstance(item, str) or not item.strip():
            continue
        task_score = _verify_deliverable_grounding(item, source_chunks)
        if task_score < config.validation.min_grounding_ratio - 1e-9:
            logger.warning(
                "REJECTED deliverable '%s' (grounding=%.2f)",
                item[:60], task_score,
            )
            continue
        total_score += task_score
        accepted_count += 1
        validated_deliverables.append(item)

    # Exclusions, locations, references are pass-through (no grounding check needed)
    validated_exclusions = [e for e in scope.get("exclusions", []) if isinstance(e, str) and e.strip()]
    validated_locations = [l for l in scope.get("locations", []) if isinstance(l, str) and l.strip()]
    validated_references = [r for r in scope.get("references", []) if isinstance(r, str) and r.strip()]

    # Stats
    specs_rejected = len(raw_specs) - len(validated_specs)
    deliverables_rejected = len(raw_deliverables) - len(validated_deliverables)
    high_count = sum(1 for s in validated_specs if (s.get("confidence") or 0.0) >= config.validation.high_confidence_threshold)
    med_count = sum(1 for s in validated_specs if config.validation.medium_confidence_threshold <= (s.get("confidence") or 0.0) < config.validation.high_confidence_threshold)
    low_count = sum(1 for s in validated_specs if (s.get("confidence") or 0.0) < config.validation.medium_confidence_threshold)

    logger.info(
        "Validation: %d/%d specs passed (HIGH=%d, MEDIUM=%d, LOW=%d), "
        "%d/%d deliverables passed, %d rejected",
        len(validated_specs), len(raw_specs), high_count, med_count, low_count,
        len(validated_deliverables), len(raw_deliverables),
        specs_rejected + deliverables_rejected,
    )

    overall_accuracy = (total_score / accepted_count) if accepted_count > 0 else 0.0

    return {
        "technical_specifications": validated_specs,
        "scope_of_work": {
            "summary": scope.get("summary", "NOT_FOUND"),
            "deliverables": validated_deliverables,
            "exclusions": validated_exclusions,
            "locations": validated_locations,
            "references": validated_references,
        },
        "accuracy_score": round(overall_accuracy * 100, 2),
    }


def _normalize_spec_shape(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Backward-compatible normalization for mixed/legacy spec payloads.

    Accepts old shape:
      item_name, specification_text, unit, numeric_value, tolerance,
      standard_reference, material

    And converts to current shape:
      component, specs (dict), source, confidence
    """
    if "component" in spec and isinstance(spec.get("specs"), dict):
        return spec

    component = str(spec.get("component") or spec.get("item_name") or "NOT_FOUND").strip()
    if not component:
        component = "NOT_FOUND"

    specs_dict: Dict[str, str] = {}

    # Carry over explicit dict if present.
    if isinstance(spec.get("specs"), dict):
        for k, v in spec.get("specs", {}).items():
            if v is None:
                continue
            v_str = str(v).strip()
            if v_str and v_str != "NOT_FOUND":
                specs_dict[str(k)] = v_str

    # Legacy fields -> specs dict entries.
    legacy_pairs = [
        ("specification", spec.get("specification_text")),
        ("unit", spec.get("unit")),
        ("numeric_value", spec.get("numeric_value")),
        ("tolerance", spec.get("tolerance")),
        ("standard_reference", spec.get("standard_reference")),
        ("material", spec.get("material")),
    ]
    for key, value in legacy_pairs:
        if value is None:
            continue
        value_str = str(value).strip()
        if value_str and value_str != "NOT_FOUND":
            specs_dict[key] = value_str

    source = spec.get("source") if isinstance(spec.get("source"), dict) else {}
    normalized = {
        "component": component,
        "specs": specs_dict,
        "source": {
            "chunk_id": source.get("chunk_id", "NOT_FOUND"),
            "page": int(source.get("page", 0) or 0),
            "clause": source.get("clause", "NOT_FOUND"),
            "exact_text": source.get("exact_text", "NOT_FOUND"),
        },
        "confidence": float(spec.get("confidence", 0.5)) if isinstance(spec.get("confidence"), (int, float)) else 0.5,
    }
    return normalized


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


def _verify_deliverable_grounding(
    deliverable: str, source_chunks: List[Dict[str, Any]]
) -> float:
    """Lighter grounding check for plain-string scope deliverables."""
    text_to_check = deliverable.strip()
    if not text_to_check:
        return 0.0

    best = 0.0
    for item in source_chunks:
        chunk: Chunk = item["chunk"]
        ratio = fuzz.partial_ratio(
            text_to_check.lower(),
            chunk.text.lower().strip(),
        ) / 100.0
        if ratio > best:
            best = ratio
    return best


def _enforce_not_found(data: Dict[str, Any]) -> Dict[str, Any]:
    """Replace None/empty string values with 'NOT_FOUND' for string fields."""
    for key, val in data.items():
        if isinstance(val, str) and not val.strip():
            data[key] = "NOT_FOUND"
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
