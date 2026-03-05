"""
schemas.py — Pydantic v2 models with strict validation.

Output schema matches the recommended tender extraction format:
  technical_specifications: list of component + specs-dict + source + confidence
  scope_of_work: summary + deliverables + exclusions + locations + references
"""

from __future__ import annotations
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class SourceCitation(BaseModel):
    """Where in the document an extraction came from."""
    chunk_id: str = Field(default="NOT_FOUND", description="ID of the source chunk")
    page: int = Field(default=0, description="1-based page number")
    clause: str = Field(default="NOT_FOUND", description="Clause or section number (e.g. '3.1.2')")
    exact_text: str = Field(
        default="NOT_FOUND",
        description="Verbatim quote from the source chunk",
    )


class TechnicalSpecification(BaseModel):
    """
    A single extracted technical specification.

    `component` is the name of the part/system/material.
    `specs` is a flat dict of parameter → value-with-units, e.g.
        {"capacity": "10 TR", "power_supply": "415 V 3-phase 50 Hz"}.
    `confidence` is a 0-1 float set by the grounding verifier.
    """
    component: str
    specs: Dict[str, str] = Field(default_factory=dict)
    source: SourceCitation = Field(default_factory=SourceCitation)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("component")
    @classmethod
    def component_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("component cannot be empty or whitespace")
        return v.strip()


class ScopeOfWork(BaseModel):
    """
    Complete scope-of-work section.

    `summary`     — concise description (≤120 words).
    `deliverables`— list of deliverable strings.
    `exclusions`  — list of items explicitly NOT in scope.
    `locations`   — site/building locations mentioned.
    `references`  — clause/page references cited in the scope section.
    """
    summary: str = Field(default="NOT_FOUND")
    deliverables: List[str] = Field(default_factory=list)
    exclusions: List[str] = Field(default_factory=list)
    locations: List[str] = Field(default_factory=list)
    references: List[str] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    """Top-level output of the entire pipeline."""
    technical_specifications: List[TechnicalSpecification] = Field(default_factory=list)
    scope_of_work: ScopeOfWork = Field(default_factory=ScopeOfWork)
    accuracy_score: float = Field(default=0.0, description="Overall grounding score 0-100")


# ── Internal models ───────────────────────────────────────────────────────

class ChunkMetadata(BaseModel):
    """Metadata attached to every chunk."""
    section: str = Field(default="Unknown")
    parent_section: str = Field(default="Unknown")
    chunk_type: str = Field(default="paragraph")   # table | paragraph | list | image_ocr
    page: int = Field(default=0)
    table_id: Optional[str] = Field(default=None)
    bbox: Optional[List[float]] = Field(default=None)


class Chunk(BaseModel):
    """A single chunk produced by the chunking stage."""
    chunk_id: str
    text: str
    metadata: ChunkMetadata


# ── Smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    spec = TechnicalSpecification(
        component="HVAC Unit",
        specs={"capacity": "10 TR", "power_supply": "415 V 3-phase 50 Hz"},
        source=SourceCitation(chunk_id="chunk_001", page=5, clause="3.2"),
        confidence=0.94,
    )
    assert spec.component == "HVAC Unit"
    assert spec.specs["capacity"] == "10 TR"
    print("Test 1 passed: valid TechnicalSpecification")

    scope = ScopeOfWork(
        summary="Supply and install 10 HVAC units.",
        deliverables=["Supply of 10 HVAC units", "Commissioning"],
        exclusions=["Civil works not included"],
        locations=["Building A"],
        references=["Clause 2, pages 4-6"],
    )
    assert scope.exclusions[0] == "Civil works not included"
    print("Test 2 passed: valid ScopeOfWork")

    try:
        bad = TechnicalSpecification(component="   ", specs={})
        print("Test 3 FAILED: should have raised ValidationError")
        sys.exit(1)
    except Exception:
        print("Test 3 passed: empty component rejected")


    # Test 3: full ExtractionResult round-trip
    result = ExtractionResult(
        technical_specifications=[spec],
        scope_of_work=ScopeOfWork(
            tasks=[ScopeTask(
                task_description="Site prep",
                source=SourceCitation(chunk_id="c2", page=8),
            )],
        ),
        accuracy_score=95.5,
    )
    data = result.dict()
    assert len(data["technical_specifications"]) == 1
    assert data["scope_of_work"]["tasks"][0]["timeline"] == "NOT_FOUND"
    assert data["accuracy_score"] == 95.5
    print("Test 3 passed: full ExtractionResult round-trip")

    print("\nAll schema tests passed.")
