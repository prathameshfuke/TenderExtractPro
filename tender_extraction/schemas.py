"""
schemas.py — Pydantic v2 models with strict validation.

These models define the extraction contract. Fields map exactly
to expected extraction outputs with appropriate defaults.
"""

from __future__ import annotations
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, validator


class SourceCitation(BaseModel):
    """Where in the document an extraction came from."""
    chunk_id: str = Field(..., description="ID of the source chunk")
    page: int = Field(..., description="1-based page number")
    exact_text: str = Field(
        default="NOT_FOUND",
        description="Verbatim quote from the source chunk",
    )


class TechnicalSpecification(BaseModel):
    """A single technical spec extracted from a tender."""
    item_name: str
    specification_text: str
    unit: str = Field(default="NOT_FOUND")
    numeric_value: str = Field(default="NOT_FOUND")
    tolerance: str = Field(default="NOT_FOUND")
    standard_reference: str = Field(default="NOT_FOUND")
    material: str = Field(default="NOT_FOUND")
    source: SourceCitation
    confidence: Literal["HIGH", "MEDIUM", "LOW"] = Field(default="LOW")

    @validator("specification_text")
    @classmethod
    def spec_text_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("specification_text cannot be empty or whitespace")
        return v


class ScopeTask(BaseModel):
    """A single task within the scope of work."""
    task_description: str
    deliverables: List[str] = Field(default_factory=list)
    timeline: str = Field(default="NOT_FOUND")
    dependencies: List[str] = Field(default_factory=list)
    source: SourceCitation


class Exclusion(BaseModel):
    """An item explicitly excluded from the scope."""
    item: str
    source: SourceCitation


class ScopeOfWork(BaseModel):
    """Complete scope-of-work section."""
    tasks: List[ScopeTask] = Field(default_factory=list)
    exclusions: List[Exclusion] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    """Top-level output of the entire pipeline."""
    technical_specifications: List[TechnicalSpecification] = Field(default_factory=list)
    scope_of_work: ScopeOfWork = Field(default_factory=ScopeOfWork)
    accuracy_score: float = Field(default=0.0, description="Overall accuracy/grounding score percentage (0-100)")


# Internal models used within the pipeline

class ChunkMetadata(BaseModel):
    """Metadata attached to every chunk."""
    section: str = Field(default="Unknown")
    parent_section: str = Field(default="Unknown")
    chunk_type: Literal["table", "paragraph", "list", "image_ocr"] = Field(default="paragraph")
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

    # Test 1: valid spec
    spec = TechnicalSpecification(
        item_name="Steel Bars",
        specification_text="Grade 60 conforming to ASTM A615",
        unit="kg",
        source=SourceCitation(chunk_id="chunk_001", page=15),
    )
    assert spec.unit == "kg"
    assert spec.tolerance == "NOT_FOUND"
    assert spec.confidence == "LOW"
    print("Test 1 passed: valid spec with defaults")

    # Test 2: empty spec_text should fail
    try:
        bad = TechnicalSpecification(
            item_name="Bad",
            specification_text="   ",
            source=SourceCitation(chunk_id="x", page=1),
        )
        print("Test 2 FAILED: should have raised ValidationError")
        sys.exit(1)
    except Exception:
        print("Test 2 passed: empty spec_text rejected")

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
