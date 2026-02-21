"""
schemas.py — Pydantic v2 models that define the extraction contract.

These models serve three purposes:
1. Constrained LLM output validation (anti-hallucination layer)
2. API response serialization
3. NOT_FOUND defaults so downstream code never has to check for None

We moved to Pydantic v2 from v1 in Feb 2026 because v2's model_validate()
is ~3x faster — matters when we're validating 200+ specs per document.
The migration was mostly painless except for the validator decorator
rename (validator → field_validator). - Prathamesh, 2026-02-12
"""

from __future__ import annotations
from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class SourceCitation(BaseModel):
    """
    Where in the document an extraction came from.
    
    Every single extracted field MUST have one of these. If the LLM
    doesn't provide a citation, the extraction gets rejected in the
    validation stage. This was the single biggest improvement to our
    hallucination rate — dropped from ~15% to ~3% just by making
    citations mandatory.
    """
    chunk_id: str = Field(..., description="ID of the source chunk")
    page: int = Field(..., description="1-based page number in original document")
    exact_text: str = Field(
        default="NOT_FOUND",
        description="Verbatim quote from the chunk. Used for grounding verification.",
    )


class TechnicalSpecification(BaseModel):
    """
    A single technical specification pulled from a tender document.
    
    The field list was designed around what procurement teams actually
    need when evaluating bids. We worked with 3 procurement officers to
    nail down these fields. Originally we had 15 fields but cut it to
    these 7 because the LLM accuracy dropped significantly beyond 7
    fields per extraction — it starts confusing tolerance with unit.
    """
    item_name: str
    specification_text: str
    unit: str = Field(default="NOT_FOUND")
    numeric_value: str = Field(default="NOT_FOUND")
    tolerance: str = Field(default="NOT_FOUND")
    standard_reference: str = Field(default="NOT_FOUND")
    material: str = Field(default="NOT_FOUND")
    source: SourceCitation
    confidence: Literal["HIGH", "MEDIUM", "LOW"] = Field(default="LOW")


class ScopeTask(BaseModel):
    """A single task within the scope of work."""
    task_description: str
    deliverables: List[str] = Field(default_factory=list)
    timeline: str = Field(default="NOT_FOUND")
    dependencies: List[str] = Field(default_factory=list)
    source: SourceCitation


class Exclusion(BaseModel):
    """
    An item explicitly excluded from scope.
    
    Tracking exclusions matters a LOT — contractors have filed claims
    worth crores because exclusions weren't captured properly from the
    original tender. That's why we extract these separately.
    """
    item: str
    source: SourceCitation


class ScopeOfWork(BaseModel):
    """Complete scope-of-work section."""
    tasks: List[ScopeTask] = Field(default_factory=list)
    exclusions: List[Exclusion] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    """
    Top-level output of the entire pipeline.
    
    This is what gets serialized to JSON and handed to the user.
    We keep it flat (just specs + scope) rather than nesting deeper
    because every level of nesting made the LLM output less reliable.
    """
    technical_specifications: List[TechnicalSpecification] = Field(
        default_factory=list
    )
    scope_of_work: ScopeOfWork = Field(default_factory=ScopeOfWork)


# -- Internal models used within the pipeline, not exposed to users --

class ChunkMetadata(BaseModel):
    """Rich metadata attached to every chunk for retrieval context."""
    section: str = Field(default="Unknown")
    parent_section: str = Field(default="Unknown")
    chunk_type: Literal["table", "paragraph", "list", "image_ocr"] = Field(
        default="paragraph"
    )
    page: int = Field(default=0)
    table_id: Optional[str] = Field(default=None)
    bbox: Optional[List[float]] = Field(default=None)


class Chunk(BaseModel):
    """A single chunk produced by the chunking stage."""
    chunk_id: str
    text: str
    metadata: ChunkMetadata
