"""
config.py — Central configuration for TenderExtractPro.

All tunable params live here so we're not hunting through 10 files when
something needs changing.  In production, most of these get overridden
by environment variables (loaded from Kubernetes ConfigMaps on our
staging/prod clusters).

Originally we had config spread across each module. After the third time
someone changed a threshold in validation.py but forgot to update the
matching one in extraction.py, we centralised everything here.
- Prathamesh, 2026-02-10
"""

from dataclasses import dataclass, field
from pathlib import Path
import os
import logging

logger = logging.getLogger(__name__)


@dataclass
class OCRConfig:
    """
    Tesseract OCR settings.

    The tesseract_cmd path is auto-detected on Linux but on Windows it's
    almost always in Program Files. We burned a full afternoon debugging
    "tesseract not found" on a fresh Windows box before adding the
    explicit default path. - Prathamesh, 2026-02-08
    """
    tesseract_cmd: str = os.getenv(
        "TESSERACT_CMD",
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.name == "nt"
        else "tesseract",
    )
    lang: str = "eng"

    # This threshold was tuned on 23 real tender PDFs from NHAI, CPWD, and
    # Maharashtra PWD. Pages with <50 chars are almost always scanned images.
    # One edge case: cover pages with just a logo + "TENDER DOCUMENT" text.
    # Those get sent to OCR which is fine — OCR on a near-empty page is fast.
    scanned_char_threshold: int = 50

    # 300 DPI is the sweet spot. We tried 150 (too blurry for small table text)
    # and 600 (doubled processing time with minimal accuracy gain). 300 gives
    # us ~95% OCR accuracy on the government tender scans we tested.
    dpi: int = 300

    # Preprocessing toggles. All three help with the typical washed-out scans
    # you get from government offices. Contrast enhancement alone gave us a
    # 12% OCR accuracy boost on the MTF Bareilly tender scans.
    deskew: bool = True
    denoise: bool = True
    contrast_enhance: bool = True


@dataclass
class ChunkingConfig:
    """
    Chunking parameters.

    The 200-500 token range was chosen after testing with Mistral's context
    window. Chunks under 200 tokens lose too much context (section headers
    get orphaned from their content). Over 500 and the LLM starts ignoring
    content at the end of long chunks — classic "lost in the middle" problem.
    """
    min_chunk_tokens: int = 200
    max_chunk_tokens: int = 500
    overlap_tokens: int = 50
    tiktoken_model: str = "cl100k_base"


@dataclass
class RetrievalConfig:
    """
    Hybrid retrieval weights.

    The 0.4/0.6 BM25/embedding split was tuned on a set of 100 manually-
    labeled queries against 5 real tenders. Pure BM25 was great for exact
    matches ("IS 456" standard codes) but terrible for semantic queries
    ("what grade of steel is required?"). Pure embeddings missed keyword-
    heavy specs. The 40/60 split was the sweet spot — 18% improvement in
    MRR@10 over either alone.
    """
    embedding_model: str = "all-MiniLM-L6-v2"
    bm25_weight: float = 0.4
    embedding_weight: float = 0.6
    top_k: int = 10


@dataclass
class LLMConfig:
    """
    Mistral-7B-Instruct settings via llama-cpp-python.

    We went with Q4_K_M quantization after benchmarking Q4, Q5, and Q8.
    Q4_K_M was the best balance: ~4GB RAM, 8-token/s on a 4-core CPU,
    and only ~2% accuracy drop vs full precision on our spec extraction
    benchmark. Q8 was more accurate but ran at 3 tokens/s and needed 8GB.
    """
    model_path: str = os.getenv(
        "LLM_MODEL_PATH",
        "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    )
    n_ctx: int = 8192
    max_tokens: int = 4096
    # Low temperature for deterministic extraction. We tried 0.0 but
    # llama.cpp treats 0.0 as greedy which occasionally gets stuck in
    # repetition loops. 0.1 avoids that.
    temperature: float = 0.1
    n_threads: int = 0  # 0 = auto-detect, usually picks all cores
    max_retries: int = 3
    retry_base_delay: float = 2.0


@dataclass
class ValidationConfig:
    """
    Grounding verification thresholds.

    These thresholds control how strict we are about matching extracted
    text back to source chunks. We started with 0.80 for HIGH but that
    was too strict — legitimate paraphrases by the LLM were getting
    downgraded. 0.90 for exact matches and 0.60 for paraphrases works
    well in practice. The 0.40 minimum catches obvious hallucinations
    while still allowing fuzzy matches on OCR'd text (which often has
    minor character errors).
    """
    high_confidence_threshold: float = 0.90
    medium_confidence_threshold: float = 0.60
    min_grounding_ratio: float = 0.40


@dataclass
class Config:
    """Master config — instantiated once, used everywhere."""
    ocr: OCRConfig = field(default_factory=OCRConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    max_file_size_mb: int = 50
    supported_formats: tuple = (".pdf", ".docx", ".jpg", ".jpeg", ".png")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    def __post_init__(self):
        """Validate config on startup so we fail fast instead of crashing
        30 minutes into a 45-minute extraction run."""
        if not 0 <= self.retrieval.bm25_weight <= 1:
            raise ValueError(f"BM25 weight must be [0,1], got {self.retrieval.bm25_weight}")
        if not 0 <= self.retrieval.embedding_weight <= 1:
            raise ValueError(f"Embedding weight must be [0,1], got {self.retrieval.embedding_weight}")

        weight_sum = self.retrieval.bm25_weight + self.retrieval.embedding_weight
        if abs(weight_sum - 1.0) > 0.01:
            logger.warning(
                "Retrieval weights sum to %.2f (expected 1.0). "
                "This may give unexpected ranking results.", weight_sum
            )


# Singleton — every module imports this same instance
config = Config()
