from __future__ import annotations

import contextlib
import io
import logging
from typing import List, Optional

from sentence_transformers import SentenceTransformer

from tender_extraction.config import config

logger = logging.getLogger(__name__)

_embed_model: Optional[SentenceTransformer] = None
_embed_model_device: Optional[str] = None


def resolve_torch_device(prefer_gpu: bool = True) -> str:
    """Return the best available torch device for the current runtime."""
    try:
        import torch
    except Exception:
        return "cpu"

    if prefer_gpu and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_embedding_device() -> str:
    device = resolve_torch_device(prefer_gpu=True)
    if config.retrieval.require_gpu and device != "cuda":
        logger.warning(
            "CUDA is not available for embeddings; falling back to CPU. "
            "Install a CUDA-enabled PyTorch build to use GPU retrieval."
        )
    return device


def get_embedding_model() -> SentenceTransformer:
    global _embed_model, _embed_model_device
    if _embed_model is None:
        model_name = config.retrieval.embedding_model
        device = resolve_embedding_device()
        logger.info("Loading embedding model: %s on %s", model_name, device)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            _embed_model = SentenceTransformer(model_name, device=device)
        _embed_model_device = device
    return _embed_model


def get_embedding_device() -> str:
    if _embed_model_device is not None:
        return _embed_model_device
    return resolve_embedding_device()


def get_embedding_dimension(model: Optional[SentenceTransformer] = None) -> int:
    embed_model = model or get_embedding_model()
    if hasattr(embed_model, "get_embedding_dimension"):
        return int(embed_model.get_embedding_dimension())
    return int(embed_model.get_sentence_embedding_dimension())


class SentenceTransformerEmbeddings:
    """
    Small adapter so SemanticChunker can reuse the same SentenceTransformer
    instance as retrieval instead of loading a second copy.
    """

    def __init__(self, model: Optional[SentenceTransformer] = None):
        self._model = model or get_embedding_model()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = self._model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=16,
            normalize_embeddings=True,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:
        vector = self._model.encode(
            [text],
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=1,
            normalize_embeddings=True,
        )[0]
        return vector.tolist()
