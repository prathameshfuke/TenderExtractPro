from __future__ import annotations

import json
import logging
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from tender_extraction.config import config
from tender_extraction.model_runtime import resolve_torch_device

logger = logging.getLogger(__name__)

_vision_model = None
_vision_processor = None
_vision_device = "cpu"

_VISION_SYSTEM_PROMPT = (
    "You extract tables from tender document pages. Return ONLY valid JSON with a top-level "
    "'tables' array. Each table object must contain 'headers' and 'rows'. "
    "Preserve cell order, units, and bilingual text exactly as shown. "
    "If no real table is visible, return {\"tables\": []}."
)

_VISION_USER_PROMPT = (
    "Analyze this tender page image and extract every visible table as JSON. "
    "Use short header strings and row arrays of strings. Ignore running prose that is not tabular."
)


def should_try_multimodal_table_fallback(
    page_number: int,
    conventional_tables_count: int,
    page_hint: Optional[Dict[str, Any]] = None,
) -> bool:
    if not config.multimodal.enabled:
        return False

    if page_hint and page_hint.get("is_ocr"):
        if not config.multimodal.retry_page_without_tables:
            return False
        return True

    if conventional_tables_count > 0:
        return False

    if not config.multimodal.retry_page_without_tables:
        return False

    text = (page_hint or {}).get("text", "") or ""
    return _looks_table_like_text(text)


def extract_tables_from_pdf_page(pdf_path: str, page_number: int) -> List[Dict[str, Any]]:
    image = render_pdf_page(pdf_path, page_number)
    if image is None:
        return []
    return extract_tables_from_image(image, page_number=page_number)


def extract_tables_from_image_file(image_path: str) -> List[Dict[str, Any]]:
    if not config.multimodal.enabled:
        return []
    with Image.open(image_path) as image:
        return extract_tables_from_image(image.convert("RGB"), page_number=1)


def extract_tables_from_image(image: Image.Image, page_number: int) -> List[Dict[str, Any]]:
    if not config.multimodal.enabled:
        return []

    model, processor = _load_multimodal_model()
    if model is None or processor is None:
        return []

    try:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": _VISION_SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": _VISION_USER_PROMPT},
                ],
            },
        ]
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(
            text=[prompt],
            images=[image.convert("RGB")],
            padding=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(_vision_device) if hasattr(value, "to") else value for key, value in inputs.items()}
        generated_ids = model.generate(**inputs, max_new_tokens=config.multimodal.max_new_tokens)
        prompt_length = inputs["input_ids"].shape[1]
        trimmed = generated_ids[:, prompt_length:]
        response_text = processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
    except Exception as exc:
        logger.warning("Multimodal table extraction failed on page %d: %s", page_number, exc)
        return []

    parsed = _parse_multimodal_json(response_text)
    return normalize_multimodal_tables(parsed, page_number=page_number)


def normalize_multimodal_tables(payload: Any, page_number: int) -> List[Dict[str, Any]]:
    tables_payload: List[Any]
    if isinstance(payload, dict):
        tables_payload = payload.get("tables", [])
    elif isinstance(payload, list):
        tables_payload = payload
    else:
        return []

    normalized_tables: List[Dict[str, Any]] = []
    for table_idx, table in enumerate(tables_payload, start=1):
        if isinstance(table, dict):
            headers = [_clean_cell(cell) for cell in table.get("headers", [])]
            rows = [_normalize_row(row) for row in table.get("rows", [])]
        else:
            continue

        rows = [row for row in rows if row and any(cell for cell in row)]
        if not rows:
            continue

        if not headers and rows and all(isinstance(row, list) for row in rows):
            headers = [f"Column {idx + 1}" for idx in range(max(len(row) for row in rows))]

        normalized_tables.append(
            {
                "table_id": f"mm_table_p{page_number:03d}_{table_idx:02d}",
                "page": page_number,
                "headers": headers,
                "rows": rows,
                "bbox": None,
                "raw": table,
            }
        )
    return normalized_tables


def render_pdf_page(pdf_path: str, page_number: int) -> Optional[Image.Image]:
    try:
        import fitz

        with fitz.open(pdf_path) as document:
            page = document[page_number - 1]
            matrix = fitz.Matrix(config.multimodal.render_scale, config.multimodal.render_scale)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            image = Image.open(BytesIO(pixmap.tobytes("png"))).convert("RGB")
            return image
    except Exception as exc:
        logger.warning("Failed to render page %d for multimodal fallback: %s", page_number, exc)
        return None


def release_multimodal_model() -> None:
    global _vision_model, _vision_processor, _vision_device
    _vision_model = None
    _vision_processor = None
    _vision_device = "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _load_multimodal_model():
    global _vision_model, _vision_processor, _vision_device
    if _vision_model is not None and _vision_processor is not None:
        return _vision_model, _vision_processor

    source = config.multimodal.model_path or config.multimodal.model_name
    if config.multimodal.model_path and not Path(config.multimodal.model_path).exists():
        logger.warning("Configured multimodal model path does not exist: %s", config.multimodal.model_path)
        return None, None

    try:
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    except Exception as exc:
        logger.warning("Multimodal dependencies are unavailable: %s", exc)
        return None, None

    requested_device = resolve_torch_device(prefer_gpu=config.multimodal.use_gpu)
    model_dtype = torch.float16 if requested_device == "cuda" else torch.float32

    try:
        logger.info("Loading multimodal model %s on %s", source, requested_device)
        processor = AutoProcessor.from_pretrained(
            source,
            local_files_only=config.multimodal.local_files_only,
        )
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            source,
            local_files_only=config.multimodal.local_files_only,
            torch_dtype=model_dtype,
            low_cpu_mem_usage=True,
        )
        model.eval()
        try:
            model.to(requested_device)
            actual_device = requested_device
        except RuntimeError as exc:
            if requested_device == "cuda":
                logger.warning(
                    "Multimodal model did not fit on GPU (%s). Falling back to CPU for table fallback.",
                    exc,
                )
                model.to("cpu")
                actual_device = "cpu"
            else:
                raise
    except Exception as exc:
        logger.warning("Unable to load multimodal table extractor %s: %s", source, exc)
        return None, None

    _vision_model = model
    _vision_processor = processor
    _vision_device = actual_device
    return _vision_model, _vision_processor


def _parse_multimodal_json(raw_text: str) -> Any:
    candidate = raw_text.strip()
    candidate = re.sub(r"```(?:json)?", "", candidate).strip()
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = candidate[start : end + 1]
    try:
        return json.loads(candidate)
    except Exception:
        return {"tables": []}


def _normalize_row(row: Any) -> List[str]:
    if isinstance(row, dict):
        return [_clean_cell(value) for value in row.values()]
    if isinstance(row, (list, tuple)):
        return [_clean_cell(value) for value in row]
    return [_clean_cell(row)]


def _clean_cell(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _looks_table_like_text(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 80:
        return False
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if not lines:
        return False

    pipe_like = sum(1 for line in lines if line.count("|") >= 2)
    numeric_dense = sum(1 for line in lines if len(re.findall(r"\d", line)) >= 3)
    short_rows = sum(1 for line in lines if len(line.split()) <= 10)
    return pipe_like >= 2 or (numeric_dense >= 3 and short_rows >= 3)
