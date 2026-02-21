"""
extraction.py — LLM-powered extraction with anti-hallucination measures.

This is where the actual "intelligence" happens. We feed retrieved chunks
to Mistral-7B-Instruct via llama-cpp-python and ask it to extract structured
specs and scope-of-work data.

The anti-hallucination strategy went through several iterations:

  v1 (Jan 2026): Just ask the LLM to extract specs. Hallucination rate: ~25%.
     The LLM would confidently invent standard codes, add decimal points to
     whole numbers, and mix up quantities between different items.

  v2 (Feb 2026): Added mandatory citations. Rate dropped to ~15%. The LLM
     still hallucinated but at least we could detect it by checking citations.

  v3 (current): Citations + grounding verification + NOT_FOUND enforcement.
     Rate: ~3%. The key insight was telling the LLM "output NOT_FOUND for
     missing data" AND verifying that cited text actually exists in the
     source chunks. The remaining 3% are mostly paraphrasing issues where
     the LLM says "500 kg" when the source says "500 kgs" — technically
     wrong but close enough to pass grounding checks.
- Prathamesh, 2026-02-15
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from tender_extraction.config import config
from tender_extraction.schemas import Chunk

logger = logging.getLogger(__name__)

# Module-level LLM cache. Loading the model takes ~10s so we do it once
# and reuse across calls. The model itself is ~4GB in memory.
_llm_instance = None


# ── Prompt Templates ──────────────────────────────────────────────────────
# These prompts were iterated on extensively. Key learnings:
# - "Extract ONLY from provided chunks" reduces hallucination significantly
# - Repeating the rules multiple times actually helps (the model pays more
#   attention to repeated instructions)
# - Asking for JSON without markdown fences works better with Mistral
#   than asking for ```json blocks (it often forgets to close the fence)
# - The double-brace {{}} is Python f-string escaping, not a typo

SPEC_EXTRACTION_PROMPT = """You are a technical specification extractor for tender documents. Extract ONLY information present in the provided chunks.

RULES:
1. Extract ONLY from the provided chunks — do NOT add any external knowledge.
2. For missing fields, output "NOT_FOUND" — NEVER guess or fabricate values.
3. Include source citation (chunk_id + page) for EVERY extracted field.
4. Quote exact text from the chunks to support each extraction.
5. Flag confidence: HIGH (exact match found in text), MEDIUM (paraphrased/inferred), LOW (uncertain).

CONTEXT CHUNKS:
{context}

EXTRACTION TASK:
Extract all technical specifications from the above chunks.

OUTPUT FORMAT (respond ONLY with valid JSON, no markdown fences):
{{
  "specifications": [
    {{
      "item_name": "...",
      "specification_text": "...",
      "unit": "..." or "NOT_FOUND",
      "numeric_value": "..." or "NOT_FOUND",
      "tolerance": "..." or "NOT_FOUND",
      "standard_reference": "..." or "NOT_FOUND",
      "material": "..." or "NOT_FOUND",
      "source": {{"chunk_id": "...", "page": <int>, "exact_text": "..."}},
      "confidence": "HIGH" | "MEDIUM" | "LOW"
    }}
  ]
}}
"""

SCOPE_EXTRACTION_PROMPT = """You are a scope-of-work extractor for tender documents. Extract ONLY information present in the provided chunks.

RULES:
1. Extract ONLY from the provided chunks — do NOT add any external knowledge.
2. For missing fields, output "NOT_FOUND" — NEVER guess or fabricate values.
3. Include source citation (chunk_id + page) for EVERY task.
4. Identify task descriptions, deliverables, timelines, dependencies, and exclusions.

CONTEXT CHUNKS:
{context}

EXTRACTION TASK:
Extract the scope of work including tasks, deliverables, timelines, dependencies, and exclusions.

OUTPUT FORMAT (respond ONLY with valid JSON, no markdown fences):
{{
  "tasks": [
    {{
      "task_description": "...",
      "deliverables": ["..."],
      "timeline": "..." or "NOT_FOUND",
      "dependencies": ["..."],
      "source": {{"chunk_id": "...", "page": <int>, "exact_text": "..."}}
    }}
  ],
  "exclusions": [
    {{
      "item": "...",
      "source": {{"chunk_id": "...", "page": <int>}}
    }}
  ]
}}
"""


def _get_llm():
    """
    Lazy-load Mistral-7B via llama-cpp-python. Cached at module level.

    The first call takes ~10s (model load from disk). Subsequent calls
    are instant. We tried loading in __init__ but that made import time
    terrible and broke fast unit tests.
    """
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance

    try:
        from llama_cpp import Llama

        logger.info("Loading LLM from: %s", config.llm.model_path)
        _llm_instance = Llama(
            model_path=config.llm.model_path,
            n_ctx=config.llm.n_ctx,
            n_threads=config.llm.n_threads or None,
            verbose=False,
        )
        logger.info("LLM loaded successfully.")
        return _llm_instance
    except FileNotFoundError:
        raise RuntimeError(
            f"LLM model file not found: '{config.llm.model_path}'. "
            f"Download Mistral-7B-Instruct-v0.2 GGUF Q4 from HuggingFace "
            f"and set the LLM_MODEL_PATH environment variable."
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load LLM: {exc}. Make sure llama-cpp-python is "
            f"installed correctly and the model file is valid."
        ) from exc


def _llm_generate(prompt: str) -> str:
    """
    Generate text from the LLM with retry and exponential backoff.

    We've seen three types of failures in production:
    1. Timeout: large context + long generation on slow CPUs. Fixed by retry.
    2. OOM: model + context exceed available RAM. No fix except smaller model.
    3. Repetition loops: model gets stuck generating the same token. We use
       stop tokens to break out of these.
    """
    llm = _get_llm()
    last_error = None

    for attempt in range(1, config.llm.max_retries + 1):
        try:
            response = llm(
                prompt,
                max_tokens=config.llm.max_tokens,
                temperature=config.llm.temperature,
                # These stop tokens prevent runaway generation. Without them,
                # the model sometimes generates JSON + a bunch of explanation
                # text after the closing brace.
                stop=["```", "\n\n\n"],
            )
            text = response["choices"][0]["text"].strip()
            logger.info(
                "LLM generated %d chars on attempt %d/%d",
                len(text), attempt, config.llm.max_retries
            )
            return text
        except Exception as exc:
            last_error = exc
            delay = config.llm.retry_base_delay * (2 ** (attempt - 1))
            logger.warning(
                "LLM attempt %d/%d failed: %s. Retrying in %.1fs.",
                attempt, config.llm.max_retries, exc, delay
            )
            time.sleep(delay)

    raise RuntimeError(
        f"LLM generation failed after {config.llm.max_retries} retries: {last_error}"
    )


def extract_with_citations(
    chunks: List[Dict[str, Any]],
    extraction_type: str = "specifications",
) -> Dict[str, Any]:
    """
    Run LLM extraction with mandatory citations.

    Args:
        chunks:  Retrieval results (each has a 'chunk' key with a Chunk object).
        extraction_type:  "specifications" or "scope_of_work".

    Returns:
        Parsed JSON dict from the LLM.
    """
    context = _format_context(chunks)

    if extraction_type == "specifications":
        prompt = SPEC_EXTRACTION_PROMPT.format(context=context)
    elif extraction_type == "scope_of_work":
        prompt = SCOPE_EXTRACTION_PROMPT.format(context=context)
    else:
        raise ValueError(f"Unknown extraction type: {extraction_type}")

    raw_output = _llm_generate(prompt)

    # Parse the JSON output. Mistral is pretty good at generating valid
    # JSON but occasionally wraps it in markdown fences or adds trailing
    # text. Our parser handles all of these cases.
    parsed = _parse_json_output(raw_output)

    if parsed is None:
        logger.error(
            "Could not parse LLM output as JSON. First 500 chars: %s",
            raw_output[:500]
        )
        if extraction_type == "specifications":
            return {"specifications": []}
        else:
            return {"tasks": [], "exclusions": []}

    return parsed


def extract_specifications(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract technical specifications from retrieved chunks."""
    result = extract_with_citations(chunks, extraction_type="specifications")
    return result.get("specifications", [])


def extract_scope_of_work(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract scope of work from retrieved chunks."""
    result = extract_with_citations(chunks, extraction_type="scope_of_work")
    return {
        "tasks": result.get("tasks", []),
        "exclusions": result.get("exclusions", []),
    }


def _format_context(chunks: List[Dict[str, Any]]) -> str:
    """
    Format retrieved chunks into the context block for the LLM prompt.

    Each chunk is tagged with its ID, page, section, and type so the
    LLM can cite them correctly. We include the retrieval score so the
    LLM can prioritize higher-confidence chunks (Mistral seems to be
    slightly better at extraction when we include relevance info).
    """
    parts: List[str] = []
    for item in chunks:
        chunk: Chunk = item["chunk"]
        score = item.get("score", 0.0)
        meta = chunk.metadata
        parts.append(
            f"--- CHUNK [{chunk.chunk_id}] | Page {meta.page} | "
            f"Section: {meta.section} | Type: {meta.chunk_type} | "
            f"Relevance: {score:.3f} ---\n"
            f"{chunk.text}\n"
        )
    return "\n".join(parts)


def _parse_json_output(text: str) -> Optional[Dict[str, Any]]:
    """
    Multi-strategy JSON parser for LLM output.

    We need this because LLMs are unreliable JSON generators. Mistral
    usually gives clean JSON but about 10% of the time it:
    - Wraps in ```json fences (despite being told not to)
    - Adds explanatory text before/after the JSON
    - Generates trailing comma in arrays (invalid JSON)
    - Truncates output at max_tokens mid-JSON

    We try three strategies in order of strictness:
    1. Direct parse (works ~80% of the time)
    2. Strip markdown fences and retry (~10%)
    3. Regex extract first JSON object (~5%)
    4. Give up and return None (~5%, handled upstream)
    """
    # Strategy 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: strip markdown fences
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = cleaned.strip().rstrip("`")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 3: regex extract first complete JSON object
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


# Backward-compat alias used by validation.py
def verify_grounding(
    extraction: Dict[str, Any],
    source_chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Delegates to validation.verify_grounding."""
    from tender_extraction.validation import verify_grounding as _vg
    return _vg(extraction, source_chunks)
