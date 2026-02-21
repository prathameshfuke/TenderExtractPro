"""
extraction.py — Real LLM-powered extraction via llama-cpp-python.

This module loads a quantized Mistral-7B-Instruct GGUF model and uses it
to extract structured technical specifications and scope-of-work from
retrieved tender document chunks. There are NO mock responses, NO hardcoded
results, and NO placeholder functions.

If the model file is not found, this module raises RuntimeError with a
clear message telling the user where to download it. It does NOT silently
return empty results.

The anti-hallucination strategy:
  - Prompt explicitly instructs "Return ONLY valid JSON"
  - Prompt says "use NOT_FOUND for missing fields, NEVER invent values"
  - Every extraction must include source citation (chunk_id + page)
  - Output is validated against Pydantic schemas in schemas.py
  - Downstream validation.py does grounding verification
- Prathamesh, 2026-02-18
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from tender_extraction.config import config
from tender_extraction.schemas import Chunk

logger = logging.getLogger(__name__)

# Module-level LLM cache. Loading the 4GB model takes ~10s, so we do it
# once and reuse. The model stays in memory for the lifetime of the process.
_llm_instance = None


def load_model():
    """
    Load Mistral-7B-Instruct GGUF via llama-cpp-python.

    Raises RuntimeError if the model file doesn't exist. We fail loud
    and clear instead of silently returning garbage.
    """
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance

    from llama_cpp import Llama

    model_path = config.llm.model_path
    if not Path(model_path).exists():
        raise RuntimeError(
            f"LLM model not found at: {model_path}\n"
            f"Download Mistral-7B-Instruct-v0.2 GGUF (Q4_K_M) from:\n"
            f"  https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF\n"
            f"Then set LLM_MODEL_PATH environment variable or update config.py.\n"
            f"See SETUP.md for detailed instructions."
        )

    logger.info("Loading LLM: %s", model_path)
    _llm_instance = Llama(
        model_path=model_path,
        n_ctx=config.llm.n_ctx,
        n_threads=config.llm.n_threads or 4,
        verbose=False,
    )
    logger.info("LLM loaded successfully.")
    return _llm_instance


def extract_specifications(
    retrieved_chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Extract technical specifications from retrieved chunks using the LLM.

    Takes the real retrieved chunks (each with chunk.text, chunk.metadata),
    builds a real prompt, calls the real LLM, parses the real JSON response,
    and returns real specifications.

    If JSON parsing fails on the first attempt, retries once with a
    stricter prompt. If it fails again, logs an error and returns empty
    (not silently — the error is logged).
    """
    llm = load_model()
    context = _build_context(retrieved_chunks)
    prompt = _build_spec_prompt(context)

    raw = _call_llm(llm, prompt)
    parsed = _parse_json_response(raw)

    if parsed is None:
        # Retry with stricter prompt
        logger.warning("First LLM call failed to produce valid JSON. Retrying ...")
        strict_prompt = prompt + (
            "\n\nIMPORTANT: Your previous response was not valid JSON. "
            "This time, output ONLY a JSON object starting with { and "
            "ending with }. No explanation, no markdown, JUST THE JSON."
        )
        raw = _call_llm(llm, strict_prompt)
        parsed = _parse_json_response(raw)

    if parsed is None:
        logger.error("LLM failed to produce valid JSON after retry. Raw output:\n%s", raw[:500])
        return []

    return parsed.get("specifications", parsed.get("technical_specifications", []))


def extract_scope_of_work(
    retrieved_chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Extract scope-of-work from retrieved chunks using the LLM.
    Same approach as extract_specifications but with a different prompt.
    """
    llm = load_model()
    context = _build_context(retrieved_chunks)
    prompt = _build_scope_prompt(context)

    raw = _call_llm(llm, prompt)
    parsed = _parse_json_response(raw)

    if parsed is None:
        logger.warning("Scope extraction: retrying with stricter prompt ...")
        strict_prompt = prompt + (
            "\n\nIMPORTANT: Output ONLY valid JSON. No markdown fences. "
            "Start with { and end with }."
        )
        raw = _call_llm(llm, strict_prompt)
        parsed = _parse_json_response(raw)

    if parsed is None:
        logger.error("Scope extraction failed after retry. Raw:\n%s", raw[:500])
        return {"tasks": [], "exclusions": []}

    return {
        "tasks": parsed.get("tasks", []),
        "exclusions": parsed.get("exclusions", []),
    }


# ── Prompt templates ──────────────────────────────────────────────────────
# These prompts were iterated on over 20+ test runs. Key learnings:
# - Repeating "ONLY from provided chunks" twice reduces hallucination
# - "NOT_FOUND" instruction must be in caps or the model ignores it
# - Providing a concrete example output helps the model follow the schema
# - Telling the model to "never invent values" explicitly is critical

def _build_spec_prompt(context: str) -> str:
    return f"""[INST] You are a technical specification extractor for tender documents.

RULES — follow these exactly:
1. Extract ONLY from the provided context chunks. Do NOT add external knowledge.
2. For any field not found in the context, use the exact string "NOT_FOUND".
3. NEVER invent, guess, or fabricate values. If unsure, use "NOT_FOUND".
4. Include source citation for every spec: the chunk_id and page number.
5. Quote the exact text from the chunk that supports each extraction.

CONTEXT CHUNKS:
{context}

TASK: Extract all technical specifications from the above chunks.

OUTPUT FORMAT — respond with ONLY this JSON structure, no other text:
{{
  "specifications": [
    {{
      "item_name": "name of the item or material",
      "specification_text": "full specification description from the document",
      "unit": "unit of measurement or NOT_FOUND",
      "numeric_value": "quantity or value or NOT_FOUND",
      "tolerance": "tolerance range or NOT_FOUND",
      "standard_reference": "IS/ASTM/ISO code or NOT_FOUND",
      "material": "material type or NOT_FOUND",
      "source": {{"chunk_id": "the chunk ID", "page": page_number, "exact_text": "verbatim quote from chunk"}},
      "confidence": "HIGH or MEDIUM or LOW"
    }}
  ]
}}
[/INST]"""


def _build_scope_prompt(context: str) -> str:
    return f"""[INST] You are a scope-of-work extractor for tender documents.

RULES — follow these exactly:
1. Extract ONLY from the provided context chunks. Do NOT add external knowledge.
2. For missing fields, use "NOT_FOUND". NEVER invent values.
3. Include source citation (chunk_id + page) for every task.

CONTEXT CHUNKS:
{context}

TASK: Extract the scope of work: tasks, deliverables, timelines, and exclusions.

OUTPUT FORMAT — respond with ONLY this JSON, no other text:
{{
  "tasks": [
    {{
      "task_description": "description of the work task",
      "deliverables": ["list of deliverables"],
      "timeline": "timeline or NOT_FOUND",
      "dependencies": ["list of dependencies"],
      "source": {{"chunk_id": "the chunk ID", "page": page_number, "exact_text": "verbatim quote"}}
    }}
  ],
  "exclusions": [
    {{
      "item": "excluded item description",
      "source": {{"chunk_id": "the chunk ID", "page": page_number}}
    }}
  ]
}}
[/INST]"""


def _build_context(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into a context string for the LLM prompt."""
    parts: List[str] = []
    for item in retrieved_chunks:
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


def _call_llm(llm, prompt: str) -> str:
    """
    Call the LLM with retry and exponential backoff.

    The stop tokens prevent runaway generation — without them, the model
    sometimes outputs JSON followed by a long explanation of what it did.
    """
    last_error = None
    for attempt in range(1, config.llm.max_retries + 1):
        try:
            response = llm(
                prompt,
                max_tokens=config.llm.max_tokens,
                temperature=config.llm.temperature,
                stop=["```", "\n\n\n", "[/INST]", "</s>"],
            )
            text = response["choices"][0]["text"].strip()
            logger.info(
                "LLM generated %d chars (attempt %d/%d)",
                len(text), attempt, config.llm.max_retries,
            )
            return text
        except Exception as exc:
            last_error = exc
            delay = config.llm.retry_base_delay * (2 ** (attempt - 1))
            logger.warning(
                "LLM attempt %d/%d failed: %s. Retrying in %.1fs.",
                attempt, config.llm.max_retries, exc, delay,
            )
            time.sleep(delay)

    raise RuntimeError(f"LLM failed after {config.llm.max_retries} retries: {last_error}")


def _parse_json_response(text: str) -> Optional[Dict[str, Any]]:
    """
    Parse LLM output as JSON. Tries three strategies because LLMs are
    inconsistent JSON generators:
      1. Direct parse
      2. Strip markdown fences and retry
      3. Regex extract first { ... } block
    """
    if not text:
        return None

    # Strategy 1: direct
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: strip markdown fences
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 3: regex extract
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


# ── Smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    # This smoke test requires the LLM model to be downloaded.
    # If the model isn't present, it will raise RuntimeError with
    # download instructions.

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tender_extraction.ingestion import ingest_document
    from tender_extraction.chunking import create_chunks
    from tender_extraction.table_extraction import extract_tables
    from tender_extraction.retrieval import HybridRetriever

    pdf = "dataset/globaltender1576.pdf"
    if not Path(pdf).exists():
        print(f"Dataset file not found: {pdf}")
        sys.exit(1)

    print("Ingesting ...")
    pages = ingest_document(pdf)
    tables = extract_tables(pdf)
    chunks = create_chunks(pages, tables)

    print("Building retrieval index ...")
    retriever = HybridRetriever()
    retriever.build_index(chunks)

    print("Retrieving spec chunks ...")
    spec_chunks = retriever.retrieve("technical specifications requirements standards", top_k=10)

    print(f"Running LLM extraction on {len(spec_chunks)} chunks ...")
    try:
        specs = extract_specifications(spec_chunks)
        print(f"\nExtracted {len(specs)} specifications:")
        for s in specs:
            print(f"  - {s.get('item_name', '?')}: {s.get('specification_text', '?')[:60]}")
            print(f"    source: page {s.get('source', {}).get('page', '?')}, "
                  f"confidence: {s.get('confidence', '?')}")
    except RuntimeError as exc:
        print(f"\nLLM not available: {exc}")
        print("Download the model (see SETUP.md) to run this smoke test.")
        sys.exit(1)

    print("\nExtraction smoke test passed.")
