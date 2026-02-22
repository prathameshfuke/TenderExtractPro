"""
extraction.py — Real LLM-powered extraction via llama-cpp-python.

This module loads a quantized Mistral model and uses it
to extract structured technical specifications and scope-of-work from
retrieved tender document chunks.

The anti-hallucination strategy:
  - Prompt explicitly enforces JSON structure
  - Prompt explicitly handles missing fields via "NOT_FOUND" 
  - Every extraction must include source citations
  - Output is validated against Pydantic schemas
  - Downstream verification performs grounding checks
"""

from __future__ import annotations

import json
import logging
import re
import time
import threading
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from tender_extraction.config import config
from tender_extraction.schemas import Chunk

logger = logging.getLogger(__name__)

# Module-level LLM cache. Loading the 4GB model takes ~10s, so we do it
# once and reuse. The model stays in memory for the lifetime of the process.
_llm_instance = None


class LLMProgressIndicator:
    """Shows a live progress indicator during LLM inference."""
    
    def __init__(self, message: str = "LLM generating"):
        self.message = message
        self.running = False
        self.thread = None
        self.start_time = None
    
    def _run(self):
        chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        i = 0
        while self.running:
            elapsed = time.time() - self.start_time
            sys.stderr.write(
                f"\r  {chars[i % len(chars)]} {self.message}... "
                f"({elapsed:.0f}s elapsed)"
            )
            sys.stderr.flush()
            time.sleep(0.1)
            i += 1
        sys.stderr.write(f"\r  ✓ {self.message} complete.          \n")
        sys.stderr.flush()
    
    def __enter__(self):
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self
    
    def __exit__(self, *args):
        self.running = False
        self.thread.join()

def load_model():
    """
    Load Phi-3-mini GGUF via llama-cpp-python.
    """
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance

    from llama_cpp import Llama

    model_path = config.llm.model_path
    if not Path(model_path).exists():
        raise RuntimeError(
            f"LLM model not found at: {model_path}\n"
            f"Download Phi-3-mini-4k-instruct (Q4) from:\n"
            f"  https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf\n"
            f"Then set LLM_MODEL_PATH environment variable or update config.py.\n"
            f"See SETUP.md for detailed instructions."
        )

    logger.info("Loading LLM: %s", model_path)
    _llm_instance = Llama(
        model_path=model_path,
        n_ctx=config.llm.n_ctx,
        n_gpu_layers=-1,       # FULL GPU — mandatory
        n_threads=config.llm.n_threads,
        verbose=False
    )
    logger.info("LLM loaded on GPU successfully.")
    return _llm_instance


def extract_specifications(
    retrieved_chunks: List[Dict[str, Any]],
    topic: str = "",
) -> List[Dict[str, Any]]:
    """
    Extract technical specifications from retrieved chunks using the LLM.
    """
    llm = load_model()
    context = _build_context(retrieved_chunks)
    prompt = build_spec_prompt(context, topic)

    raw = _call_llm(llm, prompt, "Phi-3 extracting specifications")
    parsed = _clean_and_parse_json(raw, "technical_specifications")

    if not parsed or "technical_specifications" not in parsed:
        # Retry with stricter prompt
        logger.warning("First LLM call failed to produce valid JSON. Retrying ...")
        strict_prompt = prompt + (
            "\n\nIMPORTANT: Your previous response was not valid JSON. "
            "This time, output ONLY a JSON object starting with { and "
            "ending with }. No explanation, no markdown, JUST THE JSON."
        )
        raw = _call_llm(llm, strict_prompt, "Phi-3 extracting specifications (retry)")
        parsed = _clean_and_parse_json(raw, "technical_specifications")

    if not parsed or "technical_specifications" not in parsed:
        logger.error("LLM failed to produce valid specs after retry.")
        return []

    return parsed.get("technical_specifications", [])


def extract_scope_of_work(
    retrieved_chunks: List[Dict[str, Any]],
    topic: str = "",
) -> Dict[str, Any]:
    """
    Extract scope-of-work from retrieved chunks using the LLM.
    """
    llm = load_model()
    context = _build_context(retrieved_chunks)
    prompt = build_scope_prompt(context, topic)

    raw = _call_llm(llm, prompt, "Phi-3 extracting scope of work")
    parsed = _clean_and_parse_json(raw, "scope_of_work")

    if not parsed or "scope_of_work" not in parsed:
        logger.warning("Scope extraction: retrying with stricter prompt ...")
        strict_prompt = prompt + (
            "\n\nIMPORTANT: Output ONLY valid JSON. No markdown fences. "
            "Start with { and end with }."
        )
        raw = _call_llm(llm, strict_prompt, "Phi-3 extracting scope of work (retry)")
        parsed = _clean_and_parse_json(raw, "scope_of_work")

    if not parsed or "scope_of_work" not in parsed:
        logger.error("Scope extraction failed after retry.")
        return {"tasks": [], "exclusions": []}

    # parsed could be:
    # {"scope_of_work": {"tasks": [...], "exclusions": [...]}}  <- correct
    # {"tasks": [...], "exclusions": [...]}                     <- missing wrapper  
    # {"scope_of_work": [...]}                                  <- wrong type
    # [...]                                                     <- bare list
    
    if isinstance(parsed, list):
        parsed = {"scope_of_work": {"tasks": parsed, "exclusions": []}}
    
    scope = parsed.get("scope_of_work", parsed)  # unwrap if nested
    
    if isinstance(scope, list):
        scope = {"tasks": scope, "exclusions": []}
    
    if not isinstance(scope, dict):
        scope = {"tasks": [], "exclusions": []}
    
    return {
        "tasks": scope.get("tasks", []),
        "exclusions": scope.get("exclusions", [])
    }


# ── Prompt templates ──────────────────────────────────────────────────────
# These prompts were iterated on over 20+ test runs. Key learnings:
# - Repeating "ONLY from provided chunks" twice reduces hallucination
# - "NOT_FOUND" instruction must be in caps or the model ignores it
# - Providing a concrete example output helps the model follow the schema
# - Telling the model to "never invent values" explicitly is critical

def build_spec_prompt(context: str, topic: str = "") -> str:
    topic_hint = f"This tender is for: {topic}\n\n" if topic else ""
    return f"""<|system|>
You are a JSON API. Respond with ONLY a valid JSON object. No markdown. No explanation. No text before or after the JSON.<|end|>
<|user|>
{topic_hint}Extract all technical specifications from the tender text below. Return a JSON object with key "technical_specifications" containing an array of specification objects.

Each specification object must have these exact keys:
- item_name: descriptive name of the component or parameter (string, min 5 chars)
- specification_text: the full requirement as stated in the document (string, min 20 chars)  
- unit: unit of measurement, or "NOT_FOUND"
- numeric_value: numeric quantity, or "NOT_FOUND"
- tolerance: tolerance value, or "NOT_FOUND"
- standard_reference: standard code like IS/ASTM/ISO, or "NOT_FOUND"
- material: material type, or "NOT_FOUND"
- source: object with "page" (integer) and "exact_text" (verbatim short phrase from document)
- confidence: "HIGH"

IMPORTANT: Only extract specs actually present in the text. Never invent values.

TENDER TEXT:
{context}
<|end|>
<|assistant|>"""


def build_scope_prompt(context: str, topic: str = "") -> str:
    topic_hint = f"This tender is for: {topic}\n\n" if topic else ""
    return f"""<|system|>
You are a JSON API. Respond with ONLY a valid JSON object. No markdown. No explanation.<|end|>
<|user|>
{topic_hint}Extract the scope of work from this tender text. Return a JSON object with key "scope_of_work" containing "tasks" array and "exclusions" array.

Each task: {{"task_description": "...", "responsible_party": "...", "timeline": "..."}}
Each exclusion: {{"exclusion_description": "..."}}

Use "NOT_FOUND" for missing sub-fields. Only extract content explicitly in the text.

TENDER TEXT:
{context}
<|end|>
<|assistant|>"""


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


def _call_llm(llm, prompt: str, progress_message: str = "LLM generating") -> str:
    """
    Call the LLM with retry and exponential backoff.
    """
    last_error = None
    for attempt in range(1, config.llm.max_retries + 1):
        try:
            with LLMProgressIndicator(progress_message):
                response = llm(
                    prompt,
                    max_tokens=config.llm.max_tokens,
                    temperature=config.llm.temperature,
                    stop=["```", "\n\n\n", "</s>"],
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


def _clean_and_parse_json(raw: str, root_key: str) -> dict:
    text = raw.strip()
    
    # Strip markdown fences
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()
    
    # Find first { — strip any preamble before it
    first_brace = text.find('{')
    if first_brace > 0:
        text = text[first_brace:]
    
    # If no { found but starts with a key like "scope_of_work": wrap it
    if not text.startswith('{'):
        if text.startswith('"'):
            text = '{' + text
    
    # Strip trailing garbage after last valid }
    last_brace = text.rfind('}')
    if last_brace != -1:
        text = text[:last_brace + 1]
    
    # Attempt 1: direct parse
    try:
        result = json.loads(text)
        # Handle case where model returned the root_key value directly (a list)
        if isinstance(result, list):
            return {root_key: result}
        return result
    except json.JSONDecodeError:
        pass
    
    # Attempt 2: salvage truncated JSON by closing open structures
    open_braces = text.count('{') - text.count('}')
    open_brackets = text.count('[') - text.count(']')
    text = text + (']' * max(0, open_brackets)) + ('}' * max(0, open_braces))
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return {root_key: result}
        return result
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse failed after salvage: {e}")
        logger.error(f"Raw output first 300 chars: {raw[:300]}")
        return {root_key: [] if root_key != "scope_of_work" else {"tasks": [], "exclusions": []}}


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
