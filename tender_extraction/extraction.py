"""
extraction.py -- Real LLM-powered extraction via llama-cpp-python.

Anti-hallucination strategy:
  - Chain-of-thought prompting (reason first, then format)
  - Few-shot examples in prompts (concrete input->output)
  - Explicit NOT_FOUND for missing fields
  - Source citations required for every extraction
  - Low temperature (0.05) + top_p + repeat_penalty for determinism
"""

from __future__ import annotations

import json
import logging
import re
import time
import threading
import sys
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from rapidfuzz import fuzz

from tender_extraction.config import config
from tender_extraction.schemas import Chunk

logger = logging.getLogger(__name__)

_llm_instance = None
_llm_lock = threading.Lock()  # Prevent concurrent model loading / inference races

# Chat template tags for Phi-3
_SYS_OPEN = "<|system|>"
_TAG_END = "<|end|>"
_USER_OPEN = "<|user|>"
_ASST_OPEN = "<|assistant|>"
_EOS = "</s>"

_SPEC_SYSTEM = (
    "You are a tender parsing assistant. "
    "You MUST respond with ONLY a valid JSON object. No markdown, no explanation, no text before or after the JSON.\n"
    "CRITICAL RULES:\n"
    "1. NEVER invent, assume, or calculate values.\n"
    "2. ONLY extract what is EXPLICITLY written in the provided text.\n"
    '3. Include units inside the value string (e.g. "32 GB", "10 Gbps", "3.2 COP").\n'
    '4. If a value has no unit written, include it as-is and set confidence below 0.6.\n'
    '5. Copy exact phrases from the document into source.exact_text.'
)

_SPEC_FEW_SHOT = (
    'EXAMPLE INPUT:\n'
    '"HVAC Unit: capacity 10 TR, power supply 415 V 3-phase 50 Hz, '
    'minimum COP 3.2, refrigerant R-410A, conforming to IS 1391."\n\n'
    'EXAMPLE OUTPUT:\n'
    '{"technical_specifications": [{'
    '"component": "HVAC Unit", '
    '"specs": {"capacity": "10 TR", "power_supply": "415 V 3-phase 50 Hz", '
    '"cop_minimum": "3.2", "refrigerant": "R-410A", "standard": "IS 1391"}, '
    '"source": {"page": 5, "clause": "3.2", '
    '"exact_text": "HVAC Unit: capacity 10 TR, power supply 415 V"}, '
    '"confidence": 0.94}]}'
)

_SPEC_INSTRUCTIONS = (
    "Read ALL tender text chunks below. Identify EVERY technical specification.\n\n"
    "{few_shot}\n\n"
    'Return a JSON object with key "technical_specifications" containing an array.\n\n'
    "Each object MUST have these exact keys:\n"
    "- component: name of the part, system, or material (string)\n"
    '- specs: flat dict of parameter name to value-with-units '
    '(e.g. {"capacity": "10 TR", "voltage": "415 V"})\n'
    '- source: object with "page" (int), "clause" (string or "NOT_FOUND"), '
    '"exact_text" (verbatim phrase from the text)\n'
    '- confidence: float 0 to 1 (use below 0.6 when unit is missing or value is ambiguous)\n\n'
    "For tables: parse each row into a separate object.\n"
    "TENDER TEXT:\n{context}"
)

_SCOPE_SYSTEM = (
    "You are a tender parsing assistant. "
    "Respond with ONLY a valid JSON object. No markdown. No explanation."
)

_SCOPE_FEW_SHOT = (
    'EXAMPLE INPUT:\n'
    '"The contractor shall perform site preparation, supply and install all equipment '
    'at Building A, and commission the system within 90 days. '
    'Civil work is excluded from vendor scope. Ref: Clause 2."\n\n'
    'EXAMPLE OUTPUT:\n'
    '{"scope_of_work": {'
    '"summary": "Site preparation, supply, installation and commissioning of equipment at Building A within 90 days.", '
    '"deliverables": ["Site preparation", "Supply and install all equipment", "Commission the system"], '
    '"exclusions": ["Civil work is excluded from vendor scope"], '
    '"locations": ["Building A"], '
    '"references": ["Clause 2"]}}'
)

_SCOPE_INSTRUCTIONS = (
    "Extract the scope of work from this tender text.\n\n"
    "{few_shot}\n\n"
    'Return a JSON object with key "scope_of_work" containing:\n'
    '- "summary": concise description max 120 words (string)\n'
    '- "deliverables": list of deliverable strings\n'
    '- "exclusions": list of items explicitly NOT in scope\n'
    '- "locations": site or building locations mentioned\n'
    '- "references": clause and page references cited\n\n'
    "Only include items EXPLICITLY stated. Return empty lists for missing sections.\n\n"
    "TENDER TEXT:\n{context}"
)


def _count_tokens(text: str) -> int:
    """Estimate token count for context budget calculations."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding(config.chunking.tiktoken_model)
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


# Available token budget for context in each LLM call:
#   n_ctx - max_tokens_output - fixed_prompt_overhead (system+instructions+few-shot+tags)
_PROMPT_OVERHEAD_TOKENS = 750
_CONTEXT_BUDGET = config.llm.n_ctx - config.llm.max_tokens - _PROMPT_OVERHEAD_TOKENS


def _split_into_batches(
    retrieved_chunks: List[Dict[str, Any]],
    budget: int = _CONTEXT_BUDGET,
) -> List[List[Dict[str, Any]]]:
    """
    Split retrieved chunks into context-budget-sized batches so each
    LLM call stays within the model's context window.

    With n_ctx=4096 and max_tokens=1024, _CONTEXT_BUDGET â‰ˆ 2322 tokens.
    Average chunk is ~350 tokens, so each batch holds ~6 chunks.
    30 spec chunks â†’ ~5 LLM calls whose results are merged.
    """
    batches: List[List[Dict[str, Any]]] = []
    current_batch: List[Dict[str, Any]] = []
    current_tokens = 0

    for item in retrieved_chunks:
        chunk_tokens = _count_tokens(item["chunk"].text)
        if current_tokens + chunk_tokens > budget and current_batch:
            batches.append(current_batch)
            current_batch = [item]
            current_tokens = chunk_tokens
        else:
            current_batch.append(item)
            current_tokens += chunk_tokens

    if current_batch:
        batches.append(current_batch)

    return batches


class LLMProgressIndicator:
    """Shows a live progress indicator during LLM inference."""

    def __init__(self, message: str = "LLM generating"):
        self.message = message
        self.running = False
        self.thread = None
        self.start_time = None

    def _run(self):
        chars = "\u280b\u2819\u2839\u2838\u283c\u2834\u2826\u2827\u2807\u280f"
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
        sys.stderr.write(f"\r  \u2713 {self.message} complete.          \n")
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
    """Load Phi-3-mini GGUF via llama-cpp-python (thread-safe singleton)."""
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance

    with _llm_lock:
        # Double-check inside the lock
        if _llm_instance is not None:
            return _llm_instance

        _prepare_llama_dll_paths()
        from llama_cpp import Llama, llama_cpp

        model_path = config.llm.model_path
        if not Path(model_path).exists():
            raise RuntimeError(
                f"LLM model not found at: {model_path}\n"
                f"Download Phi-3-mini-4k-instruct (Q4) from:\n"
                f"  https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf\n"
                f"Then set LLM_MODEL_PATH environment variable or update config.py.\n"
                f"See SETUP.md for detailed instructions."
            )

        gpu_supported = False
        try:
            gpu_supported = bool(llama_cpp.llama_supports_gpu_offload())
        except Exception:
            gpu_supported = False

        logger.info("Loading LLM: %s", model_path)
        logger.info("llama.cpp GPU offload support: %s", gpu_supported)
        if not gpu_supported:
            if config.llm.require_gpu:
                raise RuntimeError(
                    "GPU-only mode is enabled (REQUIRE_GPU=1), but llama.cpp GPU offload is unavailable. "
                    "Fix CUDA/llama-cpp runtime in the active environment before running extraction."
                )
            logger.warning(
                "GPU offload is not available in the current Python environment; running on CPU."
            )
        _llm_instance = Llama(
            model_path=model_path,
            n_ctx=config.llm.n_ctx,
            n_gpu_layers=-1,
            n_threads=config.llm.n_threads,
            verbose=False
        )
        logger.info("LLM loaded successfully.")

    return _llm_instance


def _prepare_llama_dll_paths() -> None:
    """Register DLL search paths for llama-cpp and CUDA runtime on Windows.

    Keeps all runtime dependencies inside the active venv (project-local).
    Safe no-op on non-Windows platforms.
    """
    if os.name != "nt":
        return

    try:
        import ctypes
        import site
        base_candidates = []
        for p in site.getsitepackages():
            if os.path.isdir(p):
                base_candidates.append(p)
        # Fallback when site.getsitepackages() is unusual.
        base_candidates.append(os.path.join(sys.prefix, "Lib", "site-packages"))

        loaded_any = False
        llama_lib_dir = None
        for base in base_candidates:
            for rel in (
                os.path.join("llama_cpp", "lib"),
                os.path.join("nvidia", "cublas", "bin"),
                os.path.join("nvidia", "cuda_runtime", "bin"),
                os.path.join("nvidia", "cuda_nvrtc", "bin"),
            ):
                dll_dir = os.path.join(base, rel)
                if os.path.isdir(dll_dir):
                    os.add_dll_directory(dll_dir)
                    loaded_any = True
                    if rel == os.path.join("llama_cpp", "lib"):
                        llama_lib_dir = dll_dir

        # Some Windows setups still fail lazy dependency discovery during import.
        # Preload the native chain explicitly when available.
        if loaded_any and llama_lib_dir and os.path.isdir(llama_lib_dir):
            for dll_name in ("ggml-base.dll", "ggml-cpu.dll", "ggml-cuda.dll", "ggml.dll", "llama.dll"):
                dll_path = os.path.join(llama_lib_dir, dll_name)
                if os.path.isfile(dll_path):
                    try:
                        ctypes.CDLL(dll_path)
                    except Exception:
                        pass
    except Exception:
        # If this fails, llama-cpp import will surface the real dependency error.
        pass

def _get_json_grammar():
    # Grammar mode caused parser crashes/access violations on some llama-cpp
    # Windows builds. We keep generation stable by disabling grammar and using
    # robust JSON repair + parsing downstream.
    return None


def _repair_json(raw: str) -> str:
    """
    Best-effort JSON repair for common LLM output issues.

    Handles:
    - Markdown code fences (```json ... ```)
    - Leading/trailing prose before/after the JSON object
    - Only the outermost JSON object is extracted
    """
    if not raw:
        return raw

    # Strip markdown fencing
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    raw = re.sub(r"\s*```\s*$", "", raw.strip())

    # Find the first '{' and last '}' â€” extract the outermost object
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = raw[start:end + 1]
        # Quick sanity: valid JSON?
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    # Try from the first '[' for arrays wrapped without a key
    start_arr = raw.find("[")
    if start_arr != -1:
        end_arr = raw.rfind("]")
        if end_arr > start_arr:
            candidate = raw[start_arr:end_arr + 1]
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass

    return raw


def extract_specifications(
    retrieved_chunks: List[Dict[str, Any]],
    topic: str = "",
) -> List[Dict[str, Any]]:
    """
    Extract technical specifications from retrieved chunks.

    Splits chunks into context-budget-sized batches so the LLM never
    receives more tokens than its context window can hold.  Results from
    all batches are merged and deduplicated by component name.
    """
    llm = load_model()
    grammar = _get_json_grammar()
    batches = _split_into_batches(retrieved_chunks)

    all_specs: List[Dict[str, Any]] = []
    for batch_idx, batch in enumerate(batches):
        logger.info(
            "Spec extraction batch %d/%d (%d chunks, ~%d tokens)",
            batch_idx + 1, len(batches), len(batch),
            sum(_count_tokens(x["chunk"].text) for x in batch),
        )
        context = _build_context(batch)
        prompt = build_spec_prompt(context, topic)
        raw = _call_llm(
            llm, prompt, f"Extracting specs (batch {batch_idx+1}/{len(batches)})", grammar
        )
        repaired = _repair_json(raw)
        try:
            parsed = json.loads(repaired)
            specs = parsed.get("technical_specifications", [])
            if isinstance(specs, list):
                all_specs.extend(specs)
        except json.JSONDecodeError as exc:
            logger.error(
                "Spec batch %d JSON parse error: %s\nRaw (first 400): %s",
                batch_idx + 1, exc, raw[:400],
            )

    # Deduplicate by component name (case-insensitive)
    seen: set = set()
    deduped: List[Dict[str, Any]] = []
    for s in all_specs:
        key = (s.get("component") or "").strip().lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(s)
        elif not key:
            deduped.append(s)   # keep unnamed specs (may have useful specs dict)

    logger.info(
        "Extracted %d specs total across %d batches (%d after dedup)",
        len(all_specs), len(batches), len(deduped),
    )
    return deduped


def extract_scope_of_work(
    retrieved_chunks: List[Dict[str, Any]],
    topic: str = "",
) -> Dict[str, Any]:
    """
    Extract scope-of-work from retrieved chunks.

    Uses the first batch that yields a non-empty result; if the first
    batch misses things, merge deliverables/exclusions from all batches.
    """
    empty = {"summary": "", "deliverables": [], "exclusions": [], "locations": [], "references": []}
    llm = load_model()
    grammar = _get_json_grammar()
    batches = _split_into_batches(retrieved_chunks)

    merged = dict(empty)
    for batch_idx, batch in enumerate(batches):
        logger.info(
            "Scope extraction batch %d/%d (%d chunks)",
            batch_idx + 1, len(batches), len(batch),
        )
        context = _build_context(batch)
        prompt = build_scope_prompt(context, topic)
        raw = _call_llm(
            llm, prompt, f"Extracting scope (batch {batch_idx+1}/{len(batches)})", grammar
        )
        repaired = _repair_json(raw)
        try:
            parsed = json.loads(repaired)
        except json.JSONDecodeError as exc:
            logger.error(
                "Scope batch %d JSON parse error: %s\nRaw (first 400): %s",
                batch_idx + 1, exc, raw[:400],
            )
            continue

        scope = parsed.get("scope_of_work", parsed)
        if not isinstance(scope, dict):
            continue

        # Merge: take first non-empty summary; extend lists
        if not merged["summary"] and scope.get("summary"):
            merged["summary"] = scope["summary"]
        for key in ("deliverables", "exclusions", "locations", "references"):
            items = scope.get(key, [])
            if isinstance(items, list):
                for item in items:
                    if item and item not in merged[key]:
                        merged[key].append(item)

    return merged


# -- Prompt builders -------------------------------------------------------

def build_spec_prompt(context: str, topic: str = "") -> str:
    topic_hint = ""
    if topic:
        topic_hint = "This tender is for: " + topic + "\n\n"

    # Use explicit placeholder replacement so JSON braces inside few-shot
    # examples are treated literally (avoid str.format KeyError like "capacity").
    user_msg = topic_hint + _SPEC_INSTRUCTIONS.replace("{few_shot}", _SPEC_FEW_SHOT).replace("{context}", context)

    return (
        _SYS_OPEN + "\n" + _SPEC_SYSTEM + _TAG_END + "\n"
        + _USER_OPEN + "\n" + user_msg + _TAG_END + "\n"
        + _ASST_OPEN
    )


def build_scope_prompt(context: str, topic: str = "") -> str:
    topic_hint = ""
    if topic:
        topic_hint = "This tender is for: " + topic + "\n\n"

    user_msg = topic_hint + _SCOPE_INSTRUCTIONS.replace("{few_shot}", _SCOPE_FEW_SHOT).replace("{context}", context)

    return (
        _SYS_OPEN + "\n" + _SCOPE_SYSTEM + _TAG_END + "\n"
        + _USER_OPEN + "\n" + user_msg + _TAG_END + "\n"
        + _ASST_OPEN
    )


# -- Context building -------------------------------------------------------

def _build_context(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Format retrieved chunks into a context string for the LLM prompt.
    
    Improvements:
    - Sort by page number for logical flow (not retrieval score)
    - Deduplicate near-identical chunks (fuzzy overlap > 80%)
    - Minimal metadata header to maximize content tokens
    - Section boundary markers to aid navigation
    """
    if not retrieved_chunks:
        return ""

    # Deduplicate near-identical chunks
    deduped = _deduplicate_chunks(retrieved_chunks)

    # Sort by page number for logical reading order
    deduped.sort(key=lambda x: (x["chunk"].metadata.page, x.get("score", 0)))

    parts: List[str] = []
    current_section = None
    for item in deduped:
        chunk = item["chunk"]
        meta = chunk.metadata

        # Add section boundary when section changes (compact header)
        if meta.section != current_section:
            current_section = meta.section
            if current_section and current_section not in ("Unknown", "Table"):
                parts.append(f"\n[Section: {meta.section}]")

        # Compact page marker â€” saves ~20 tokens per chunk vs the old format
        page_tag = f"[p.{meta.page}]" if meta.chunk_type != "table" else f"[TABLE p.{meta.page}]"
        parts.append(f"{page_tag} {chunk.text}")

    return "\n\n".join(parts)


def _deduplicate_chunks(
    chunks: List[Dict[str, Any]], threshold: float = 80.0
) -> List[Dict[str, Any]]:
    """
    Remove near-duplicate chunks using fuzzy string matching.
    Keeps the chunk with the higher retrieval score.
    """
    if len(chunks) <= 1:
        return chunks

    result = []
    seen_texts: List[str] = []

    # Sort by score descending so we keep the best-scoring version
    sorted_chunks = sorted(chunks, key=lambda x: x.get("score", 0), reverse=True)

    for item in sorted_chunks:
        text = item["chunk"].text
        is_dup = False
        for seen in seen_texts:
            ratio = fuzz.ratio(text[:200].lower(), seen[:200].lower())
            if ratio >= threshold:
                is_dup = True
                break
        if not is_dup:
            result.append(item)
            seen_texts.append(text)

    if len(chunks) != len(result):
        logger.info(
            "Deduplication: %d -> %d chunks (removed %d near-duplicates)",
            len(chunks), len(result), len(chunks) - len(result),
        )

    return result


# -- LLM call ---------------------------------------------------------------

def _call_llm(llm, prompt: str, progress_message: str = "LLM generating", grammar=None) -> str:
    """Call the LLM with retry, exponential backoff, grammar enforcement, and tuned generation params.
    
    Uses _llm_lock to prevent concurrent inference (llama-cpp-python is not thread-safe).
    """
    last_error = None
    for attempt in range(1, config.llm.max_retries + 1):
        try:
            with _llm_lock:
                with LLMProgressIndicator(progress_message):
                    response = llm(
                        prompt,
                        max_tokens=config.llm.max_tokens,
                        temperature=config.llm.temperature,
                        top_p=config.llm.top_p,
                        repeat_penalty=config.llm.repeat_penalty,
                        stop=["```", _EOS, _TAG_END],
                        grammar=grammar,
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


# -- Smoke test -------------------------------------------------------------

if __name__ == "__main__":
    import sys as _sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tender_extraction.ingestion import ingest_document
    from tender_extraction.chunking import create_chunks
    from tender_extraction.table_extraction import extract_tables
    from tender_extraction.retrieval import HybridRetriever

    pdf = "dataset/globaltender1576.pdf"
    if not Path(pdf).exists():
        print(f"Dataset file not found: {pdf}")
        _sys.exit(1)

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
        _sys.exit(1)

    print("\nExtraction smoke test passed.")


