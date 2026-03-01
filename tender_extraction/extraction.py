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

_SPEC_SYSTEM = 'You are a precise JSON extraction API for tender documents. You MUST respond with ONLY a valid JSON object. No markdown. No explanation. No text before or after the JSON.\nCRITICAL RULES:\n1. NEVER invent, assume, or calculate values\n2. ONLY extract what is EXPLICITLY written in the provided text\n3. Use "NOT_FOUND" for ANY field not explicitly stated in the text\n4. Copy exact phrases from the document for specification_text and source.exact_text'

_SPEC_FEW_SHOT = 'EXAMPLE INPUT:\n"Steel reinforcement bars shall be Grade 60 conforming to ASTM A615. Minimum yield strength 420 MPa. Tolerance +/- 2%."\n\nEXAMPLE OUTPUT:\n{"technical_specifications": [{"item_name": "Steel Reinforcement Bars", "specification_text": "Steel reinforcement bars shall be Grade 60 conforming to ASTM A615. Minimum yield strength 420 MPa.", "unit": "MPa", "numeric_value": "420", "tolerance": "+/- 2%", "standard_reference": "ASTM A615", "material": "Steel Grade 60", "source": {"page": 15, "exact_text": "Steel reinforcement bars shall be Grade 60"}, "confidence": "HIGH"}]}'

_SPEC_INSTRUCTIONS = 'First, carefully read ALL the tender text chunks below. Then, identify every technical specification mentioned. Finally, format them as a JSON object.\n\n{few_shot}\n\nNow extract ALL technical specifications from this tender text. Return a JSON object with key "technical_specifications" containing an array.\n\nEach specification object must have these exact keys:\n- item_name: descriptive name of the component or parameter (string, min 5 chars)\n- specification_text: the full requirement as stated in the document (string, min 20 chars, copy verbatim)\n- unit: unit of measurement, or "NOT_FOUND"\n- numeric_value: numeric quantity as string, or "NOT_FOUND"\n- tolerance: tolerance value, or "NOT_FOUND"\n- standard_reference: standard code like IS/ASTM/ISO, or "NOT_FOUND"\n- material: material type, or "NOT_FOUND"\n- source: object with "page" (integer) and "exact_text" (short verbatim phrase from document)\n- confidence: "HIGH"\n\nIMPORTANT: If a value like unit, tolerance, or material is NOT explicitly written in the text, you MUST use "NOT_FOUND". Do NOT guess or infer.\n\nTENDER TEXT:\n{context}'

_SCOPE_SYSTEM = 'You are a precise JSON extraction API. Respond with ONLY a valid JSON object. No markdown. No explanation.'

_SCOPE_FEW_SHOT = 'EXAMPLE INPUT:\n"The contractor shall perform site preparation, supply and install all equipment, and commission the system within 90 days. Civil work is excluded from vendor scope."\n\nEXAMPLE OUTPUT:\n{"scope_of_work": {"tasks": [{"task_description": "Site preparation", "responsible_party": "Contractor", "timeline": "NOT_FOUND"}, {"task_description": "Supply and install all equipment", "responsible_party": "Contractor", "timeline": "NOT_FOUND"}, {"task_description": "Commission the system", "responsible_party": "Contractor", "timeline": "90 days"}], "exclusions": [{"exclusion_description": "Civil work is excluded from vendor scope"}]}}'

_SCOPE_INSTRUCTIONS = 'Extract the scope of work from this tender text.\n\n{few_shot}\n\nReturn a JSON object with key "scope_of_work" containing:\n- "tasks": array of task objects with keys: task_description, responsible_party (or "NOT_FOUND"), timeline (or "NOT_FOUND")\n- "exclusions": array of objects with key: exclusion_description\n\nOnly include items EXPLICITLY stated. Use "NOT_FOUND" for missing sub-fields.\n\nTENDER TEXT:\n{context}'


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
            n_gpu_layers=-1,
            n_threads=config.llm.n_threads,
            verbose=False
        )
        logger.info("LLM loaded successfully.")

    return _llm_instance

def _get_json_grammar():
    try:
        from llama_cpp import LlamaGrammar
        # Strict JSON grammar to enforce valid output from the LLM.
        # Uses GBNF (GGML BNF) syntax — no inline # comments allowed.
        # The {4} quantifier requires llama.cpp >= b1700; if it fails we
        # fall back gracefully by returning None (no grammar constraint).
        prompt_grammar = r"""
            root   ::= object
            value  ::= object | array | string | number | "true" | "false" | "null"
            object ::=
            "{" ws "}"
            | "{" ws string ws ":" ws value ws ("," ws string ws ":" ws value ws)* "}"
            array  ::=
            "[" ws "]"
            | "[" ws value ws ("," ws value ws)* "]"
            string ::=
            "\"" (
                [^"\\\x7F\x00-\x1F] |
                "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
            )* "\""
            number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?
            ws     ::= ([ \t\n] ws)?
        """
        return LlamaGrammar.from_string(prompt_grammar)
    except Exception as exc:
        logger.warning("Failed to load JSON grammar: %s. Proceeding without grammar.", exc)
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

    # Find the first '{' and last '}' — extract the outermost object
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
    """Extract technical specifications from retrieved chunks using the LLM with Grammar enforcement."""
    llm = load_model()
    context = _build_context(retrieved_chunks)
    prompt = build_spec_prompt(context, topic)

    grammar = _get_json_grammar()
    raw = _call_llm(llm, prompt, "Phi-3 extracting specifications", grammar)
    repaired = _repair_json(raw)
    
    try:
        parsed = json.loads(repaired)
        specs = parsed.get("technical_specifications", [])
        if not isinstance(specs, list):
            logger.warning("Unexpected specs type: %s", type(specs))
            return []
        logger.info("LLM extracted %d specifications.", len(specs))
        return specs
    except json.JSONDecodeError as exc:
        logger.error("LLM failed to produce valid JSON specs: %s\nRaw (first 500): %s",
                     exc, raw[:500])
        return []


def extract_scope_of_work(
    retrieved_chunks: List[Dict[str, Any]],
    topic: str = "",
) -> Dict[str, Any]:
    """Extract scope-of-work from retrieved chunks using the LLM with Grammar enforcement."""
    llm = load_model()
    context = _build_context(retrieved_chunks)
    prompt = build_scope_prompt(context, topic)
    
    grammar = _get_json_grammar()
    raw = _call_llm(llm, prompt, "Phi-3 extracting scope of work", grammar)
    repaired = _repair_json(raw)
    
    try:
        parsed = json.loads(repaired)
    except json.JSONDecodeError as exc:
        logger.error("Scope extraction failed to produce valid JSON: %s\nRaw (first 500): %s",
                     exc, raw[:500])
        return {"tasks": [], "exclusions": []}

    if isinstance(parsed, list):
        parsed = {"scope_of_work": {"tasks": parsed, "exclusions": []}}

    scope = parsed.get("scope_of_work", parsed)

    if isinstance(scope, list):
        scope = {"tasks": scope, "exclusions": []}

    if not isinstance(scope, dict):
        scope = {"tasks": [], "exclusions": []}

    return {
        "tasks": scope.get("tasks", []),
        "exclusions": scope.get("exclusions", [])
    }


# -- Prompt builders -------------------------------------------------------

def build_spec_prompt(context: str, topic: str = "") -> str:
    topic_hint = ""
    if topic:
        topic_hint = "This tender is for: " + topic + "\n\n"

    user_msg = topic_hint + _SPEC_INSTRUCTIONS.format(
        few_shot=_SPEC_FEW_SHOT, context=context
    )

    return (
        _SYS_OPEN + "\n" + _SPEC_SYSTEM + _TAG_END + "\n"
        + _USER_OPEN + "\n" + user_msg + _TAG_END + "\n"
        + _ASST_OPEN
    )


def build_scope_prompt(context: str, topic: str = "") -> str:
    topic_hint = ""
    if topic:
        topic_hint = "This tender is for: " + topic + "\n\n"

    user_msg = topic_hint + _SCOPE_INSTRUCTIONS.format(
        few_shot=_SCOPE_FEW_SHOT, context=context
    )

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

        # Compact page marker — saves ~20 tokens per chunk vs the old format
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
                        stop=["```", "\n\n\n", _EOS, _TAG_END],
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

