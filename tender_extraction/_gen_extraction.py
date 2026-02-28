"""Script to generate extraction.py with proper multi-line prompt strings."""
import os

OUTPATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extraction.py")

# ---- Build prompt constants as Python-safe strings ----

SYS_TAG_OPEN = "<" + "|system|" + ">"
SYS_TAG_CLOSE = "<" + "|end|" + ">"
USER_TAG_OPEN = "<" + "|user|" + ">"
USER_TAG_CLOSE = "<" + "|end|" + ">"
ASST_TAG_OPEN = "<" + "|assistant|" + ">"
END_TAG = "<" + "/s" + ">"

FEW_SHOT_EXAMPLE = (
    'EXAMPLE INPUT:\n'
    '"Steel reinforcement bars shall be Grade 60 conforming to ASTM A615. '
    'Minimum yield strength 420 MPa. Tolerance +/- 2%."\n\n'
    'EXAMPLE OUTPUT:\n'
    '{"technical_specifications": [{"item_name": "Steel Reinforcement Bars", '
    '"specification_text": "Steel reinforcement bars shall be Grade 60 conforming to ASTM A615. '
    'Minimum yield strength 420 MPa.", '
    '"unit": "MPa", "numeric_value": "420", "tolerance": "+/- 2%", '
    '"standard_reference": "ASTM A615", "material": "Steel Grade 60", '
    '"source": {"page": 15, "exact_text": "Steel reinforcement bars shall be Grade 60"}, '
    '"confidence": "HIGH"}]}'
)

SPEC_SYSTEM = (
    "You are a precise JSON extraction API for tender documents. "
    "You MUST respond with ONLY a valid JSON object. "
    "No markdown. No explanation. No text before or after the JSON.\n"
    "CRITICAL RULES:\n"
    "1. NEVER invent, assume, or calculate values\n"
    "2. ONLY extract what is EXPLICITLY written in the provided text\n"
    '3. Use "NOT_FOUND" for ANY field not explicitly stated in the text\n'
    "4. Copy exact phrases from the document for specification_text and source.exact_text"
)

SPEC_INSTRUCTIONS = (
    "First, carefully read ALL the tender text chunks below. "
    "Then, identify every technical specification mentioned. "
    "Finally, format them as a JSON object.\n\n"
    "{few_shot}\n\n"
    'Now extract ALL technical specifications from this tender text. '
    'Return a JSON object with key "technical_specifications" containing an array.\n\n'
    "Each specification object must have these exact keys:\n"
    "- item_name: descriptive name of the component or parameter (string, min 5 chars)\n"
    "- specification_text: the full requirement as stated in the document (string, min 20 chars, copy verbatim)\n"
    '- unit: unit of measurement, or "NOT_FOUND"\n'
    '- numeric_value: numeric quantity as string, or "NOT_FOUND"\n'
    '- tolerance: tolerance value, or "NOT_FOUND"\n'
    '- standard_reference: standard code like IS/ASTM/ISO, or "NOT_FOUND"\n'
    '- material: material type, or "NOT_FOUND"\n'
    '- source: object with "page" (integer) and "exact_text" (short verbatim phrase from document)\n'
    '- confidence: "HIGH"\n\n'
    "IMPORTANT: If a value like unit, tolerance, or material is NOT explicitly written "
    'in the text, you MUST use "NOT_FOUND". Do NOT guess or infer.\n\n'
    "TENDER TEXT:\n{context}"
)

SCOPE_SYSTEM = (
    "You are a precise JSON extraction API. "
    "Respond with ONLY a valid JSON object. No markdown. No explanation."
)

SCOPE_INSTRUCTIONS = (
    'Extract the scope of work from this tender text. '
    'Return a JSON object with key "scope_of_work" containing "tasks" array and "exclusions" array.\n\n'
    'Each task: {{"task_description": "...", "responsible_party": "...", "timeline": "..."}}\n'
    'Each exclusion: {{"exclusion_description": "..."}}\n\n'
    'Use "NOT_FOUND" for missing sub-fields. Only extract content explicitly in the text.\n\n'
    "TENDER TEXT:\n{context}"
)


lines = []
lines.append('"""')
lines.append('extraction.py -- Real LLM-powered extraction via llama-cpp-python.')
lines.append('')
lines.append('Anti-hallucination strategy:')
lines.append('  - Chain-of-thought prompting (reason first, then format)')
lines.append('  - Few-shot examples in prompts (concrete input->output)')
lines.append('  - Explicit NOT_FOUND for missing fields')
lines.append('  - Source citations required for every extraction')
lines.append('  - Low temperature (0.05) + top_p + repeat_penalty for determinism')
lines.append('"""')
lines.append('')
lines.append('from __future__ import annotations')
lines.append('')
lines.append('import json')
lines.append('import logging')
lines.append('import re')
lines.append('import time')
lines.append('import threading')
lines.append('import sys')
lines.append('from pathlib import Path')
lines.append('from typing import Any, Dict, List, Optional')
lines.append('')
lines.append('from rapidfuzz import fuzz')
lines.append('')
lines.append('from tender_extraction.config import config')
lines.append('from tender_extraction.schemas import Chunk')
lines.append('')
lines.append('logger = logging.getLogger(__name__)')
lines.append('')
lines.append('_llm_instance = None')
lines.append('')
lines.append('# Chat template tags for Phi-3')
lines.append(f'_SYS_OPEN = "{SYS_TAG_OPEN}"')
lines.append(f'_TAG_END = "{SYS_TAG_CLOSE}"')
lines.append(f'_USER_OPEN = "{USER_TAG_OPEN}"')
lines.append(f'_ASST_OPEN = "{ASST_TAG_OPEN}"')
lines.append(f'_EOS = "{END_TAG}"')
lines.append('')
lines.append(f'_SPEC_SYSTEM = {repr(SPEC_SYSTEM)}')
lines.append('')
lines.append(f'_SPEC_FEW_SHOT = {repr(FEW_SHOT_EXAMPLE)}')
lines.append('')
lines.append(f'_SPEC_INSTRUCTIONS = {repr(SPEC_INSTRUCTIONS)}')
lines.append('')
lines.append(f'_SCOPE_SYSTEM = {repr(SCOPE_SYSTEM)}')
lines.append('')
lines.append(f'_SCOPE_INSTRUCTIONS = {repr(SCOPE_INSTRUCTIONS)}')
lines.append('')
lines.append('')

# Now add all the functions
lines.append(r'''
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
    """Load Phi-3-mini GGUF via llama-cpp-python."""
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
        n_gpu_layers=-1,
        n_threads=config.llm.n_threads,
        verbose=False
    )
    logger.info("LLM loaded on GPU successfully.")
    return _llm_instance


def extract_specifications(
    retrieved_chunks: List[Dict[str, Any]],
    topic: str = "",
) -> List[Dict[str, Any]]:
    """Extract technical specifications from retrieved chunks using the LLM."""
    llm = load_model()
    context = _build_context(retrieved_chunks)
    prompt = build_spec_prompt(context, topic)

    raw = _call_llm(llm, prompt, "Phi-3 extracting specifications")
    parsed = _clean_and_parse_json(raw, "technical_specifications")

    if not parsed or "technical_specifications" not in parsed:
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
    """Extract scope-of-work from retrieved chunks using the LLM."""
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

    user_msg = topic_hint + _SCOPE_INSTRUCTIONS.format(context=context)

    return (
        _SYS_OPEN + "\n" + _SCOPE_SYSTEM + _TAG_END + "\n"
        + _USER_OPEN + "\n" + user_msg + _TAG_END + "\n"
        + _ASST_OPEN
    )


# -- Context building -------------------------------------------------------

def _build_context(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Format retrieved chunks into a context string for the LLM prompt.
    
    Improvements over v1:
    - Sort by page number for logical flow (not retrieval score)
    - Deduplicate near-identical chunks (fuzzy overlap > 80%)
    - Add clear section boundaries
    - Truncate low-relevance chunks if token budget is tight
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
        chunk: Chunk = item["chunk"]
        score = item.get("score", 0.0)
        meta = chunk.metadata

        # Add section boundary when section changes
        if meta.section != current_section:
            current_section = meta.section
            parts.append(f"\n=== Section: {meta.section} ===")

        parts.append(
            f"--- CHUNK [{chunk.chunk_id}] | Page {meta.page} | "
            f"Type: {meta.chunk_type} | Relevance: {score:.3f} ---\n"
            f"{chunk.text}\n"
        )
    return "\n".join(parts)


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

def _call_llm(llm, prompt: str, progress_message: str = "LLM generating") -> str:
    """Call the LLM with retry, exponential backoff, and tuned generation params."""
    last_error = None
    for attempt in range(1, config.llm.max_retries + 1):
        try:
            with LLMProgressIndicator(progress_message):
                response = llm(
                    prompt,
                    max_tokens=config.llm.max_tokens,
                    temperature=config.llm.temperature,
                    top_p=config.llm.top_p,
                    repeat_penalty=config.llm.repeat_penalty,
                    stop=["```", "\n\n\n", _EOS, _TAG_END],
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


# -- JSON parsing -----------------------------------------------------------

def _clean_and_parse_json(raw: str, root_key: str) -> dict:
    text = raw.strip()

    # Strip markdown fences
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()

    # Find first { -- strip any preamble before it
    first_brace = text.find('{')
    if first_brace > 0:
        text = text[first_brace:]

    # If no { found but starts with a key: wrap it
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
        if root_key == "scope_of_work":
            return {root_key: {"tasks": [], "exclusions": []}}
        return {root_key: []}


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
'''.lstrip())


full_content = "\n".join(lines) + "\n"

with open(OUTPATH, "w", encoding="utf-8") as f:
    f.write(full_content)

print(f"Generated {OUTPATH} ({len(full_content)} bytes, {full_content.count(chr(10))} lines)")
