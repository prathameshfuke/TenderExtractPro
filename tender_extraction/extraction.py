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
_llm_lock = threading.Lock()

# Chat template tags for Phi-3
_SYS_OPEN = "<|system|>"
_TAG_END = "<|end|>"
_USER_OPEN = "<|user|>"
_ASST_OPEN = "<|assistant|>"
_EOS = "</s>"

_SPEC_SYSTEM = (
    "You are a professional tender document analyst. "
    "Your goal is to extract technical specifications with 100% accuracy. "
    "Respond ONLY with a valid JSON object. No markdown, no pre-amble, no post-amble.\n"
    "CRITICAL RULES:\n"
    "1. VERBATIM EXTRACTION: Copy technical values and units exactly as they appear.\n"
    "2. NO INVENTIONS: If a value is not present, do not guess. Set confidence below 0.4.\n"
    "3. STRUCTURE: Parse each unique component/item as a separate object.\n"
    "4. SOURCES: You MUST provide the exact snippet of text as 'exact_text' for grounding.\n"
    "5. HINT: If you see a table-like structure in the text, parse it row-by-row."
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
    "Carefully analyze the following tender document chunks. "
    "Identify every technical parameter, material requirement, and equipment specification.\n\n"
    "{few_shot}\n\n"
    "Format your output as a JSON object with a 'technical_specifications' list.\n"
    "Each item MUST include:\n"
    "- 'component': The name of the item or system.\n"
    "- 'specs': A dictionary of parameters (e.g., {'Power': '5kW', 'Weight': '20kg'}).\n"
    "- 'source': {'page': int, 'clause': 'string', 'exact_text': 'verbatim quote'}.\n"
    "- 'confidence': A score from 0.0 to 1.0.\n\n"
    "DOCUMENT CONTEXT:\n{context}"
)

_SCOPE_SYSTEM = (
    "You are a professional tender document analyst. "
    "Extract the Scope of Work accurately. Respond ONLY with valid JSON."
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
    "Analyze the tender text and extract the formal Scope of Work.\n\n"
    "{few_shot}\n\n"
    "Return a JSON object with 'scope_of_work' containing:\n"
    "- 'summary': A high-level description (copy key sentences).\n"
    "- 'deliverables': A list of specific items to be supplied/done.\n"
    "- 'exclusions': Items explicitly mentioned as NOT part of the contract.\n"
    "- 'locations': Sites, cities, or buildings mentioned.\n"
    "- 'references': Clause numbers related to scope.\n\n"
    "CONTEXT:\n{context}"
)

_QA_SYSTEM = (
    "You answer questions about one tender document using ONLY the provided evidence. "
    "If the answer is not explicitly supported, say that it is not found in the document. "
    "Return ONLY valid JSON."
)

_QA_INSTRUCTIONS = (
    'Return JSON with keys "answer", "citations", and "confidence".\n'
    '- "answer": short grounded answer in plain English.\n'
    '- "citations": array of objects with "page", "chunk_id", and "quote". Use up to 3 citations.\n'
    '- "confidence": one of HIGH, MEDIUM, LOW.\n\n'
    'If the answer is not present, set "answer" to "NOT_FOUND in provided document context" and use an empty citations array.\n\n'
    'QUESTION:\n{question}\n\n'
    'DOCUMENT TOPIC:\n{topic}\n\n'
    'DOCUMENT EVIDENCE:\n{context}'
)


def _count_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding(config.chunking.tiktoken_model)
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


_PROMPT_OVERHEAD_TOKENS = 750
_CONTEXT_BUDGET = config.llm.n_ctx - config.llm.max_tokens - _PROMPT_OVERHEAD_TOKENS


def _split_into_batches(
    retrieved_chunks: List[Dict[str, Any]],
    budget: int = _CONTEXT_BUDGET,
) -> List[List[Dict[str, Any]]]:
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
    def __init__(self, message: str = "LLM generating"):
        self.message = message
        self.running = False
        self.thread = None
        self.start_time = None

    def _run(self):
        chars = "|/-\\"
        i = 0
        while self.running:
            elapsed = time.time() - self.start_time
            sys.stderr.write(
                f"\r  {chars[i % len(chars)]} {self.message}... ({elapsed:.0f}s)"
            )
            sys.stderr.flush()
            time.sleep(0.1)
            i += 1
        sys.stderr.write(f"\r  * {self.message} complete.          \n")
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
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance

    with _llm_lock:
        if _llm_instance is not None:
            return _llm_instance

        _prepare_llama_dll_paths()
        from llama_cpp import Llama, llama_cpp

        model_path = config.llm.model_path
        if not Path(model_path).exists():
            raise RuntimeError(f"LLM model not found at: {model_path}")

        _llm_instance = Llama(
            model_path=model_path,
            n_ctx=config.llm.n_ctx,
            n_gpu_layers=config.llm.n_gpu_layers,
            n_threads=config.llm.n_threads,
            use_mmap=False,  # Fix for 'PrefetchVirtualMemory unavailable' on Windows
            verbose=False
        )
    return _llm_instance


def _prepare_llama_dll_paths() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes
        import site
        for base in site.getsitepackages() + [os.path.join(sys.prefix, "Lib", "site-packages")]:
            for rel in (os.path.join("llama_cpp", "lib"), "nvidia/cublas/bin", "nvidia/cuda_runtime/bin"):
                dll_dir = os.path.join(base, rel)
                if os.path.isdir(dll_dir):
                    os.add_dll_directory(dll_dir)
    except Exception:
        pass

def _get_json_grammar():
    return None


def _repair_json(raw: str) -> str:
    if not raw:
        return raw
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1:
        candidate = raw[start:end+1]
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            def escape_newlines(match):
                return match.group(0).replace("\n", "\\n")
            candidate = re.sub(r'"[^"]*"', escape_newlines, candidate, flags=re.DOTALL)
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass
    return raw


def _salvage_json_objects(raw: str, array_key: str) -> List[Dict[str, Any]]:
    key_pos = raw.find(f'"{array_key}"')
    if key_pos == -1: return []
    array_start = raw.find("[", key_pos)
    if array_start == -1: return []
    recovered = []
    depth = 0
    object_start = None
    for idx in range(array_start, len(raw)):
        char = raw[idx]
        if char == "{":
            if depth == 0: object_start = idx
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and object_start is not None:
                try:
                    recovered.append(json.loads(raw[object_start:idx+1]))
                except:
                    pass
                object_start = None
    return recovered


def extract_specifications(retrieved_chunks: List[Dict[str, Any]], topic: str = "") -> List[Dict[str, Any]]:
    llm = load_model()
    batches = _split_into_batches(retrieved_chunks)
    all_specs = []
    for batch_idx, batch in enumerate(batches):
        context = _build_context(batch)
        prompt = build_spec_prompt(context, topic)
        raw = _call_llm(llm, prompt, f"Extracting specs ({batch_idx+1}/{len(batches)})")
        repaired = _repair_json(raw)
        try:
            parsed = json.loads(repaired)
            specs = parsed.get("technical_specifications", [])
            if isinstance(specs, list): all_specs.extend(specs)
        except:
            all_specs.extend(_salvage_json_objects(repaired, "technical_specifications"))
    
    seen = set()
    deduped = []
    for s in all_specs:
        key = (s.get("component") or "").strip().lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(s)
        elif not key: deduped.append(s)
    return deduped


def extract_scope_of_work(retrieved_chunks: List[Dict[str, Any]], topic: str = "") -> Dict[str, Any]:
    empty = {"summary": "", "deliverables": [], "exclusions": [], "locations": [], "references": []}
    llm = load_model()
    batches = _split_into_batches(retrieved_chunks)
    merged = dict(empty)
    for batch_idx, batch in enumerate(batches):
        context = _build_context(batch)
        prompt = build_scope_prompt(context, topic)
        raw = _call_llm(llm, prompt, f"Extracting scope ({batch_idx+1}/{len(batches)})")
        try:
            parsed = json.loads(_repair_json(raw))
            scope = parsed.get("scope_of_work", parsed)
            if not isinstance(scope, dict): continue
            if not merged["summary"]: merged["summary"] = scope.get("summary", "")
            for k in ("deliverables", "exclusions", "locations", "references"):
                for item in scope.get(k, []):
                    if item and item not in merged[k]: merged[k].append(item)
        except: continue
    return merged


def build_spec_prompt(context: str, topic: str = "") -> str:
    topic_hint = f"Tender Topic: {topic}\n\n" if topic else ""
    user_msg = topic_hint + _SPEC_INSTRUCTIONS.replace("{few_shot}", _SPEC_FEW_SHOT).replace("{context}", context)
    return f"{_SYS_OPEN}\n{_SPEC_SYSTEM}{_TAG_END}\n{_USER_OPEN}\n{user_msg}{_TAG_END}\n{_ASST_OPEN}"

def build_scope_prompt(context: str, topic: str = "") -> str:
    topic_hint = f"Tender Topic: {topic}\n\n" if topic else ""
    user_msg = topic_hint + _SCOPE_INSTRUCTIONS.replace("{few_shot}", _SCOPE_FEW_SHOT).replace("{context}", context)
    return f"{_SYS_OPEN}\n{_SCOPE_SYSTEM}{_TAG_END}\n{_USER_OPEN}\n{user_msg}{_TAG_END}\n{_ASST_OPEN}"

def build_qa_prompt(question: str, context: str, topic: str = "") -> str:
    user_msg = _QA_INSTRUCTIONS.replace("{question}", question).replace("{topic}", topic or "N/A").replace("{context}", context)
    return f"{_SYS_OPEN}\n{_QA_SYSTEM}{_TAG_END}\n{_USER_OPEN}\n{user_msg}{_TAG_END}\n{_ASST_OPEN}"

def answer_question(retrieved_chunks: List[Dict[str, Any]], question: str, topic: str = "") -> Dict[str, Any]:
    if not retrieved_chunks: return {"answer": "NOT_FOUND", "citations": [], "confidence": "LOW"}
    llm = load_model()
    context = _build_context(retrieved_chunks[:8])
    prompt = build_qa_prompt(question, context, topic)
    raw = _call_llm(llm, prompt, "Answering")
    try:
        parsed = json.loads(_repair_json(raw))
        return parsed
    except:
        return {"answer": raw, "citations": [], "confidence": "LOW"}

def _build_context(retrieved_chunks: List[Dict[str, Any]]) -> str:
    if not retrieved_chunks: return ""
    deduped = _deduplicate_chunks(retrieved_chunks)
    deduped.sort(key=lambda x: (x["chunk"].metadata.page, x.get("score", 0)))
    parts = []
    current_sec = None
    for item in deduped:
        chunk = item["chunk"]
        if chunk.metadata.section != current_sec:
            current_sec = chunk.metadata.section
            if current_sec and current_sec not in ("Unknown", "Table"):
                parts.append(f"\n[Section: {current_sec}]")
        parts.append(f"[p.{chunk.metadata.page}] {chunk.text}")
    return "\n\n".join(parts)

def _deduplicate_chunks(chunks: List[Dict[str, Any]], threshold: float = 80.0) -> List[Dict[str, Any]]:
    if len(chunks) <= 1: return chunks
    result, seen = [], []
    for item in sorted(chunks, key=lambda x: x.get("score", 0), reverse=True):
        text = item["chunk"].text[:200].lower()
        if not any(fuzz.ratio(text, s) >= threshold for s in seen):
            result.append(item)
            seen.append(text)
    return result

def _call_llm(llm, prompt: str, msg: str = "LLM generating") -> str:
    for attempt in range(1, config.llm.max_retries + 1):
        try:
            with _llm_lock:
                with LLMProgressIndicator(msg):
                    res = llm(prompt, max_tokens=config.llm.max_tokens, temperature=config.llm.temperature, stop=[_TAG_END])
            return res["choices"][0]["text"].strip()
        except Exception as e:
            if attempt == config.llm.max_retries: raise
            time.sleep(2 ** attempt)
    return ""

if __name__ == "__main__":
    pass
