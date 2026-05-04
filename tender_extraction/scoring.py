"""
scoring.py -- Scores and ranks a tender against a Company Profile using LLM.
"""

import json
import logging
from typing import Dict, Any

from tender_extraction.extraction import load_model, _call_llm, _repair_json

logger = logging.getLogger(__name__)

_SCORING_SYSTEM = (
    "You are a professional business development analyst evaluating tender requirements against a company profile. "
    "Respond ONLY with a valid JSON object. No markdown, no pre-amble, no post-amble."
)

_SCORING_INSTRUCTIONS = (
    "Evaluate how well the company profile matches the given tender requirements (scope and specs).\n\n"
    "COMPANY PROFILE:\n{profile}\n\n"
    "TENDER SUMMARY:\n{tender_summary}\n\n"
    "Based on the company capabilities, exclusions, and preferred locations, evaluate the match.\n"
    "Provide a JSON object with:\n"
    "- 'match_score': An integer from 0 to 100 representing the overall alignment.\n"
    "- 'cost_feasibility': 'High', 'Medium', or 'Low' (guess based on budget limits vs scope size).\n"
    "- 'reasoning': A concise paragraph explaining the score and any red flags.\n"
)

def score_tender_match(company_profile: Dict[str, Any], tender_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Score the tender against the company profile.
    """
    llm = load_model()
    
    # Compress tender_result into a readable summary to fit in context
    scope = tender_result.get("scope_of_work", {})
    summary = scope.get("summary", "")

    deliverables_list = [
        d for d in scope.get("deliverables", [])
        if isinstance(d, str) and d.strip()
    ]
    deliverables = ", ".join(deliverables_list[:5])

    specs = tender_result.get("technical_specifications", [])
    components_list = [
        s.get("component", "").strip()
        for s in specs
        if isinstance(s, dict) and isinstance(s.get("component"), str) and s.get("component").strip()
    ]
    components = ", ".join(components_list[:10])
    
    
    tender_summary = f"Scope Summary: {summary}\nDeliverables: {deliverables}\nKey Components: {components}"
    profile_text = json.dumps(company_profile, indent=2)
    
    prompt_user = _SCORING_INSTRUCTIONS.replace("{profile}", profile_text).replace("{tender_summary}", tender_summary)
    
    prompt = f"<|system|>\n{_SCORING_SYSTEM}<|end|>\n<|user|>\n{prompt_user}<|end|>\n<|assistant|>"
    
    raw = _call_llm(llm, prompt, "Scoring Match")
    repaired = _repair_json(raw)
    
    try:
        parsed = json.loads(repaired)
        return {
            "match_score": parsed.get("match_score", 0),
            "cost_feasibility": parsed.get("cost_feasibility", "Unknown"),
            "reasoning": parsed.get("reasoning", "Could not parse reasoning from LLM output."),
        }
    except Exception as e:
        logger.warning(f"Failed to parse scoring output: {e}\nRaw: {raw}")
        return {
            "match_score": 0,
            "cost_feasibility": "Unknown",
            "reasoning": "Failed to evaluate due to LLM parsing error."
        }
