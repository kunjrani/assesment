"""Extract reusable operator preferences from edit diffs (no API required)."""

from __future__ import annotations

import logging
from typing import Any

from legal_rag import config as cfg

logger = logging.getLogger(__name__)


def extract_patterns_rule_based(edit_record: dict[str, Any]) -> dict[str, Any]:
    """
    Deterministic pattern extraction from diff — works without OpenAI.
    """
    diff = edit_record.get("diff", {})
    patterns: list[str] = []
    rules: list[str] = []

    before = diff.get("summary_before", "")
    after = diff.get("summary_after", "")

    if diff.get("summary_changed") and after:
        if before and after.lower().startswith(("plaintiff", "petitioner", "defendant")):
            patterns.append("Lead the summary with party names before procedural details.")
            rules.append("summary_lead_parties")
        elif any(tok in after[:60].lower() for tok in ("case no", "case number", "docket")):
            patterns.append("Open the summary with the case number for quick scanning.")
            rules.append("summary_lead_case_number")
        if len(after) > len(before) * 1.2:
            patterns.append("Prefer a longer, more explicit summary with concrete dates and parties.")
            rules.append("summary_more_detail")

    for change in diff.get("checklist_changes", []):
        if change.get("change") == "status" and change.get("to") == "present":
            patterns.append(
                f"Mark checklist item '{change.get('item')}' as present when supported by evidence."
            )
            rules.append(f"checklist_{change.get('item', 'item')}_present")

    if diff.get("claims_removed"):
        patterns.append("Remove claims that are not clearly supported by cited evidence.")
        rules.append("drop_weak_claims")

    if diff.get("claims_added"):
        patterns.append("Add explicit claims with source_ids for each key fact.")
        rules.append("add_explicit_claims")

    if not patterns:
        patterns.append("Match the operator's edited summary style and checklist statuses exactly.")

    intent = " ".join(patterns)
    return {
        "intent": intent,
        "patterns": patterns,
        "rules": rules,
        "extractor": "rule_based",
    }


def extract_patterns_llm(edit_record: dict[str, Any]) -> dict[str, Any]:
    """Optional LLM extraction when API key is available."""
    if not cfg.OPENAI_API_KEY:
        return extract_patterns_rule_based(edit_record)

    from openai import OpenAI

    client = OpenAI(api_key=cfg.OPENAI_API_KEY)
    diff = edit_record.get("diff", {})
    prompt = f"""Analyze this operator edit on a legal draft and return JSON:
{{"intent": "one paragraph", "patterns": ["rule 1", "rule 2"], "rules": ["short_tag"]}}

Edit diff:
{diff}
"""
    try:
        resp = client.chat.completions.create(
            model=cfg.LLM_MODEL,
            messages=[
                {"role": "system", "content": "Return JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        import json

        data = json.loads(resp.choices[0].message.content or "{}")
        data["extractor"] = "openai"
        return data
    except Exception as exc:
        logger.warning("LLM pattern extraction failed (%s); using rules", exc)
        return extract_patterns_rule_based(edit_record)


def extract_patterns(edit_record: dict[str, Any], use_llm: bool = False) -> dict[str, Any]:
    if use_llm and cfg.OPENAI_API_KEY:
        return extract_patterns_llm(edit_record)
    return extract_patterns_rule_based(edit_record)
