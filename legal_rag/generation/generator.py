"""LLM draft generation with OpenAI or deterministic mock (no API key)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from legal_rag import config as cfg
from legal_rag.ingestion.models import ProcessedDocument
from legal_rag.retrieval.models import EvidenceChunk

from legal_rag.generation.grounding_check import verify_draft
from legal_rag.generation.prompt_builder import build_prompt_for_document

logger = logging.getLogger(__name__)


class GenerationError(Exception):
    pass


def _parse_json_response(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise GenerationError(f"LLM returned invalid JSON: {exc}") from exc


def generate_with_openai(messages: list[dict[str, str]]) -> dict[str, Any]:
    if not cfg.OPENAI_API_KEY:
        raise GenerationError("OPENAI_API_KEY is not set")
    from openai import OpenAI

    client = OpenAI(api_key=cfg.OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=cfg.LLM_MODEL,
        messages=messages,
        temperature=cfg.LLM_TEMPERATURE,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or "{}"
    return _parse_json_response(content)


def generate_mock_draft(
    doc: ProcessedDocument,
    evidence: list[EvidenceChunk],
    task: str,
) -> dict[str, Any]:
    """
    Deterministic extractive draft when no API key — still grounded in evidence.
    Useful for assessment demos without paid OpenAI access.
    """
    if not evidence:
        raise GenerationError("No evidence chunks to draft from")

    top = evidence[0]
    fields = doc.fields
    case_no = fields.case_numbers[0] if fields.case_numbers else "UNKNOWN"
    parties = " vs ".join(fields.parties[:2]) if fields.parties else "UNKNOWN"
    hearing_date = next(
        (d for d in fields.dates if "202" in d),
        fields.dates[0] if fields.dates else "UNKNOWN",
    )

    claims: list[dict[str, Any]] = []
    for ch in evidence[:5]:
        sentence = ch.text.strip().split(". ")[0][:200]
        if len(sentence) < 20:
            continue
        claims.append(
            {
                "statement": sentence if sentence.endswith(".") else sentence + ".",
                "source_ids": [ch.source_id],
                "supporting_excerpt": ch.text[:240],
            }
        )

    if not claims:
        claims.append(
            {
                "statement": top.text[:200],
                "source_ids": [top.source_id],
                "supporting_excerpt": top.text[:240],
            }
        )

    checklist = [
        {
            "item": "Case number identified",
            "status": "present" if fields.case_numbers else "missing",
            "source_ids": [claims[0]["source_ids"][0]],
        },
        {
            "item": "Parties identified",
            "status": "present" if len(fields.parties) >= 1 else "missing",
            "source_ids": [claims[0]["source_ids"][0]],
        },
        {
            "item": "Hearing or key date identified",
            "status": "present" if fields.dates else "unclear",
            "source_ids": [claims[0]["source_ids"][0]],
        },
    ]

    summary = (
        f"First-pass memo for {case_no}: {parties}. "
        f"Key date referenced: {hearing_date}. Task: {task[:80]}."
    )

    return {
        "draft_type": "case_fact_summary",
        "summary": summary,
        "checklist": checklist,
        "claims": claims,
        "generation_mode": "mock_extractive",
    }


def generate_draft(
    doc: ProcessedDocument,
    evidence: list[EvidenceChunk],
    task: str,
    few_shot_examples: list[str] | None = None,
    force_mock: bool = False,
) -> dict[str, Any]:
    """Generate draft, verify grounding, return final JSON."""
    backend = cfg.LLM_BACKEND.lower()

    if force_mock or backend == "mock":
        draft = generate_mock_draft(doc, evidence, task)
    elif backend == "openai":
        messages = build_prompt_for_document(doc, evidence, task, few_shot_examples)
        draft = generate_with_openai(messages)
        draft["generation_mode"] = "openai"
    else:
        # auto
        if cfg.OPENAI_API_KEY:
            try:
                messages = build_prompt_for_document(doc, evidence, task, few_shot_examples)
                draft = generate_with_openai(messages)
                draft["generation_mode"] = "openai"
            except Exception as exc:
                logger.warning("OpenAI generation failed (%s); using mock draft", exc)
                draft = generate_mock_draft(doc, evidence, task)
        else:
            logger.info("No OPENAI_API_KEY — using mock extractive draft (still grounded)")
            draft = generate_mock_draft(doc, evidence, task)

    draft["doc_id"] = doc.doc_id
    draft["task"] = task
    draft["evidence_ids"] = [ch.source_id for ch in evidence]
    return verify_draft(draft, evidence)
