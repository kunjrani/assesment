"""Assemble system prompt, evidence context, and optional few-shot examples."""

from __future__ import annotations

from legal_rag.ingestion.models import ProcessedDocument, StructuredFields
from legal_rag.retrieval.models import EvidenceChunk

SYSTEM_PROMPT = """You are a legal operations assistant. Produce a first-pass internal memo \
grounded ONLY in the provided evidence chunks.

Rules:
- Every factual claim MUST cite one or more source_ids from the evidence.
- If a fact is not in the evidence, write "UNKNOWN" — do not guess.
- Do not provide legal advice or conclusions about who will win.
- Output valid JSON matching the schema given in the user message.
"""

DRAFT_JSON_SCHEMA = """
{
  "draft_type": "case_fact_summary",
  "summary": "string — 2-4 sentences",
  "checklist": [
    {"item": "string", "status": "present|missing|unclear", "source_ids": ["chunk_id"]}
  ],
  "claims": [
    {
      "statement": "string",
      "source_ids": ["chunk_id"],
      "supporting_excerpt": "short quote from evidence"
    }
  ]
}
"""


def format_evidence_block(chunks: list[EvidenceChunk]) -> str:
    lines: list[str] = []
    for ch in chunks:
        lines.append(f"[{ch.source_id}] (page {ch.page_num})\n{ch.text.strip()}\n")
    return "\n".join(lines)


def format_fields_block(fields: StructuredFields) -> str:
    parts: list[str] = []
    if fields.document_type:
        parts.append(f"document_type: {fields.document_type}")
    if fields.case_numbers:
        parts.append(f"case_numbers: {', '.join(fields.case_numbers)}")
    if fields.parties:
        parts.append(f"parties: {', '.join(fields.parties)}")
    if fields.dates:
        parts.append(f"dates: {', '.join(fields.dates)}")
    if fields.jurisdictions:
        parts.append(f"jurisdictions: {', '.join(fields.jurisdictions)}")
    if fields.statute_refs:
        parts.append(f"statute_refs: {', '.join(fields.statute_refs)}")
    return "\n".join(parts) if parts else "(no structured fields extracted)"


def build_messages(
    task: str,
    evidence: list[EvidenceChunk],
    fields: StructuredFields,
    few_shot_examples: list[str] | None = None,
) -> list[dict[str, str]]:
    few_shot_block = ""
    if few_shot_examples:
        few_shot_block = "Learned operator preferences:\n" + "\n---\n".join(few_shot_examples) + "\n\n"

    user_content = f"""Task: {task}

Structured fields from ingestion:
{format_fields_block(fields)}

Evidence chunks (cite using source_id in brackets):
{format_evidence_block(evidence)}

{few_shot_block}Return JSON only with this schema:
{DRAFT_JSON_SCHEMA}
"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_prompt_for_document(
    doc: ProcessedDocument,
    evidence: list[EvidenceChunk],
    task: str,
    few_shot_examples: list[str] | None = None,
) -> list[dict[str, str]]:
    return build_messages(task, evidence, doc.fields, few_shot_examples)
