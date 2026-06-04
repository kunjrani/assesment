"""End-to-end draft pipeline: retrieve → generate → save."""

from __future__ import annotations

import json
from pathlib import Path

from legal_rag import config as cfg
from legal_rag.ingestion.models import ProcessedDocument
from legal_rag.retrieval.hybrid_retriever import HybridRetriever, prepare_query

from legal_rag.feedback.few_shot_store import FewShotStore, apply_patterns_to_mock_draft
from legal_rag.generation.generator import generate_draft

DRAFTS_DIR = cfg.DATA_DIR / "drafts"


def run_draft_pipeline(
    doc_id: str,
    task: str,
    extra_context: str = "",
    top_k: int | None = None,
    force_mock: bool = False,
) -> dict:
    processed_path = cfg.PROCESSED_DIR / f"{doc_id}.json"
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed document not found: {processed_path}")

    doc = ProcessedDocument.load(processed_path)
    query = prepare_query(task, extra_context)
    evidence = HybridRetriever().retrieve(query, top_k=top_k)

    few_shot = FewShotStore().retrieve(task)
    draft = generate_draft(doc, evidence, task, few_shot_examples=few_shot, force_mock=force_mock)

    if draft.get("generation_mode") == "mock_extractive" and few_shot:
        draft = apply_patterns_to_mock_draft(draft, few_shot)

    DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DRAFTS_DIR / f"{doc_id}_draft.json"
    out_path.write_text(json.dumps(draft, indent=2, ensure_ascii=False), encoding="utf-8")
    draft["_output_path"] = str(out_path)
    return draft
