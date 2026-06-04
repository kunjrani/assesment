"""Stage 5: capture edit → extract patterns → store few-shot examples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from legal_rag import config as cfg
from legal_rag.feedback.edit_store import capture_from_files
from legal_rag.feedback.few_shot_store import FewShotStore
from legal_rag.feedback.pattern_extractor import extract_patterns

DRAFTS_DIR = cfg.DATA_DIR / "drafts"


def default_draft_path(doc_id: str) -> Path:
    return DRAFTS_DIR / f"{doc_id}_draft.json"


def run_feedback_pipeline(
    doc_id: str,
    task: str,
    edited_path: Path,
    original_path: Path | None = None,
    use_llm_patterns: bool = False,
) -> dict[str, Any]:
    original_path = original_path or default_draft_path(doc_id)
    if not original_path.exists():
        raise FileNotFoundError(f"Original draft not found: {original_path}")
    if not edited_path.exists():
        raise FileNotFoundError(f"Edited draft not found: {edited_path}")

    record = capture_from_files(doc_id, task, original_path, edited_path)
    patterns_payload = extract_patterns(record, use_llm=use_llm_patterns)

    store = FewShotStore()
    store.add_example(
        edit_id=record["edit_id"],
        task=task,
        patterns=patterns_payload.get("patterns", []),
        intent=patterns_payload.get("intent", ""),
        diff=record["diff"],
    )

    result = {
        "edit_id": record["edit_id"],
        "edit_path": record.get("_path"),
        "patterns": patterns_payload,
        "stored_for_future_drafts": True,
    }
    meta_path = cfg.EDITS_DIR / f"{record['edit_id']}_patterns.json"
    meta_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def simulate_operator_edit(doc_id: str) -> Path:
    """
    Create a plausible operator-edited draft from the default draft file.
    For assessment demos when you don't have a real human edit.
    """
    original_path = default_draft_path(doc_id)
    if not original_path.exists():
        raise FileNotFoundError(f"Run draft first: {original_path}")

    original = json.loads(original_path.read_text(encoding="utf-8"))
    edited = json.loads(original_path.read_text(encoding="utf-8"))

    # Typical operator improvements
    edited["summary"] = (
        f"Case No. priority — {edited.get('summary', '')} "
        "Lead with parties and hearing date in the first sentence."
    ).strip()
    for item in edited.get("checklist", []):
        if "date" in str(item.get("item", "")).lower():
            item["status"] = "present"

    edited_path = DRAFTS_DIR / f"{doc_id}_draft_edited.json"
    edited_path.write_text(json.dumps(edited, indent=2, ensure_ascii=False), encoding="utf-8")
    return edited_path
