"""Capture operator edits as structured JSON diffs."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from legal_rag import config as cfg


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compute_edit_diff(original: dict[str, Any], edited: dict[str, Any]) -> dict[str, Any]:
    """Diff two draft JSON objects — reusable signal for pattern learning."""
    orig_summary = str(original.get("summary", ""))
    edit_summary = str(edited.get("summary", ""))

    orig_claims = {c.get("statement"): c for c in original.get("claims", []) if c.get("statement")}
    edit_claims = {c.get("statement"): c for c in edited.get("claims", []) if c.get("statement")}

    claims_added = [s for s in edit_claims if s not in orig_claims]
    claims_removed = [s for s in orig_claims if s not in edit_claims]

    orig_check = {c.get("item"): c for c in original.get("checklist", []) if c.get("item")}
    edit_check = {c.get("item"): c for c in edited.get("checklist", []) if c.get("item")}
    checklist_changes: list[dict[str, Any]] = []
    for item, edited_row in edit_check.items():
        orig_row = orig_check.get(item)
        if not orig_row:
            checklist_changes.append({"item": item, "change": "added", "to": edited_row})
        elif orig_row.get("status") != edited_row.get("status"):
            checklist_changes.append(
                {
                    "item": item,
                    "change": "status",
                    "from": orig_row.get("status"),
                    "to": edited_row.get("status"),
                }
            )

    return {
        "summary_changed": orig_summary != edit_summary,
        "summary_before": orig_summary,
        "summary_after": edit_summary,
        "claims_added": claims_added,
        "claims_removed": claims_removed,
        "checklist_changes": checklist_changes,
    }


def save_edit_record(
    doc_id: str,
    task: str,
    original: dict[str, Any],
    edited: dict[str, Any],
    edit_id: str | None = None,
) -> dict[str, Any]:
    """Persist edit diff + snapshots under data/edits/."""
    cfg.EDITS_DIR.mkdir(parents=True, exist_ok=True)
    edit_id = edit_id or f"{doc_id}_{uuid.uuid4().hex[:8]}"
    diff = compute_edit_diff(original, edited)

    record = {
        "edit_id": edit_id,
        "doc_id": doc_id,
        "task": task,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "diff": diff,
        "original_draft": original,
        "edited_draft": edited,
    }
    path = cfg.EDITS_DIR / f"{edit_id}.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    record["_path"] = str(path)
    return record


def capture_from_files(
    doc_id: str,
    task: str,
    original_path: Path,
    edited_path: Path,
) -> dict[str, Any]:
    original = _load_json(original_path)
    edited = _load_json(edited_path)
    return save_edit_record(doc_id, task, original, edited)
