"""Store and retrieve operator edit patterns for few-shot prompt injection."""

from __future__ import annotations

import json
import logging
from typing import Any

from legal_rag import config as cfg
from legal_rag.retrieval.embedder import get_embedding_provider

logger = logging.getLogger(__name__)

COLLECTION_NAME = "operator_feedback"


class FewShotStore:
    def __init__(self) -> None:
        import chromadb
        from chromadb.config import Settings

        self._embedder = get_embedding_provider()
        self._client = chromadb.PersistentClient(
            path=str(cfg.FEEDBACK_CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add_example(
        self,
        edit_id: str,
        task: str,
        patterns: list[str],
        intent: str,
        diff: dict[str, Any],
    ) -> None:
        document = self._format_example(intent, patterns, diff)
        embedding = self._embedder.embed_documents([document])[0]
        self._collection.upsert(
            ids=[edit_id],
            documents=[document],
            embeddings=[embedding],
            metadatas=[
                {
                    "task": task[:500],
                    "intent": intent[:1000],
                    "patterns": json.dumps(patterns)[:2000],
                }
            ],
        )
        logger.info("Stored feedback example edit_id=%s", edit_id)

    def retrieve(self, task: str, top_k: int | None = None) -> list[str]:
        k = top_k or cfg.FEEDBACK_TOP_K
        if self._collection.count() == 0:
            return []

        embedding = self._embedder.embed_query(task)
        n = min(k, self._collection.count())
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        docs = (results.get("documents") or [[]])[0]
        return [d for d in docs if d]

    @staticmethod
    def _format_example(intent: str, patterns: list[str], diff: dict[str, Any]) -> str:
        before = diff.get("summary_before", "")[:400]
        after = diff.get("summary_after", "")[:400]
        pattern_text = "\n".join(f"- {p}" for p in patterns)
        return (
            f"Operator intent: {intent}\n"
            f"Rules learned:\n{pattern_text}\n"
            f"Before summary: {before}\n"
            f"After summary: {after}"
        )


def apply_patterns_to_mock_draft(draft: dict[str, Any], examples: list[str]) -> dict[str, Any]:
    """Apply learned preferences to mock drafts (visible improvement without API)."""
    if not examples:
        return draft

    combined = " ".join(examples).lower()
    summary = str(draft.get("summary", ""))
    fields_prefix = ""

    if "case number" in combined or "case no" in combined:
        # Move case number fragment to front if present in summary
        for token in summary.split():
            if "-" in token and any(c.isdigit() for c in token):
                fields_prefix = f"Case {token}: "
                summary = summary.replace(f"for {token}:", "for").replace(token, "", 1).strip()
                break

    if "party names" in combined or "lead with party" in combined or "parties" in combined:
        if "vs" in summary and not summary.lower().startswith(("plaintiff", "petitioner", "case")):
            # Already has parties mid-sentence — prepend emphasis
            summary = f"Parties — {summary}"

    if "longer" in combined or "more explicit" in combined:
        summary = summary + " (Operator preference: include explicit dates and cited facts.)"

    if fields_prefix:
        summary = fields_prefix + summary

    draft["summary"] = summary
    draft["few_shot_applied"] = True
    return draft
