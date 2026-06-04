"""
Post-generation grounding verifier (lexical overlap + source_id validation).

Inspired by claim→source auditing in Dokis / GroundGuard-style constrained checks.
"""

from __future__ import annotations

import re
from typing import Any

from legal_rag.retrieval.models import EvidenceChunk

_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _overlap_score(claim: str, excerpt: str) -> float:
    claim_t = _tokens(claim)
    if not claim_t:
        return 0.0
    excerpt_t = _tokens(excerpt)
    if not excerpt_t:
        return 0.0
    return len(claim_t & excerpt_t) / len(claim_t)


def verify_draft(
    draft: dict[str, Any],
    evidence: list[EvidenceChunk],
    min_overlap: float = 0.25,
) -> dict[str, Any]:
    """
    Validate claims against cited chunks. Unsupported claims are moved to
    unsupported_removed; supported claims are returned in verified_claims.
    """
    by_id = {ch.source_id: ch for ch in evidence}
    by_id.update({ch.chunk_id: ch for ch in evidence})

    verified: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []

    for claim in draft.get("claims", []):
        statement = str(claim.get("statement", "")).strip()
        source_ids = list(claim.get("source_ids") or [])
        excerpt = str(claim.get("supporting_excerpt", "")).strip()

        if not statement:
            removed.append({**claim, "reason": "empty_statement"})
            continue

        if not source_ids:
            removed.append({**claim, "reason": "missing_source_ids"})
            continue

        best_score = 0.0
        for sid in source_ids:
            chunk = by_id.get(sid)
            if not chunk:
                continue
            text = excerpt or chunk.text
            best_score = max(best_score, _overlap_score(statement, text))

        if best_score >= min_overlap:
            claim["grounding_score"] = round(best_score, 3)
            verified.append(claim)
        else:
            removed.append({**claim, "reason": "low_overlap", "grounding_score": round(best_score, 3)})

    draft["claims"] = verified
    draft["unsupported_removed"] = removed
    draft["grounding_passed"] = len(removed) == 0
    draft["grounding_stats"] = {
        "verified": len(verified),
        "removed": len(removed),
    }
    return draft
