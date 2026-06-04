"""Data contracts for chunking and retrieval."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TextChunk:
    chunk_id: str
    doc_id: str
    text: str
    page_num: int
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceChunk:
    """Retrieved chunk with score and provenance (Stage 3 output)."""

    chunk_id: str
    doc_id: str
    text: str
    page_num: int
    score: float
    source_id: str
    rank: int
    retrieval_source: str  # "vector" | "bm25" | "rrf"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
