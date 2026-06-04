"""BM25 sparse index (rank_bm25) with disk persistence."""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path

from legal_rag import config as cfg
from legal_rag.retrieval.models import EvidenceChunk, TextChunk

logger = logging.getLogger(__name__)


@dataclass
class BM25Index:
    chunk_ids: list[str] = field(default_factory=list)
    corpus: list[str] = field(default_factory=list)
    metadatas: list[dict] = field(default_factory=list)
    _bm25: object | None = field(default=None, repr=False)

    def build(self, chunks: list[TextChunk]) -> None:
        from rank_bm25 import BM25Okapi

        self.chunk_ids = [c.chunk_id for c in chunks]
        self.corpus = [c.text for c in chunks]
        self.metadatas = [
            {
                "doc_id": c.doc_id,
                "page_num": c.page_num,
                "source_id": c.chunk_id,
            }
            for c in chunks
        ]
        tokenized = [_tokenize(doc) for doc in self.corpus]
        self._bm25 = BM25Okapi(tokenized)
        logger.info("Built BM25 index with %d documents", len(self.corpus))

    def query(self, query_text: str, top_k: int | None = None) -> list[EvidenceChunk]:
        if not self._bm25 or not self.corpus:
            return []
        k = top_k or cfg.BM25_TOP_K
        tokens = _tokenize(query_text)
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        evidence: list[EvidenceChunk] = []
        for rank, idx in enumerate(ranked, start=1):
            score = float(scores[idx])
            if score <= 0:
                continue
            meta = self.metadatas[idx]
            evidence.append(
                EvidenceChunk(
                    chunk_id=self.chunk_ids[idx],
                    doc_id=str(meta["doc_id"]),
                    text=self.corpus[idx],
                    page_num=int(meta["page_num"]),
                    score=score,
                    source_id=str(meta["source_id"]),
                    rank=rank,
                    retrieval_source="bm25",
                )
            )
        return evidence

    def save(self, path: Path | None = None) -> Path:
        path = path or cfg.BM25_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "chunk_ids": self.chunk_ids,
            "corpus": self.corpus,
            "metadatas": self.metadatas,
            "bm25": self._bm25,
        }
        path.write_bytes(pickle.dumps(payload))
        logger.info("Saved BM25 index to %s", path)
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> BM25Index:
        path = path or cfg.BM25_PATH
        if not path.exists():
            return cls()
        payload = pickle.loads(path.read_bytes())
        index = cls(
            chunk_ids=list(payload["chunk_ids"]),
            corpus=list(payload["corpus"]),
            metadatas=list(payload["metadatas"]),
            _bm25=payload["bm25"],
        )
        return index


def _tokenize(text: str) -> list[str]:
    """Simple English tokenizer (production systems often use NLTK/spacy)."""
    import re

    return re.findall(r"[a-z0-9]+", text.lower())

