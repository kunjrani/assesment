"""
Hybrid retrieval: vector + BM25 fused with Reciprocal Rank Fusion (RRF).

RRF: score(d) = sum 1/(k + rank) — Cormack et al. SIGIR 2009;
used in LangChain EnsembleRetriever and trustgraph hybrid RAG.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from legal_rag import config as cfg
from legal_rag.retrieval.bm25_store import BM25Index
from legal_rag.retrieval.embedder import get_embedding_provider
from legal_rag.retrieval.models import EvidenceChunk
from legal_rag.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    result_lists: list[list[EvidenceChunk]],
    top_k: int | None = None,
    k: int | None = None,
) -> list[EvidenceChunk]:
    """Fuse ranked lists with RRF; dedupe by chunk_id."""
    k = k or cfg.RRF_K
    top_k = top_k or cfg.FUSION_TOP_K
    scores: dict[str, float] = defaultdict(float)
    best: dict[str, EvidenceChunk] = {}

    for results in result_lists:
        for rank, item in enumerate(results, start=1):
            scores[item.chunk_id] += 1.0 / (k + rank)
            if item.chunk_id not in best or item.score > best[item.chunk_id].score:
                best[item.chunk_id] = item

    fused_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)[:top_k]
    fused: list[EvidenceChunk] = []
    for rank, cid in enumerate(fused_ids, start=1):
        base = best[cid]
        fused.append(
            EvidenceChunk(
                chunk_id=base.chunk_id,
                doc_id=base.doc_id,
                text=base.text,
                page_num=base.page_num,
                score=scores[cid],
                source_id=base.source_id,
                rank=rank,
                retrieval_source="rrf",
            )
        )
    return fused


class HybridRetriever:
    def __init__(self) -> None:
        self._embedder = get_embedding_provider()
        self._vector = VectorStore(self._embedder)
        self._bm25 = BM25Index.load()

    def retrieve(self, query: str, top_k: int | None = None) -> list[EvidenceChunk]:
        if self._vector.count == 0 and not self._bm25.corpus:
            raise ValueError("No indexed documents. Run: python -m legal_rag.main index --doc-id <id>")

        vector_hits = self._vector.query(query, top_k=cfg.VECTOR_TOP_K)
        bm25_hits = self._bm25.query(query, top_k=cfg.BM25_TOP_K)
        fused = reciprocal_rank_fusion([vector_hits, bm25_hits], top_k=top_k)
        logger.info(
            "Retrieved vector=%d bm25=%d fused=%d",
            len(vector_hits),
            len(bm25_hits),
            len(fused),
        )
        return fused


def prepare_query(task: str, extra_context: str = "") -> str:
    """Light query expansion for legal drafting tasks."""
    parts = [task.strip()]
    if extra_context.strip():
        parts.append(extra_context.strip())
    return "\n".join(parts)
