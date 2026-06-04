"""ChromaDB vector store wrapper."""

from __future__ import annotations

import logging
from typing import Any

from legal_rag import config as cfg
from legal_rag.retrieval.embedder import EmbeddingProvider
from legal_rag.retrieval.models import EvidenceChunk, TextChunk

logger = logging.getLogger(__name__)

COLLECTION_NAME = "legal_chunks"


class VectorStore:
    def __init__(self, embedder: EmbeddingProvider) -> None:
        import chromadb
        from chromadb.config import Settings

        self._embedder = embedder
        self._client = chromadb.PersistentClient(
            path=str(cfg.CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(self, chunks: list[TextChunk]) -> int:
        if not chunks:
            return 0
        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [_chunk_metadata(c) for c in chunks]
        embeddings = self._embedder.embed_documents(documents)
        self._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info("Upserted %d chunks into Chroma (%s)", len(chunks), self._embedder.name)
        return len(chunks)

    def query(self, query_text: str, top_k: int | None = None) -> list[EvidenceChunk]:
        if self._collection.count() == 0:
            return []
        k = top_k or cfg.VECTOR_TOP_K
        embedding = self._embedder.embed_query(query_text)
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        return _results_to_evidence(results, source="vector")

    @property
    def count(self) -> int:
        return self._collection.count()


def _chunk_metadata(chunk: TextChunk) -> dict[str, Any]:
    meta = {
        "doc_id": chunk.doc_id,
        "page_num": chunk.page_num,
        "chunk_index": chunk.chunk_index,
        "source_id": chunk.chunk_id,
    }
    for key, value in chunk.metadata.items():
        if value is None:
            continue
        meta[key] = value if isinstance(value, (str, int, float, bool)) else str(value)
    return meta


def _results_to_evidence(results: dict, source: str) -> list[EvidenceChunk]:
    ids = (results.get("ids") or [[]])[0]
    docs = (results.get("documents") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]
    dists = (results.get("distances") or [[]])[0]

    evidence: list[EvidenceChunk] = []
    for rank, chunk_id in enumerate(ids):
        text = docs[rank] if rank < len(docs) else ""
        meta = metas[rank] if rank < len(metas) else {}
        # Chroma cosine distance: lower is better -> convert to similarity-like score
        dist = dists[rank] if rank < len(dists) else 1.0
        score = 1.0 / (1.0 + float(dist))
        evidence.append(
            EvidenceChunk(
                chunk_id=chunk_id,
                doc_id=str(meta.get("doc_id", "")),
                text=text,
                page_num=int(meta.get("page_num", 0)),
                score=score,
                source_id=str(meta.get("source_id", chunk_id)),
                rank=rank + 1,
                retrieval_source=source,
            )
        )
    return evidence
