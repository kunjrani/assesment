"""Stage 2 orchestration: chunk → embed → Chroma + BM25."""

from __future__ import annotations

import logging
from pathlib import Path

from legal_rag import config as cfg
from legal_rag.ingestion.models import ProcessedDocument
from legal_rag.retrieval.bm25_store import BM25Index
from legal_rag.retrieval.models import TextChunk
from legal_rag.retrieval.chunker import chunk_document
from legal_rag.retrieval.embedder import get_embedding_provider
from legal_rag.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


class IndexingError(Exception):
    pass


def index_document(doc: ProcessedDocument, rebuild_bm25: bool = False) -> dict:
    """
    Index one processed document.
    - Vector store: upsert by chunk_id
    - BM25: append to corpus (or rebuild if rebuild_bm25=True)
    """
    chunks = chunk_document(doc)
    if not chunks:
        raise IndexingError(f"No chunks produced for doc_id={doc.doc_id}")

    embedder = get_embedding_provider()
    vector_store = VectorStore(embedder)
    added = vector_store.upsert_chunks(chunks)

    bm25 = BM25Index.load()
    prefix = f"{doc.doc_id}::"
    combined: list[TextChunk] = []
    if bm25.corpus and not rebuild_bm25:
        for cid, text, meta in zip(bm25.chunk_ids, bm25.corpus, bm25.metadatas):
            if cid.startswith(prefix):
                continue
            combined.append(
                TextChunk(
                    chunk_id=cid,
                    doc_id=str(meta["doc_id"]),
                    text=text,
                    page_num=int(meta["page_num"]),
                    chunk_index=0,
                )
            )
    combined.extend(chunks)
    bm25 = BM25Index()
    bm25.build(combined)
    bm25.save()

    return {
        "doc_id": doc.doc_id,
        "chunks_indexed": added,
        "embedding_backend": embedder.name,
        "vector_store_count": vector_store.count,
        "bm25_docs": len(bm25.corpus),
    }


def index_from_processed_json(path: Path, rebuild_bm25: bool = False) -> dict:
    doc = ProcessedDocument.load(path)
    return index_document(doc, rebuild_bm25=rebuild_bm25)
