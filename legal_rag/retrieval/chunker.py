"""
Semantic chunking for processed documents.

Pattern: LangChain RecursiveCharacterTextSplitter with token-aware sizing
(see LangChain docs + common RAG tutorials).
"""

from __future__ import annotations

import logging

from legal_rag import config as cfg
from legal_rag.ingestion.models import ProcessedDocument
from legal_rag.retrieval.models import TextChunk

logger = logging.getLogger(__name__)


def _token_length_fn():
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")

        def length(text: str) -> int:
            return len(enc.encode(text))

        return length
    except Exception:
        # ~4 chars per token heuristic
        return lambda text: max(1, len(text) // 4)


def chunk_document(doc: ProcessedDocument) -> list[TextChunk]:
    """Split each page into overlapping chunks with stable IDs."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    length_fn = _token_length_fn()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.CHUNK_SIZE,
        chunk_overlap=cfg.CHUNK_OVERLAP,
        length_function=length_fn,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[TextChunk] = []
    global_index = 0

    for page in doc.pages:
        text = (page.text or "").strip()
        if not text:
            continue

        pieces = splitter.split_text(text)
        for local_idx, piece in enumerate(pieces):
            piece = piece.strip()
            if not piece:
                continue
            chunk_id = f"{doc.doc_id}::p{page.page_num}::c{local_idx}"
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    text=piece,
                    page_num=page.page_num,
                    chunk_index=global_index,
                    metadata={
                        "extraction_method": page.extraction_method,
                        "page_confidence": page.confidence,
                        "document_type": doc.fields.document_type,
                    },
                )
            )
            global_index += 1

    if not chunks and doc.full_text.strip():
        # Fallback: chunk full document if pages were empty but full_text exists
        for local_idx, piece in enumerate(splitter.split_text(doc.full_text)):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(
                TextChunk(
                    chunk_id=f"{doc.doc_id}::p0::c{local_idx}",
                    doc_id=doc.doc_id,
                    text=piece,
                    page_num=0,
                    chunk_index=local_idx,
                    metadata={"document_type": doc.fields.document_type},
                )
            )

    logger.info("Chunked doc_id=%s into %d chunks", doc.doc_id, len(chunks))
    return chunks
