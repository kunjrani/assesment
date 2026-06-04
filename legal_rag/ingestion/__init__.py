"""Stage 1: document ingestion and field extraction."""

from legal_rag.ingestion.extractor import ingest_pdf, make_doc_id
from legal_rag.ingestion.models import IngestionError, ProcessedDocument

__all__ = ["ingest_pdf", "make_doc_id", "ProcessedDocument", "IngestionError"]
