"""
PDF text extraction with native-first routing and OCR fallback.

Pipeline inspired by:
- jsvine/pdfplumber (native text + tables)
- bibhu342/PDF-Parser-Pro (OCR fallback path)
- pdfplumber discussion #717 (scanned PDF detection)
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path
from typing import Callable

import legal_rag.config as cfg
from legal_rag.ingestion.field_parser import extract_structured_fields
from legal_rag.ingestion.models import (
    ExtractionMethod,
    IngestionError,
    PageBlock,
    ProcessedDocument,
    build_full_text,
    new_ingest_metadata,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text quality helpers
# ---------------------------------------------------------------------------


def _garbled_ratio(text: str) -> float:
    """Fraction of non-alphanumeric characters — high ratio suggests OCR noise."""
    if not text:
        return 1.0
    alnum = sum(1 for c in text if c.isalnum() or c.isspace())
    return 1.0 - (alnum / len(text))


def _needs_ocr(text: str) -> bool:
    stripped = (text or "").strip()
    if len(stripped) < cfg.MIN_PAGE_TEXT_CHARS:
        return True
    if _garbled_ratio(stripped) > cfg.GARBLED_RATIO_THRESHOLD:
        return True
    return False


def normalize_text(text: str) -> str:
    """Clean extracted text for downstream chunking."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # De-hyphenate line breaks: "plain-\ntiff" -> "plaintiff"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Collapse excessive whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Native extractors
# ---------------------------------------------------------------------------


def _extract_page_pdfplumber(page: object) -> str:
    text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
    if text.strip():
        return text

    tables: list[str] = []
    for table in page.extract_tables() or []:
        if not table:
            continue
        rows = [" | ".join(str(cell or "").strip() for cell in row) for row in table]
        tables.append("\n".join(rows))
    return "\n\n".join(tables) if tables else ""


def _extract_native_pdfplumber(pdf_path: Path) -> list[tuple[int, str]]:
    import pdfplumber

    pages: list[tuple[int, str]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            pages.append((i, _extract_page_pdfplumber(page)))
    return pages


def _extract_native_pymupdf(pdf_path: Path) -> list[tuple[int, str]]:
    import fitz  # pymupdf

    pages: list[tuple[int, str]] = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            pages.append((i, page.get_text("text") or ""))
    return pages


def _extract_native(pdf_path: Path) -> list[tuple[int, str]]:
    """Try pdfplumber, fall back to pymupdf."""
    try:
        return _extract_native_pdfplumber(pdf_path)
    except Exception as exc:
        logger.warning("pdfplumber failed (%s); trying pymupdf", exc)
    try:
        return _extract_native_pymupdf(pdf_path)
    except Exception as exc:
        raise IngestionError(f"Native extraction failed for {pdf_path.name}: {exc}") from exc


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------


def _configure_tesseract() -> None:
    if not cfg.TESSERACT_CMD:
        return
    import pytesseract

    pytesseract.pytesseract.tesseract_cmd = cfg.TESSERACT_CMD


def _ocr_page(pdf_path: Path, page_num: int) -> tuple[str, float]:
    """
    OCR a single page. Returns (text, confidence 0-1).
    Confidence is mean Tesseract word confidence when available.
    """
    from pdf2image import convert_from_path
    import pytesseract

    _configure_tesseract()

    kwargs: dict = {
        "dpi": cfg.OCR_DPI,
        "first_page": page_num,
        "last_page": page_num,
    }
    if cfg.POPPLER_PATH:
        kwargs["poppler_path"] = cfg.POPPLER_PATH

    try:
        images = convert_from_path(str(pdf_path), **kwargs)
    except Exception as exc:
        raise IngestionError(
            f"OCR render failed for page {page_num}. "
            f"Install Poppler and set POPPLER_PATH if needed. Detail: {exc}"
        ) from exc

    if not images:
        return "", 0.0

    image = images[0]
    try:
        data = pytesseract.image_to_data(
            image, lang=cfg.OCR_LANG, output_type=pytesseract.Output.DICT
        )
        words = [
            str(w).strip()
            for w, conf in zip(data.get("text", []), data.get("conf", []))
            if w and str(w).strip() and int(float(conf)) >= 0
        ]
        confs = [
            int(float(c))
            for c in data.get("conf", [])
            if c not in (-1, "-1") and str(c).strip()
        ]
        valid_confs = [c for c in confs if c >= 0]
        text = " ".join(words)
        confidence = (sum(valid_confs) / len(valid_confs) / 100.0) if valid_confs else 0.4
        if not text:
            text = pytesseract.image_to_string(image, lang=cfg.OCR_LANG)
            confidence = 0.35
        return normalize_text(text), min(max(confidence, 0.0), 1.0)
    except Exception as exc:
        raise IngestionError(f"Tesseract OCR failed on page {page_num}: {exc}") from exc


# ---------------------------------------------------------------------------
# Page routing
# ---------------------------------------------------------------------------


def _process_page(
    pdf_path: Path,
    page_num: int,
    native_text: str,
    ocr_fn: Callable[[Path, int], tuple[str, float]] = _ocr_page,
) -> PageBlock:
    warnings: list[str] = []
    native_clean = normalize_text(native_text)
    method: ExtractionMethod = "native"
    confidence = 0.95
    final_text = native_clean

    if _needs_ocr(native_clean):
        try:
            ocr_text, ocr_conf = ocr_fn(pdf_path, page_num)
        except IngestionError:
            if len(native_clean) >= cfg.MIN_PAGE_TEXT_CHARS // 2:
                warnings.append("ocr_failed_using_partial_native")
                return PageBlock(
                    page_num=page_num,
                    text=native_clean,
                    extraction_method="native",
                    confidence=0.45,
                    warnings=warnings,
                )
            raise

        if ocr_text and len(ocr_text) >= len(native_clean):
            final_text = ocr_text
            method = "ocr" if not native_clean else "hybrid"
            confidence = ocr_conf
        elif native_clean:
            final_text = native_clean
            method = "hybrid"
            confidence = 0.55
            warnings.append("ocr_lower_quality_kept_native")
        else:
            final_text = ocr_text
            method = "ocr"
            confidence = ocr_conf

    if not final_text.strip():
        warnings.append("empty_page")
        confidence = 0.0
    elif confidence < 0.5:
        warnings.append("low_confidence")

    if method == "ocr" and _garbled_ratio(final_text) > cfg.GARBLED_RATIO_THRESHOLD:
        warnings.append("possible_garbled_ocr")

    return PageBlock(
        page_num=page_num,
        text=final_text,
        extraction_method=method,
        confidence=round(confidence, 3),
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def make_doc_id(pdf_path: Path) -> str:
    stem = pdf_path.stem.lower().replace(" ", "_")
    safe = re.sub(r"[^a-z0-9_\-]", "", stem) or "document"
    return safe[:80]


def ingest_pdf(pdf_path: Path, doc_id: str | None = None) -> ProcessedDocument:
    """
    Full Stage 1 pipeline: extract -> normalize -> structured fields -> ProcessedDocument.
    """
    pdf_path = pdf_path.resolve()
    if not pdf_path.exists():
        raise IngestionError(f"File not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise IngestionError(f"Expected a PDF file, got: {pdf_path.suffix}")

    doc_id = doc_id or make_doc_id(pdf_path)
    logger.info("Ingesting %s as doc_id=%s", pdf_path.name, doc_id)

    native_pages = _extract_native(pdf_path)
    if not native_pages:
        raise IngestionError(f"No pages found in {pdf_path.name}")

    page_blocks: list[PageBlock] = []
    ocr_count = 0
    for page_num, native_text in native_pages:
        block = _process_page(pdf_path, page_num, native_text)
        if block.extraction_method in ("ocr", "hybrid"):
            ocr_count += 1
        page_blocks.append(block)

    non_empty = [p for p in page_blocks if p.text.strip()]
    if not non_empty:
        raise IngestionError(
            f"No extractable text in {pdf_path.name}. "
            "Check OCR setup (Tesseract + Poppler) or provide a digital PDF."
        )

    full_text = build_full_text(page_blocks)
    fields = extract_structured_fields(full_text)

    metadata = new_ingest_metadata(
        pdf_path,
        {
            "total_pages": len(page_blocks),
            "non_empty_pages": len(non_empty),
            "ocr_pages": ocr_count,
            "native_pages": sum(1 for p in page_blocks if p.extraction_method == "native"),
            "low_confidence_pages": [p.page_num for p in page_blocks if p.confidence < 0.5],
        },
    )

    return ProcessedDocument(
        doc_id=doc_id,
        source_path=str(pdf_path),
        pages=page_blocks,
        full_text=full_text,
        fields=fields,
        ingest_metadata=metadata,
    )
