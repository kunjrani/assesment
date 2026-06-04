"""Structured field extraction from legal-style text (regex-first, deterministic)."""

from __future__ import annotations

import re
from typing import Iterable

from legal_rag.ingestion.models import StructuredFields

# --- Regex patterns (legal-style heuristics) ---

CASE_NUMBER_PATTERNS = [
    re.compile(
        r"\b(?:Case|Civil|Criminal|Docket|Cause)\s*(?:No\.?|Number|#)\s*[:#]?\s*"
        r"([A-Za-z0-9][A-Za-z0-9\-/_\.]{2,40})\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b([A-Z]{1,4}-\d{2,4}-\d{3,8})\b"),
]

DATE_PATTERNS = [
    re.compile(
        r"\b(?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+\d{1,2},?\s+\d{4}\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
]

PARTY_LINE_PATTERNS = [
    re.compile(
        r"^\s*(Plaintiff|Petitioner|Applicant|Defendant|Respondent|Appellant|Appellee)"
        r"(?:s)?\s*[:\-]\s*(.+?)\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(
        r"\b(?:Plaintiff|Petitioner|Defendant|Respondent)\s*[:\-]\s*"
        r"([A-Z][A-Za-z0-9\s\.,&'\-]{2,80})",
        re.IGNORECASE,
    ),
]

JURISDICTION_PATTERNS = [
    re.compile(
        r"\b(?:In the|Before the)\s+"
        r"(District|Superior|Supreme|Appellate|Circuit|County|High)\s+Court\s+of\s+([^,\n]{3,60})",
        re.IGNORECASE,
    ),
    re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:District|County)\b"),
]

STATUTE_PATTERN = re.compile(
    r"\b(\d+\s+U\.?\s*S\.?\s*C\.?\s*(?:§|Sec\.?|Section)\s*\d+[\w\-]*)",
    re.IGNORECASE,
)

DOCUMENT_TYPE_KEYWORDS: list[tuple[str, str]] = [
    ("notice of hearing", "notice_of_hearing"),
    ("memorandum", "memorandum"),
    ("affidavit", "affidavit"),
    ("complaint", "complaint"),
    ("motion", "motion"),
    ("order", "order"),
    ("title review", "title_review"),
    ("settlement", "settlement_agreement"),
    ("contract", "contract"),
]


def _unique_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in items:
        value = " ".join(raw.split()).strip()
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _detect_document_type(text: str) -> str | None:
    lower = text.lower()[:8000]
    for phrase, doc_type in DOCUMENT_TYPE_KEYWORDS:
        if phrase in lower:
            return doc_type
    return None


def extract_structured_fields(text: str) -> StructuredFields:
    """Extract structured fields from full document text."""
    case_numbers: list[str] = []
    for pattern in CASE_NUMBER_PATTERNS:
        for match in pattern.finditer(text):
            case_numbers.append(match.group(1).strip())

    dates: list[str] = []
    for pattern in DATE_PATTERNS:
        dates.extend(m.group(0).strip() for m in pattern.finditer(text))

    parties: list[str] = []
    for pattern in PARTY_LINE_PATTERNS:
        for match in pattern.finditer(text):
            # Last group is the captured name
            name = match.group(match.lastindex or 1).strip()
            if len(name) >= 2:
                parties.append(name)

    jurisdictions: list[str] = []
    for pattern in JURISDICTION_PATTERNS:
        for match in pattern.finditer(text):
            jurisdictions.append(match.group(0).strip())

    statute_refs = [m.group(1).strip() for m in STATUTE_PATTERN.finditer(text)]

    return StructuredFields(
        parties=_unique_preserve_order(parties)[:20],
        dates=_unique_preserve_order(dates)[:30],
        case_numbers=_unique_preserve_order(case_numbers)[:15],
        document_type=_detect_document_type(text),
        jurisdictions=_unique_preserve_order(jurisdictions)[:10],
        statute_refs=_unique_preserve_order(statute_refs)[:20],
    )
