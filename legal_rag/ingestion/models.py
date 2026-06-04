"""Data contracts for Stage 1 document ingestion."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

ExtractionMethod = Literal["native", "ocr", "hybrid"]


@dataclass
class PageBlock:
    page_num: int
    text: str
    extraction_method: ExtractionMethod
    confidence: float
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StructuredFields:
    parties: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    case_numbers: list[str] = field(default_factory=list)
    document_type: str | None = None
    jurisdictions: list[str] = field(default_factory=list)
    statute_refs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProcessedDocument:
    doc_id: str
    source_path: str
    pages: list[PageBlock]
    full_text: str
    fields: StructuredFields
    ingest_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def low_confidence_pages(self) -> list[int]:
        return [p.page_num for p in self.pages if p.confidence < 0.5]

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "source_path": self.source_path,
            "pages": [p.to_dict() for p in self.pages],
            "full_text": self.full_text,
            "fields": self.fields.to_dict(),
            "ingest_metadata": self.ingest_metadata,
        }

    def save(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{self.doc_id}.json"
        payload = self.to_dict()
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return out_path

    @classmethod
    def load(cls, path: Path) -> ProcessedDocument:
        data = json.loads(path.read_text(encoding="utf-8"))
        pages = [
            PageBlock(
                page_num=p["page_num"],
                text=p["text"],
                extraction_method=p["extraction_method"],
                confidence=float(p["confidence"]),
                warnings=list(p.get("warnings", [])),
            )
            for p in data["pages"]
        ]
        fields_data = data.get("fields", {})
        fields = StructuredFields(
            parties=list(fields_data.get("parties", [])),
            dates=list(fields_data.get("dates", [])),
            case_numbers=list(fields_data.get("case_numbers", [])),
            document_type=fields_data.get("document_type"),
            jurisdictions=list(fields_data.get("jurisdictions", [])),
            statute_refs=list(fields_data.get("statute_refs", [])),
        )
        return cls(
            doc_id=data["doc_id"],
            source_path=data["source_path"],
            pages=pages,
            full_text=data["full_text"],
            fields=fields,
            ingest_metadata=dict(data.get("ingest_metadata", {})),
        )


class IngestionError(Exception):
    """Raised when a document cannot be processed."""


def build_full_text(pages: list[PageBlock]) -> str:
    """Concatenate pages with stable markers for downstream chunking/citations."""
    parts: list[str] = []
    for page in pages:
        marker = f"--- PAGE {page.page_num} ---"
        parts.append(f"{marker}\n{page.text.strip()}")
    return "\n\n".join(parts)


def new_ingest_metadata(source_path: Path, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "source_filename": source_path.name,
        "source_size_bytes": source_path.stat().st_size if source_path.exists() else 0,
    }
    if extra:
        meta.update(extra)
    return meta
