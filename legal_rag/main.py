"""
CLI entry point for the legal RAG pipeline.

Usage (from project parent directory):
    python -m legal_rag.main ingest --pdf legal_rag/samples/sample_input.pdf
    python -m legal_rag.main ingest --pdf path/to/doc.pdf --doc-id my_case_001
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import legal_rag.config as cfg
from legal_rag.ingestion import IngestionError, ingest_pdf, make_doc_id
from legal_rag.feedback import run_feedback_pipeline, simulate_operator_edit
from legal_rag.generation import GenerationError, run_draft_pipeline
from legal_rag.retrieval import HybridRetriever, IndexingError, index_from_processed_json, prepare_query

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("legal_rag")


def cmd_ingest(args: argparse.Namespace) -> int:
    pdf_path = Path(args.pdf).resolve()
    doc_id = args.doc_id or make_doc_id(pdf_path)

    try:
        doc = ingest_pdf(pdf_path, doc_id=doc_id)
    except IngestionError as exc:
        logger.error("%s", exc)
        return 1

    out_path = doc.save(cfg.PROCESSED_DIR)
    logger.info("Saved processed document: %s", out_path)
    logger.info(
        "Pages=%d | OCR pages=%d | doc_type=%s | case_numbers=%s",
        doc.page_count,
        doc.ingest_metadata.get("ocr_pages", 0),
        doc.fields.document_type,
        doc.fields.case_numbers or "none",
    )

    if args.verbose:
        summary = {
            "doc_id": doc.doc_id,
            "output": str(out_path),
            "fields": doc.fields.to_dict(),
            "ingest_metadata": doc.ingest_metadata,
            "page_preview": [
                {
                    "page": p.page_num,
                    "method": p.extraction_method,
                    "confidence": p.confidence,
                    "chars": len(p.text),
                    "warnings": p.warnings,
                }
                for p in doc.pages
            ],
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))

    return 0


def cmd_index(args: argparse.Namespace) -> int:
    processed_path = cfg.PROCESSED_DIR / f"{args.doc_id}.json"
    if not processed_path.exists():
        logger.error("Processed document not found: %s (run ingest first)", processed_path)
        return 1
    try:
        stats = index_from_processed_json(processed_path, rebuild_bm25=args.rebuild_bm25)
    except IndexingError as exc:
        logger.error("%s", exc)
        return 1

    logger.info(
        "Indexed doc_id=%s | chunks=%s | backend=%s | chroma=%s | bm25=%s",
        stats["doc_id"],
        stats["chunks_indexed"],
        stats["embedding_backend"],
        stats["vector_store_count"],
        stats["bm25_docs"],
    )
    if args.verbose:
        print(json.dumps(stats, indent=2))
    return 0


def cmd_retrieve(args: argparse.Namespace) -> int:
    query = prepare_query(args.query, args.context or "")
    try:
        retriever = HybridRetriever()
        hits = retriever.retrieve(query, top_k=args.top_k)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    payload = [h.to_dict() for h in hits]
    if args.verbose:
        print(json.dumps({"query": query, "results": payload}, indent=2, ensure_ascii=False))
    else:
        for h in hits:
            preview = h.text[:120].replace("\n", " ")
            print(f"[{h.rank}] {h.chunk_id} (score={h.score:.4f}) {preview}...")
    return 0


def cmd_draft(args: argparse.Namespace) -> int:
    task = args.task or "Prepare a case fact summary and document checklist from the evidence."
    try:
        draft = run_draft_pipeline(
            doc_id=args.doc_id,
            task=task,
            extra_context=args.context or "",
            top_k=args.top_k,
            force_mock=args.mock,
        )
    except (GenerationError, FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    logger.info(
        "Draft saved: %s | mode=%s | claims=%s | grounding_passed=%s",
        draft.get("_output_path"),
        draft.get("generation_mode"),
        draft.get("grounding_stats", {}).get("verified", "?"),
        draft.get("grounding_passed"),
    )
    if args.verbose:
        draft_out = {k: v for k, v in draft.items() if not k.startswith("_")}
        print(json.dumps(draft_out, indent=2, ensure_ascii=False))
    return 0


def cmd_feedback(args: argparse.Namespace) -> int:
    task = args.task or "Prepare a case fact summary and document checklist from the evidence."
    try:
        if args.simulate:
            edited_path = simulate_operator_edit(args.doc_id)
            logger.info("Simulated operator edit: %s", edited_path)
        else:
            if not args.edited:
                logger.error("Provide --edited path or use --simulate")
                return 1
            edited_path = Path(args.edited).resolve()

        result = run_feedback_pipeline(
            doc_id=args.doc_id,
            task=task,
            edited_path=edited_path,
            use_llm_patterns=args.use_llm,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    logger.info("Feedback captured: edit_id=%s", result["edit_id"])
    if args.verbose:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Legal RAG — grounded drafting from messy legal PDFs",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ingest_parser = sub.add_parser("ingest", help="Stage 1: extract and structure a PDF")
    ingest_parser.add_argument("--pdf", required=True, help="Path to input PDF")
    ingest_parser.add_argument("--doc-id", default=None, help="Optional document ID")
    ingest_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print JSON summary to stdout"
    )
    ingest_parser.set_defaults(func=cmd_ingest)

    index_parser = sub.add_parser("index", help="Stage 2: chunk, embed, store in Chroma + BM25")
    index_parser.add_argument("--doc-id", required=True, help="Document ID from ingest")
    index_parser.add_argument(
        "--rebuild-bm25", action="store_true", help="Rebuild entire BM25 corpus from scratch"
    )
    index_parser.add_argument("-v", "--verbose", action="store_true")
    index_parser.set_defaults(func=cmd_index)

    retrieve_parser = sub.add_parser("retrieve", help="Stage 3: hybrid search (vector + BM25 + RRF)")
    retrieve_parser.add_argument("--query", required=True, help="Drafting task / question")
    retrieve_parser.add_argument("--context", default="", help="Optional extra context")
    retrieve_parser.add_argument("--top-k", type=int, default=None, help="Fused top-k (default 5)")
    retrieve_parser.add_argument("-v", "--verbose", action="store_true")
    retrieve_parser.set_defaults(func=cmd_retrieve)

    draft_parser = sub.add_parser("draft", help="Stage 4: retrieve evidence + grounded JSON draft")
    draft_parser.add_argument("--doc-id", required=True)
    draft_parser.add_argument("--task", default=None, help="Drafting instruction")
    draft_parser.add_argument("--context", default="", help="Extra retrieval context")
    draft_parser.add_argument("--top-k", type=int, default=None)
    draft_parser.add_argument(
        "--mock", action="store_true", help="Force extractive mock draft (no API key)"
    )
    draft_parser.add_argument("-v", "--verbose", action="store_true")
    draft_parser.set_defaults(func=cmd_draft)

    feedback_parser = sub.add_parser(
        "feedback", help="Stage 5: capture operator edit and learn patterns"
    )
    feedback_parser.add_argument("--doc-id", required=True)
    feedback_parser.add_argument("--task", default=None)
    feedback_parser.add_argument("--edited", default=None, help="Path to operator-edited draft JSON")
    feedback_parser.add_argument(
        "--simulate", action="store_true", help="Auto-generate a sample operator edit"
    )
    feedback_parser.add_argument(
        "--use-llm", action="store_true", help="Use OpenAI for pattern extraction (optional)"
    )
    feedback_parser.add_argument("-v", "--verbose", action="store_true")
    feedback_parser.set_defaults(func=cmd_feedback)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
