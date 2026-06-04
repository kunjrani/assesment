"""Central configuration for paths, thresholds, and model names."""

from __future__ import annotations

import os
from pathlib import Path

# Project roots
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CHROMA_DIR = DATA_DIR / "chroma"
BM25_PATH = DATA_DIR / "bm25.pkl"
EDITS_DIR = DATA_DIR / "edits"
FEEDBACK_CHROMA_DIR = DATA_DIR / "feedback_chroma"
SAMPLES_DIR = PROJECT_ROOT / "samples"

# Ensure runtime directories exist
for _dir in (DATA_DIR, PROCESSED_DIR, CHROMA_DIR, EDITS_DIR, FEEDBACK_CHROMA_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# --- Stage 1: Ingestion ---
MIN_PAGE_TEXT_CHARS = int(os.getenv("MIN_PAGE_TEXT_CHARS", "40"))
GARBLED_RATIO_THRESHOLD = float(os.getenv("GARBLED_RATIO_THRESHOLD", "0.35"))
OCR_DPI = int(os.getenv("OCR_DPI", "200"))
OCR_LANG = os.getenv("OCR_LANG", "eng")

# Optional: Windows Tesseract path (e.g. C:\Program Files\Tesseract-OCR\tesseract.exe)
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "")

# Poppler path for pdf2image on Windows (folder containing pdftoppm.exe)
POPPLER_PATH = os.getenv("POPPLER_PATH", "") or None

# --- Stage 2–4 (used in later stages) ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))
# auto | openai | local — local works without any paid API key
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "auto")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
# auto | openai | mock — mock works without any API key
LLM_BACKEND = os.getenv("LLM_BACKEND", "auto")
VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "20"))
BM25_TOP_K = int(os.getenv("BM25_TOP_K", "20"))
RRF_K = int(os.getenv("RRF_K", "60"))
FUSION_TOP_K = int(os.getenv("FUSION_TOP_K", "5"))
FEEDBACK_TOP_K = int(os.getenv("FEEDBACK_TOP_K", "3"))

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
