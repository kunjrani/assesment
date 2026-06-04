# Legal RAG — Ambitio Assessment

Grounded legal-style drafting from messy PDFs: ingest → retrieve → generate → learn from edits.

## Project structure

```
legal_rag/
├── main.py                  # CLI (ingest implemented)
├── config.py                # paths, thresholds, model names
├── ingestion/
│   ├── extractor.py         # pdfplumber → pymupdf → OCR fallback
│   ├── field_parser.py      # regex structured fields
│   └── models.py            # ProcessedDocument contract
├── retrieval/               # Stage 2–3 (next)
├── generation/              # Stage 4 (next)
├── feedback/                # Stage 5 (next)
├── scripts/
│   └── generate_sample_pdf.py
├── samples/
├── data/processed/          # JSON output from ingest
└── requirements.txt
```

## Setup

```powershell
cd C:\Users\91635\OneDrive\Desktop\Assignment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r legal_rag\requirements.txt
```

### System dependencies (OCR path)

| Tool | Purpose | Windows |
|------|---------|---------|
| [Tesseract](https://github.com/tesseract-ocr/tesseract) | Scanned page OCR | Install, set `TESSERACT_CMD` in `.env` |
| [Poppler](https://github.com/osber/poppler-windows) | PDF → image for OCR | Set `POPPLER_PATH` to `bin` folder |

Digital PDFs work **without** Tesseract/Poppler (native pdfplumber path).

## Stage 1 — Ingest

```powershell
# Generate synthetic legal PDF
python legal_rag\scripts\generate_sample_pdf.py

# Ingest (from Assignment folder)
python -m legal_rag.main ingest --pdf legal_rag\samples\sample_input.pdf -v
```

Output: `legal_rag/data/processed/sample_input.json`

## API key — do you need a paid OpenAI account?

| Stage | API key required? |
|-------|-------------------|
| Stage 1 ingest | **No** |
| Stage 2 index | **No** (default: Chroma local MiniLM embeddings) |
| Stage 3 retrieve | **No** |
| Stage 4 draft (next) | **Yes** for best results — or document mock output |

OpenAI does **not** offer a fully free API tier; new accounts sometimes get **small trial credits**. For this assignment, set `EMBEDDING_BACKEND=local` or leave `OPENAI_API_KEY` empty — indexing uses **free local embeddings** automatically.

## Stage 2 — Index

```powershell
python -m legal_rag.main index --doc-id sample_input -v
```

## How the project runs WITHOUT an OpenAI key

Your `.env` can leave `OPENAI_API_KEY` empty. With defaults (`EMBEDDING_BACKEND=auto`, `LLM_BACKEND=auto`):

| Step | What runs instead |
|------|-------------------|
| **ingest** | pdfplumber / OCR — local only |
| **index** | Chroma **local MiniLM** embeddings (free, downloads once) |
| **retrieve** | BM25 + vector + RRF — local only |
| **draft** | **Mock extractive** draft: facts pulled from retrieved chunks + grounding verifier |
| **feedback** | Rule-based pattern extraction — local only |

So the **full pipeline works for the assessment** without payment. Quality of *wording* is better with an API key, but **grounding, retrieval, and the edit loop** are fully demonstrable without one.

### Does OpenAI offer a “free” API key?

- **ChatGPT free** (web/app) is **not** the same as the **API**. They are separate products.
- The **API** is billed per usage. New accounts may get **limited trial credits** (varies by region/time) — not a permanent free tier.
- If you have trial credits, set `OPENAI_API_KEY` and remove `--mock` on `draft` for nicer prose.
- If you have **no credits**, keep the key unset — your project still runs end-to-end.

## Stage 3 — Retrieve (hybrid + RRF)

```powershell
python -m legal_rag.main retrieve --query "Summarize case facts and hearing date" -v
```

### Pipeline (Stage 1)

1. **Native extract** — `pdfplumber`, fallback `pymupdf`
2. **Quality gate** — if text too short or garbled → OCR page via `pdf2image` + `pytesseract`
3. **Normalize** — unicode cleanup, de-hyphenation, whitespace
4. **Fields** — case numbers, dates, parties, jurisdiction, document type (regex)
5. **Emit** — `ProcessedDocument` JSON ready for chunking

### Inspiration

- [jsvine/pdfplumber](https://github.com/jsvine/pdfplumber) — native text + tables
- [bibhu342/PDF-Parser-Pro](https://github.com/bibhu342/PDF-Parser-Pro) — OCR fallback layout
- [pdfplumber #717](https://github.com/jsvine/pdfplumber/discussions/717) — scanned PDF detection

## Stage 4 — Draft

```powershell
python -m legal_rag.main draft --doc-id sample_input --mock -v
```

## Stage 5 — Feedback loop

```powershell
# After draft: simulate an operator edit, learn patterns, store few-shot
python -m legal_rag.main feedback --doc-id sample_input --simulate -v

# Re-run draft — prompt / mock output uses learned patterns
python -m legal_rag.main draft --doc-id sample_input --mock -v
```

Real edit: save your JSON as `data/drafts/sample_input_draft_edited.json`, then:

```powershell
python -m legal_rag.main feedback --doc-id sample_input --edited legal_rag\data\drafts\sample_input_draft_edited.json -v
```
