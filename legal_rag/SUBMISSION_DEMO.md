# End-to-end demo & what to submit

## What PDF / data to use

| File | Use for |
|------|---------|
| **`samples/sample_input.pdf`** | **Primary demo** — synthetic notice of hearing (case no., parties, dates). Regenerate anytime: `python legal_rag\scripts\generate_sample_pdf.py` |
| **Your own PDF** (optional) | Second demo — any legal-style PDF (notice, memo, complaint). Use: `ingest --pdf path\to\file.pdf --doc-id my_doc` then same `index` / `draft` with that `doc-id` |

**For submission:** include `sample_input.pdf` + at least one **output JSON per stage** (see below). Optional: add a second real/scanned PDF to show OCR (needs Tesseract + Poppler).

---

## One-command full pipeline (PowerShell)

From `Assignment` folder (parent of `legal_rag`):

```powershell
cd C:\Users\91635\OneDrive\Desktop\Assignment
.\.venv\Scripts\Activate.ps1
powershell -ExecutionPolicy Bypass -File legal_rag\scripts\run_e2e_demo.ps1
```

---

## Step-by-step (manual — good for screenshots)

### 0. Setup (once)

```powershell
cd C:\Users\91635\OneDrive\Desktop\Assignment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r legal_rag\requirements.txt
```

### 1. Stage 1 — Ingest

```powershell
python legal_rag\scripts\generate_sample_pdf.py
python -m legal_rag.main ingest --pdf legal_rag\samples\sample_input.pdf -v
```

**Check:** `legal_rag\data\processed\sample_input.json`  
**Show reviewers:** `fields.case_numbers`, `fields.parties`, `fields.dates`, `pages[].extraction_method`

### 2. Stage 2 — Index

```powershell
python -m legal_rag.main index --doc-id sample_input -v
```

**Check:** terminal shows `chunks_indexed`, `embedding_backend` (e.g. `chroma-default:onnx-minilm`)  
**Show reviewers:** `data\chroma\` folder exists, `data\bm25.pkl` exists

### 3. Stage 3 — Retrieve

```powershell
python -m legal_rag.main retrieve --query "case facts parties hearing date" -v
```

**Check:** JSON with `results[]` each having `chunk_id`, `source_id`, `score`, `text`  
**Show reviewers:** top chunks mention case number / hearing / parties

### 4. Stage 4 — Draft (first pass)

```powershell
python -m legal_rag.main draft --doc-id sample_input --mock -v
```

**Check:** `legal_rag\data\drafts\sample_input_draft.json`  
**Show reviewers:** `claims[].source_ids`, `grounding_passed`, `grounding_stats`

### 5. Stage 5 — Feedback

```powershell
python -m legal_rag.main feedback --doc-id sample_input --simulate -v
```

**Check:** `data\edits\*.json` and `data\feedback_chroma\`  
**Show reviewers:** `patterns.patterns` list (learned rules)

### 6. Stage 4 again — Draft (after learning)

```powershell
python -m legal_rag.main draft --doc-id sample_input --mock -v
```

**Check:** second draft may have `few_shot_applied: true` and different `summary` vs first draft  
**Show reviewers:** before/after summary = improvement loop works

---

## Files to zip for submission

```
legal_rag/
  README.md
  SUBMISSION_DEMO.md          (this file)
  requirements.txt
  source code (ingestion/, retrieval/, generation/, feedback/)
  samples/
    sample_input.pdf          ← input
    sample_output.json        ← example expected shape
  data/                       ← optional but proves pipeline ran
    processed/sample_input.json
    drafts/sample_input_draft.json
    drafts/sample_input_draft_edited.json
    edits/<edit_id>.json
```

**Tip:** Run the e2e script once before zipping so `data/` contains real outputs.

---

## Quick verification checklist

- [ ] Ingest: `sample_input.json` has non-empty `full_text` and `fields.case_numbers`
- [ ] Index: no error; `chroma` + `bm25.pkl` created
- [ ] Retrieve: at least 1 result with relevant text
- [ ] Draft: `claims` non-empty; each claim has `source_ids`
- [ ] Feedback: `patterns` JSON written under `data/edits/`
- [ ] Second draft: differs from first OR `few_shot_applied: true`

---

## Optional: second document

```powershell
python -m legal_rag.main ingest --pdf "C:\path\to\your_notice.pdf" --doc-id my_notice -v
python -m legal_rag.main index --doc-id my_notice -v
python -m legal_rag.main draft --doc-id my_notice --mock -v
```

Use a **digital** PDF first (no OCR setup). For a **scanned** PDF, install Tesseract + Poppler and set paths in `.env` (see `.env.example`).
