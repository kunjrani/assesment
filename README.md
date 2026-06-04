**Legal RAG — Submission README**

Short summary
-------------
This repository contains a local end-to-end Legal RAG pipeline that ingests legal-style PDFs, indexes evidence (local vector store + BM25), retrieves relevant chunks, generates grounded draft summaries (mock LLM when no API key), and captures feedback to improve subsequent drafts.

Architecture (short)
--------------------
- `legal_rag/ingestion`: PDF extraction (pdfplumber/pymupdf) with OCR fallback, normalization, and structured field parsing (case numbers, parties, dates, statutes).
- `legal_rag/retrieval`: chunking, local embeddings (Chroma with ONNX MiniLM), BM25 (rank-bm25), and hybrid retriever (vector + BM25 + RRF fusion).
- `legal_rag/generation`: draft pipeline producing grounded JSON drafts; supports `--mock` extractive mode when no OpenAI key.
- `legal_rag/feedback`: capture operator edits, extract patterns/rules, store few-shot examples, and apply to later drafts.
- `legal_rag/data/`: runtime stores for processed JSON, vector store, BM25, drafts, and edits.

What I ran (concise)
--------------------
1. Create venv and install deps:
   - `python -m venv .venv`
   - `.\.venv\Scripts\Activate.ps1`
   - `.\.venv\Scripts\python.exe -m pip install -r legal_rag\requirements.txt`
2. Generate sample PDF and ingest:
   - `python legal_rag\scripts\generate_sample_pdf.py`
   - `python -m legal_rag.main ingest --pdf legal_rag\samples\sample_input.pdf -v`
3. Index (chunks + embeddings + BM25):
   - `python -m legal_rag.main index --doc-id sample_input -v`
4. Retrieve:
   - `python -m legal_rag.main retrieve --query "case facts parties hearing date" -v`
5. Draft (mock mode):
   - `python -m legal_rag.main draft --doc-id sample_input --mock -v`
6. Feedback (simulate operator edit) + re-draft:
   - `python -m legal_rag.main feedback --doc-id sample_input --simulate -v`
   - `python -m legal_rag.main draft --doc-id sample_input --mock -v`

Key extracted elements (from the sample run)
------------------------------------------
- Case number: `CGC-24-582910`
- Parties: `Rivera Properties LLC` (Plaintiff) vs `Bayview Tenants Association` (Defendant)
- Hearing date: `March 15, 2025`
- Statute ref: `42 U.S.C. Section 1983`
- Support excerpt: first page/notice text used as claim evidence

Outputs to include in submission
--------------------------------
- `legal_rag/samples/sample_input.pdf`
- `legal_rag/data/processed/sample_input.json`
- `legal_rag/data/drafts/sample_input_draft.json`
- `legal_rag/data/drafts/sample_input_draft_edited.json`
- `legal_rag/data/edits/<edit_id>.json`
- `legal_rag/data/chroma/` (vector files) and `legal_rag/data/bm25.pkl`

Screenshots
-----------
Include terminal screenshots showing retrieve/draft outputs and the saved draft JSON. Put them under `legal_rag/screenshots/`.

Simple git steps (do this from repo root)
--------------------------------------
1. Ensure `.gitignore` exists (already added) so `.venv` and data are excluded.
2. Stage and commit:
   - `git add .`
   - `git commit -m "Initial submission: Legal RAG demo outputs and README"`
3. Create remote (on GitHub) and push (replace `<repo>`):
   - `git remote add origin https://github.com/kunjrani/<repo>.git`
   - `git branch -M main`
   - `git push -u origin main`

Notes
-----
- The project runs end-to-end locally with `--mock` and local embeddings; no paid OpenAI key is required for the demo outputs.
- If you want, I can run the local commit (I will not push to remote without your confirmation of the GitHub repo URL). I can also prepare a ZIP of the output files for upload.
