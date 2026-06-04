Submission instructions and run notes

Summary
-------
The pipeline ran correctly on this machine. Stages completed locally:
- Stage 1: ingest (generated `legal_rag/data/processed/sample_input.json`)
- Stage 2: index (Chroma local ONNX MiniLM embeddings; `legal_rag/data/bm25.pkl` created)
- Stage 3: retrieve (fused RRF results returned)
- Stage 4: draft (mock extractive draft saved to `legal_rag/data/drafts/sample_input_draft.json`)
- Stage 5: feedback (simulated edit stored under `legal_rag/data/edits/` and applied on re-draft)

Important note: this demo was run WITHOUT an OpenAI API key — it used local embeddings and the mock LLM (`EMBEDDING_BACKEND=local`, `LLM_BACKEND=mock`). The project is designed to run end-to-end without paid API credentials.

Reproduce the run (from the Assignment folder)
--------------------------------------------
PowerShell (recommended):

```powershell
cd C:\Users\91635\OneDrive\Desktop\Assignment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install -U pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r legal_rag\requirements.txt

# Optional: run full script (does all stages)
powershell -ExecutionPolicy Bypass -File legal_rag\scripts\run_e2e_demo.ps1

# Or run stages manually:
.\.venv\Scripts\python.exe -m legal_rag.main ingest --pdf legal_rag\samples\sample_input.pdf -v
.\.venv\Scripts\python.exe -m legal_rag.main index --doc-id sample_input -v
.\.venv\Scripts\python.exe -m legal_rag.main retrieve --query "case facts parties hearing date" -v
.\.venv\Scripts\python.exe -m legal_rag.main draft --doc-id sample_input --mock -v
.\.venv\Scripts\python.exe -m legal_rag.main feedback --doc-id sample_input --simulate -v
.\.venv\Scripts\python.exe -m legal_rag.main draft --doc-id sample_input --mock -v
```

Files to include in your submission (examples produced above)
----------------------------------------------------------
- legal_rag/samples/sample_input.pdf
- legal_rag/data/processed/sample_input.json
- legal_rag/data/drafts/sample_input_draft.json
- legal_rag/data/drafts/sample_input_draft_edited.json
- legal_rag/data/edits/<edit_id>.json
- legal_rag/data/chroma/ (vector store files)
- legal_rag/data/bm25.pkl

Add screenshots
---------------
Save screenshots showing the terminal outputs and the JSON files under `legal_rag/screenshots/` (create the folder). Example filenames:
- legal_rag/screenshots/retrieve.png
- legal_rag/screenshots/draft_before_after.png

Git / push instructions
-----------------------
Set your git user (important if you're currently using someone else's credentials):

```powershell
git config user.name "Your Name"
git config user.email "you@yourdomain.com"
```

Create a repo on GitHub (web) or using the `gh` CLI, then add the remote and push:

```powershell
# replace <repo> with your repo name
git init
git add .
git commit -m "Add legal_rag demo outputs and submission README"
git remote add origin https://github.com/kunjrani/<repo>.git
git branch -M main
git push -u origin main

# If you have `gh` installed, you can create+push in one step:
gh repo create kunjrani/<repo> --public --source=. --push
```

If you prefer, I can create the `SUBMISSION_README.md` (this file) in the repo and commit it locally for you — I will not push to GitHub without your confirmation of the target repository URL and credentials.

If you want, I can also prepare a small ZIP of the required output files for upload to the submission portal.
