# End-to-end demo - run from Assignment folder:
#   powershell -ExecutionPolicy Bypass -File legal_rag\scripts\run_e2e_demo.ps1

$ErrorActionPreference = "Stop"

# Run from Assignment folder (parent of legal_rag)
if (-not (Test-Path "legal_rag\main.py")) {
    $assignmentRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
    Set-Location $assignmentRoot
}
if (-not (Test-Path "legal_rag\main.py")) {
    throw "Run this script from the Assignment folder (contains legal_rag\main.py)"
}

$Python = "python"
if (Test-Path ".\.venv\Scripts\python.exe") {
    $Python = ".\.venv\Scripts\python.exe"
}

# Prefer local/free backends when no API key in environment
if (-not $env:OPENAI_API_KEY) {
    $env:EMBEDDING_BACKEND = "local"
    $env:LLM_BACKEND = "mock"
}

function Invoke-Step {
    param(
        [string]$Label,
        [scriptblock]$Action
    )
    Write-Host ""
    Write-Host $Label -ForegroundColor Cyan
    & $Action
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed (exit $LASTEXITCODE): $Label"
    }
}

Write-Host "Checking dependencies..." -ForegroundColor Yellow
& $Python -m pip install -q -r legal_rag\requirements.txt
if ($LASTEXITCODE -ne 0) {
    throw "pip install failed. Run: pip install -r legal_rag\requirements.txt"
}

Invoke-Step "=== [0] Generate sample PDF ===" {
    & $Python legal_rag\scripts\generate_sample_pdf.py
}

Invoke-Step "=== [1] INGEST - Stage 1 ===" {
    & $Python -m legal_rag.main ingest --pdf legal_rag\samples\sample_input.pdf -v
}

Invoke-Step "=== [2] INDEX - Stage 2 ===" {
    & $Python -m legal_rag.main index --doc-id sample_input -v
}

Invoke-Step "=== [3] RETRIEVE - Stage 3 ===" {
    & $Python -m legal_rag.main retrieve --query "case facts parties hearing date case number" -v
}

Invoke-Step "=== [4] DRAFT 1st pass - Stage 4 ===" {
    & $Python -m legal_rag.main draft --doc-id sample_input --mock -v
}

Invoke-Step "=== [5] FEEDBACK - Stage 5 ===" {
    & $Python -m legal_rag.main feedback --doc-id sample_input --simulate -v
}

Invoke-Step "=== [6] DRAFT 2nd pass after learning - Stage 4 and 5 ===" {
    & $Python -m legal_rag.main draft --doc-id sample_input --mock -v
}

Write-Host ""
Write-Host "=== OUTPUT FILES TO SUBMIT / SCREENSHOT ===" -ForegroundColor Green
$outputs = @(
    "legal_rag\samples\sample_input.pdf",
    "legal_rag\data\processed\sample_input.json",
    "legal_rag\data\drafts\sample_input_draft.json",
    "legal_rag\data\drafts\sample_input_draft_edited.json",
    "legal_rag\data\edits\",
    "legal_rag\samples\sample_output.json"
)
foreach ($path in $outputs) {
    Write-Host ("  " + $path)
}

Write-Host ""
Write-Host "Done. Compare 1st vs 2nd draft summary for feedback improvement." -ForegroundColor Green
Write-Host ""
