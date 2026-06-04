"""Generate a synthetic legal-style PDF for local testing (no API keys required)."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as: python legal_rag/scripts/generate_sample_pdf.py
ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"


def main() -> None:
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except ImportError:
        print("Install reportlab: pip install reportlab")
        sys.exit(1)

    SAMPLES.mkdir(parents=True, exist_ok=True)
    out = SAMPLES / "sample_input.pdf"

    c = canvas.Canvas(str(out), pagesize=letter)
    width, height = letter
    y = height - 72

    lines = [
        "IN THE SUPERIOR COURT OF CALIFORNIA",
        "COUNTY OF SAN FRANCISCO",
        "",
        "Case No.: CGC-24-582910",
        "",
        "Plaintiff: Rivera Properties LLC",
        "Defendant: Bayview Tenants Association",
        "",
        "NOTICE OF HEARING",
        "",
        "Please take notice that a hearing on the motion for summary judgment",
        "is scheduled for March 15, 2025 at 9:00 a.m. in Department 302.",
        "",
        "The matter concerns alleged breach of the lease agreement dated",
        "January 12, 2023. See 42 U.S.C. Section 1983 for related federal claims.",
        "",
        "Dated: June 1, 2025",
        "Respectfully submitted,",
        "Counsel for Plaintiff",
    ]

    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, y, lines[0])
    y -= 24
    c.setFont("Helvetica", 10)
    for line in lines[1:]:
        if y < 72:
            c.showPage()
            y = height - 72
            c.setFont("Helvetica", 10)
        c.drawString(72, y, line)
        y -= 14

    c.save()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
