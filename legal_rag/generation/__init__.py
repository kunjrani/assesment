"""Stage 4: grounded draft generation."""

from legal_rag.generation.generator import GenerationError, generate_draft
from legal_rag.generation.pipeline import run_draft_pipeline

__all__ = ["generate_draft", "run_draft_pipeline", "GenerationError"]
