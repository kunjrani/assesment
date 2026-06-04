"""Stage 5: operator feedback and few-shot learning."""

from legal_rag.feedback.few_shot_store import FewShotStore, apply_patterns_to_mock_draft
from legal_rag.feedback.pipeline import run_feedback_pipeline, simulate_operator_edit

__all__ = [
    "FewShotStore",
    "apply_patterns_to_mock_draft",
    "run_feedback_pipeline",
    "simulate_operator_edit",
]
