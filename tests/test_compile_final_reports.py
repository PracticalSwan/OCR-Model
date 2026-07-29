from __future__ import annotations

from scripts.compile_final_reports import _model_card


def test_model_card_uses_selected_model_dataset_example_count() -> None:
    artifacts = {
        "training": {
            "checkpoint": "",
            "completed_epochs": 4,
            "optimizer_steps": 12_556,
            "best_epoch": 4,
            "best_composite_score": 0.84,
            "license": "CC-BY-NC-SA-4.0",
        },
        "dataset": {
            "usable_public_fit_pages": 11_172,
            "usable_examples": 11_684,
            "gmail_fit_rows": 0,
        },
        "model_dataset": {
            "usable_example_count": 16_781,
            "gmail_fit_rows": 0,
        },
        "unseen": {
            "qa_answer_text_recall": 0.78,
            "sample_pages": 100,
        },
    }

    card = _model_card(
        artifacts,
        {
            "entity_token_f1": 0.98,
            "entity_calibrated_f1": 0.98,
            "entity_macro_f1": 0.77,
            "relation_f1": 0.57,
            "relation_calibrated_f1": 0.56,
            "canonical_evidence_token_f1": 0.98,
            "canonical_calibrated_f1": 0.99,
        },
        {"recognized_text_coverage": 0.31, "wer": 0.82},
    )

    assert "examples: 16781" in card
    assert "examples: 11684" not in card
