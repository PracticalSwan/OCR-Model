from __future__ import annotations

import pytest

from scripts.compile_ocr_upgrade_reports import (
    ERROR_CATEGORIES,
    _calibration_ece_after,
    _selected_ocr_profile,
    error_category_rows,
    render_error_analysis,
    render_model_card,
    render_upgrade_summary,
)


def _evidence() -> dict:
    locked_counts = {
        category.key: index
        for index, category in enumerate(
            (
                item
                for item in ERROR_CATEGORIES
                if item.source == "locked_ocr"
            ),
            start=1,
        )
    }
    orientation_counts = {
        category.key: index
        for index, category in enumerate(
            (
                item
                for item in ERROR_CATEGORIES
                if item.source == "orientation"
            ),
            start=1,
        )
    }
    return {
        "locked_ocr": {
            "sample_count": 100,
            "error_counts": locked_counts,
        },
        "orientation": {
            "sample_count": 20,
            "metrics": {"error_counts": orientation_counts},
        },
    }


def test_error_analysis_quantifies_every_required_category() -> None:
    evidence = _evidence()
    rows = error_category_rows(evidence)
    report = render_error_analysis(evidence)

    assert len(rows) == 26
    assert {row["category"] for row in rows} == {
        category.label for category in ERROR_CATEGORIES
    }
    for category in ERROR_CATEGORIES:
        assert category.label in report
    assert "frequency per evaluated page × downstream impact" in report
    assert "contains no private filenames" in report


def test_error_analysis_refuses_missing_executed_category() -> None:
    evidence = _evidence()
    evidence["locked_ocr"]["error_counts"].pop(
        "recognition_turkish_character_errors"
    )

    with pytest.raises(
        KeyError,
        match="recognition_turkish_character_errors",
    ):
        error_category_rows(evidence)


def test_selected_ocr_profile_uses_actual_selection_schema() -> None:
    assert (
        _selected_ocr_profile({"selected_ocr_profile": "adaptive"})
        == "adaptive"
    )


def test_calibration_summary_reads_per_task_metrics() -> None:
    report = {
        "metrics": {
            task: {"ece_after": value}
            for task, value in (
                ("entity", 0.01),
                ("canonical", 0.02),
                ("relation", 0.03),
                ("document", 0.04),
            )
        }
    }

    assert _calibration_ece_after(report) == (
        "0.0100 / 0.0200 / 0.0300 / 0.0400"
    )


def test_final_narratives_accept_executed_report_schemas() -> None:
    evidence = _evidence()
    evidence.update(
        {
            "detector_trials": [{
                "selected": "True",
                "model_name": "detector",
                "detector_sha256": "a" * 64,
            }],
            "recognizer_trials": [{
                "selected": "True",
                "track": "general",
                "model_name": "general",
                "recognizer_sha256": "b" * 64,
            }],
            "thai_trials": [{
                "selected": "True",
                "model_name": "thai",
                "recognizer_sha256": "c" * 64,
            }],
            "layout_adaptation_trials": [{
                "selected": "True",
                "trial_id": "continued_a",
                "checkpoint_sha256": "d" * 64,
            }],
            "layout_stream_trials": [{
                "selected": "True",
                "manifest_path": "data/metadata/final.csv",
                "train_example_count": "100",
                "dev_select_example_count": "20",
            }],
            "ocr_selection": {
                "selected_configuration": "A",
                "selected_ocr_profile": "original",
                "configuration_sha256": "e" * 64,
            },
            "benchmark": {
                "sample_count": 20,
                "counts_by_dataset": {"funsd": 20},
            },
            "registry": {
                "defaults": {
                    "detector": "original",
                    "general_recognizer": "original",
                    "thai_recognizer": "custom",
                }
            },
            "preprocessing": {"chosen_profile": "original"},
            "adaptive_rendering": {"rerender_trigger_rate": 0.25},
            "tiling": {"metric_deltas": {"polygon_f1": 0.01}},
            "crop_padding": {"selected_default_profile": "B"},
            "recognition_retry": {
                "metric_deltas": {"recognized_text_coverage": 0.01}
            },
            "calibration": {
                "metrics": {
                    task: {"ece_after": 0.01}
                    for task in (
                        "entity",
                        "canonical",
                        "relation",
                        "document",
                    )
                }
            },
            "baseline": {
                "baseline_ocr": {
                    "polygon_f1": 0.2,
                    "recognized_text_coverage": 0.3,
                    "wer": 0.7,
                },
                "baseline_end_to_end": {
                    "entity_f1": 0.1,
                    "relation_f1": 0.05,
                    "canonical_field_accuracy": 0.2,
                },
                "unseen_coru": {"qa_answer_text_recall": 0.5},
            },
            "unseen_coru": {
                "sample_pages": 100,
                "successful_pages": 100,
                "failed_pages": 0,
                "qa_answer_text_recall": 0.6,
            },
            "private": {
                "successful_documents": 2,
                "failed_documents": 0,
                "gmail_fit_rows": 0,
            },
        }
    )
    evidence["locked_ocr"]["failure_count"] = 0
    evidence["locked_ocr"]["metrics"] = {
        "polygon_f1": 0.4,
        "recognized_text_coverage": 0.5,
        "wer": 0.6,
        "critical_field_exact_match": 0.7,
    }
    evidence["locked_end_to_end"] = {
        "metrics": {
            "entity_f1": 0.3,
            "relation_f1": 0.1,
            "canonical_field_accuracy": 0.4,
        }
    }
    evidence["orientation"]["metrics"].update(
        {"orientation_selection_accuracy": 0.9}
    )

    card = render_model_card(evidence)
    summary = render_upgrade_summary(evidence)

    assert "A (original)" in card
    assert "0.0100 / 0.0100 / 0.0100 / 0.0100" in card
    assert "continued_a" in summary
    assert "aggregate-only" in summary
