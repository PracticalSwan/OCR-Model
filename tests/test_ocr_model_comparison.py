from __future__ import annotations

import pytest

from scripts.evaluate_ocr_model_comparison import (
    _comparison_build_id,
    _error_categories,
    _selection_report_build_id,
    configuration_eligibility,
    ocr_model_selection_score,
    select_simplest_material_configuration,
)


def test_comparison_build_id_is_bound_to_manifest_checkpoint_and_trials() -> None:
    first = _comparison_build_id(
        manifest_sha="a" * 64,
        checkpoint_sha="b" * 64,
        configurations=["A", "F"],
    )
    repeated = _comparison_build_id(
        manifest_sha="a" * 64,
        checkpoint_sha="b" * 64,
        configurations=["A", "F"],
    )
    changed = _comparison_build_id(
        manifest_sha="a" * 64,
        checkpoint_sha="c" * 64,
        configurations=["A", "F"],
    )
    assert first == repeated
    assert first != changed


def test_selection_report_build_id_changes_with_finalized_rows() -> None:
    original = [{"configuration": "A", "detector_sha256": None}]
    finalized = [{"configuration": "A", "detector_sha256": "a" * 64}]

    assert _selection_report_build_id(original) != _selection_report_build_id(
        finalized
    )
    assert _selection_report_build_id(finalized).startswith(
        "ocr-model-selection-"
    )


def test_ocr_model_selection_score_uses_the_frozen_weights() -> None:
    score = ocr_model_selection_score(
        detection_f1=0.60,
        coverage=0.70,
        wer=0.40,
        critical_exact=0.50,
        entity_f1=0.35,
        canonical_accuracy=0.45,
        relation_f1=0.20,
        efficiency=0.80,
        failure_rate=0.05,
    )
    assert score == pytest.approx(0.555)


def test_selection_keeps_original_when_gain_is_not_material() -> None:
    rows = [
        {"configuration": "A", "selection_score": 0.50},
        {"configuration": "F", "selection_score": 0.519},
    ]
    assert select_simplest_material_configuration(rows)["configuration"] == "A"


def test_selection_prefers_simpler_configuration_within_one_point() -> None:
    rows = [
        {"configuration": "A", "selection_score": 0.50},
        {"configuration": "C", "selection_score": 0.55},
        {"configuration": "F", "selection_score": 0.559},
    ]
    assert select_simplest_material_configuration(rows)["configuration"] == "C"


def test_rejected_custom_recognizer_is_not_default_eligible() -> None:
    registry = {
        "models": {
            "custom-general": {
                "variant": "custom",
                "role": "recognizer",
                "language": "general",
                "available": True,
                "accepted": False,
            }
        }
    }
    result = configuration_eligibility(
        {"detector": "original", "general": "custom", "thai": "auto"},
        registry,
    )
    assert result["eligible_for_default"] is False
    assert result["default_ineligibility_reason"] == "general_custom_not_accepted"


def test_error_categories_are_aggregate_only_and_deterministic() -> None:
    counts = _error_categories(
        {
            "width": "800",
            "height": "600",
            "language": "tr",
            "has_table": "true",
        },
        {},
        {
            "full_text": "Invoice total 12345",
            "tables": [],
            "warnings": [],
            "ocr": {
                "mean_confidence": 0.4,
                "language_route": "general",
                "recognition_retries": {
                    "items": [{"selected_variant": "padding_B"}]
                },
            },
        },
        {"warnings": []},
        detection={
            "expected": 10,
            "predicted": 8,
            "true_positive": 6,
            "small_expected": 4,
            "small_matched": 1,
        },
        text={"character_errors": 2, "cer": 0.4},
        extraction={
            "entity": {"true_positive": 1, "expected": 2, "predicted": 2},
            "relation": {"true_positive": 0, "expected": 1},
        },
        reference_text="Invoice total 123.45!",
        reference_fields={"total_amount": "123.45"},
        predicted_fields={},
    )

    assert counts["detection_missed_small_text"] == 3
    assert counts["detection_table_miss_pages"] == 1
    assert counts["recognition_decimal_confusions"] == 1
    assert counts["recognition_crop_tight_retry_signals"] == 1
    assert counts["downstream_ocr_wrong_entity_wrong_pages"] == 1
    assert counts["downstream_relation_misses"] == 1
    assert all(isinstance(value, int) for value in counts.values())
