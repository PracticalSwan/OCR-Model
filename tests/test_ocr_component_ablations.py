from __future__ import annotations

from scripts.evaluate_ocr_component_ablations import (
    _balanced_rows,
    _effective_definition,
    _metric_deltas,
    _select_padding_profile,
)


def test_balanced_component_sample_contains_each_public_dataset() -> None:
    rows = [
        {"page_id": f"{dataset}-{index}", "dataset": dataset}
        for dataset, count in (("funsd", 20), ("sroie", 30), ("fatura", 50))
        for index in range(count)
    ]
    selected = _balanced_rows(rows, 25)
    assert len(selected) == 25
    assert {row["dataset"] for row in selected} == {"funsd", "sroie", "fatura"}


def test_component_deltas_preserve_direction() -> None:
    fields = {
        "polygon_f1": 0.1,
        "small_text_recall": 0.1,
        "critical_region_recall": 0.1,
        "recognized_text_coverage": 0.2,
        "cer": 0.8,
        "wer": 0.9,
        "critical_field_exact_match": 0.1,
        "end_to_end_entity_f1": 0.1,
        "end_to_end_canonical_accuracy": 0.1,
        "end_to_end_relation_f1": 0.1,
        "time_per_page_seconds": 1.0,
        "page_failure_rate": 0.0,
        "tiling_duplicate_rate": 0.0,
    }
    candidate = dict(fields)
    candidate["polygon_f1"] = 0.3
    candidate["wer"] = 0.7
    deltas = _metric_deltas({"metrics": fields}, {"metrics": candidate})
    assert deltas["polygon_f1"] > 0
    assert deltas["wer"] < 0


def test_effective_definition_keeps_original_models_during_adaptive_ablation() -> None:
    definition = _effective_definition(
        "tiling_only",
        {
            "adaptive_preprocessing": False,
            "tiling": True,
            "recognition_retries": False,
        },
        base_profile="original",
        cfg={"ocr": {"recognition_retries": {"padding_profile": "B"}}},
        completed_results={},
    )
    assert definition["ocr_profile"] == "adaptive"
    assert definition["detector_choice"] == "original"
    assert definition["general_recognizer_choice"] == "original"
    assert definition["thai_recognizer_choice"] == "original"


def test_padding_selection_uses_measured_scores_and_deterministic_ties() -> None:
    base_metrics = {
        "critical_field_exact_match": 0.5,
        "recognized_text_coverage": 0.6,
        "wer": 0.4,
        "end_to_end_entity_f1": 0.3,
        "end_to_end_canonical_accuracy": 0.4,
        "page_failure_rate": 0.0,
        "time_per_page_seconds": 1.0,
    }
    results = {
        "A": {"metrics": dict(base_metrics)},
        "B": {
            "metrics": {
                **base_metrics,
                "critical_field_exact_match": 0.7,
            }
        },
        "C": {"metrics": dict(base_metrics)},
    }
    assert _select_padding_profile(results) == "B"
    assert _select_padding_profile(
        {"A": results["A"], "C": results["C"]}
    ) == "A"
