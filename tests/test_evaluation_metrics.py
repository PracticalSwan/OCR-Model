import pytest

from src.evaluation.metrics import (
    edit_distance,
    extraction_metrics,
    ocr_text_metrics,
    text_detection_metrics,
)
from scripts.evaluate_end_to_end_angles import (
    _aggregate_angle,
    _with_rotation_retention,
)


def test_edit_distance_and_ocr_metrics() -> None:
    assert edit_distance("kitten", "sitting") == 3
    metrics = ocr_text_metrics("Invoice Total", "invoice total")
    assert metrics["cer"] == 0.0
    assert metrics["wer"] == 0.0


def test_entity_relation_and_field_metrics() -> None:
    entities = [
        {"id": "k", "label": "KEY", "text": "Total"},
        {"id": "v", "label": "VALUE", "text": "10.00"},
    ]
    annotation = {
        "entities": entities,
        "relations": [{"type": "KEY_VALUE", "source_id": "k", "target_id": "v"}],
        "canonical_fields": {"total_amount": {"value": "10.00"}},
    }
    page = {
        "entities": entities,
        "key_value_pairs": [{"type": "KEY_VALUE", "source_id": "k", "target_id": "v"}],
    }
    metrics = extraction_metrics(
        annotation, page, {"total_amount": {"value": "10.00"}}
    )
    assert metrics["entity"]["f1"] == 1.0
    assert metrics["relation"]["f1"] == 1.0
    assert metrics["canonical_fields"]["accuracy"] == 1.0


def test_text_detection_polygon_matching_is_one_to_one() -> None:
    reference = [
        {"polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]},
        {"polygon": [[20, 0], [30, 0], [30, 10], [20, 10]]},
    ]
    prediction = [
        {"polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]},
        {"polygon": [[1, 1], [9, 1], [9, 9], [1, 9]]},
        {"polygon": [[20, 0], [30, 0], [30, 10], [20, 10]]},
    ]
    metrics = text_detection_metrics(reference, prediction)
    assert metrics["true_positive"] == 2
    assert metrics["precision"] == 2 / 3
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 0.8


def test_text_detection_without_reference_is_unavailable() -> None:
    metrics = text_detection_metrics([], [{"bbox": [0, 0, 10, 10]}])
    assert metrics["reference_available"] is False
    assert metrics["precision"] is None
    assert metrics["recall"] is None
    assert metrics["f1"] is None


def test_end_to_end_angle_metrics_remain_separate_by_dataset() -> None:
    def row(dataset: str, coverage: float) -> dict:
        return {
            "dataset": dataset, "recognized_text_coverage": coverage,
            "wer": 1.0 - coverage, "detection_f1": coverage,
            "entity_f1": coverage, "entity_expected": 1,
            "relation_f1": coverage, "relation_expected": 1,
            "field_correct": int(coverage == 1.0), "field_applicable": 1,
            "nonempty": True, "route": "general", "table_count": 0,
            "reference_fields": {"total_amount": "10.00"},
            "predicted_fields": {
                "total_amount": "10.00" if coverage == 1.0 else "11.00"
            },
            "orientation_within_tolerance": coverage == 1.0,
            "orientation_error_degrees": 0.0 if coverage == 1.0 else 4.0,
            "duration_seconds": 1.0 if coverage == 1.0 else 3.0,
        }

    metrics = _aggregate_angle(37, [row("fatura", 1.0), row("funsd", 0.5)])

    assert metrics["angle"] == 37
    assert metrics["recognized_text_coverage"] == 0.75
    assert metrics["canonical_field_accuracy"] == 0.5
    assert metrics["critical_field_exact_match"] == 0.5
    assert metrics["orientation_selection_accuracy"] == 0.5
    assert metrics["mean_orientation_error_degrees"] == 2.0
    assert metrics["time_per_page_seconds"] == 2.0
    assert metrics["by_dataset"]["fatura"]["recognized_text_coverage"] == 1.0
    assert metrics["by_dataset"]["funsd"]["recognized_text_coverage"] == 0.5


def test_end_to_end_angle_rotation_retention_uses_upright_denominators() -> None:
    rows = [
        {
            "angle": 0,
            "recognized_text_coverage": 0.8,
            "entity_f1": 0.4,
            "canonical_field_accuracy": 0.5,
            "critical_field_exact_match": 0.25,
        },
        {
            "angle": 37,
            "recognized_text_coverage": 0.6,
            "entity_f1": 0.2,
            "canonical_field_accuracy": 0.4,
            "critical_field_exact_match": 0.125,
        },
    ]

    retained = _with_rotation_retention(rows)

    assert retained[0]["rotation_retention_vs_upright"] == {
        "coverage": 1.0,
        "entity_f1": 1.0,
        "canonical_accuracy": 1.0,
        "critical_field_exact_match": 1.0,
    }
    assert retained[1]["rotation_retention_vs_upright"] == pytest.approx({
        "coverage": 0.75,
        "entity_f1": 0.5,
        "canonical_accuracy": 0.8,
        "critical_field_exact_match": 0.5,
    })
    assert retained[1]["extraction_retention_vs_upright"] == 0.5
