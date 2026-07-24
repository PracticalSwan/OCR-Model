from __future__ import annotations

from scripts.evaluate_ocr_component_ablations import (
    _balanced_rows,
    _metric_deltas,
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
        "recognized_text_coverage": 0.2,
        "cer": 0.8,
        "wer": 0.9,
        "critical_field_exact_match": 0.1,
        "end_to_end_entity_f1": 0.1,
        "end_to_end_canonical_accuracy": 0.1,
        "end_to_end_relation_f1": 0.1,
        "time_per_page_seconds": 1.0,
        "page_failure_rate": 0.0,
    }
    candidate = dict(fields)
    candidate["polygon_f1"] = 0.3
    candidate["wer"] = 0.7
    deltas = _metric_deltas({"metrics": fields}, {"metrics": candidate})
    assert deltas["polygon_f1"] > 0
    assert deltas["wer"] < 0
