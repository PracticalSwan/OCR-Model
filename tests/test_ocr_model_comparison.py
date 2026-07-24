from __future__ import annotations

import pytest

from scripts.evaluate_ocr_model_comparison import (
    ocr_model_selection_score,
    select_simplest_material_configuration,
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
