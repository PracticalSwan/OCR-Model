from __future__ import annotations

import pytest

from scripts.compile_layout_adaptation_trials import (
    downstream_selection_score,
    regression_gate,
)


def test_downstream_selection_score_uses_frozen_weights() -> None:
    score = downstream_selection_score(
        reference_entity_f1=1.0,
        real_ocr_entity_f1=1.0,
        canonical_evidence_f1=1.0,
        critical_field_exact_match=1.0,
        relation_f1=1.0,
        rotated_real_ocr_entity_f1=1.0,
        document_macro_f1=1.0,
    )
    assert score == pytest.approx(1.0)


def test_regression_gate_enforces_reference_and_canonical_limits() -> None:
    baseline = {
        "reference_entity_f1": 0.90,
        "canonical_evidence_f1": 0.80,
        "document_accuracy": 0.95,
        "relation_f1": 0.50,
        "end_to_end_entity_f1": 0.30,
    }
    candidate = {
        **baseline,
        "reference_entity_f1": 0.879,
        "checkpoint_reload_passed": True,
    }

    result = regression_gate(candidate, baseline)

    assert result["eligible"] is False
    assert result["reference_regression_ok"] is False
    assert "reference_entity_drop_gt_0.02" in result["regression_reason"]


def test_relation_regression_can_be_accepted_for_material_e2e_gain() -> None:
    baseline = {
        "reference_entity_f1": 0.90,
        "canonical_evidence_f1": 0.80,
        "document_accuracy": 0.95,
        "relation_f1": 0.50,
        "end_to_end_entity_f1": 0.30,
    }
    candidate = {
        **baseline,
        "relation_f1": 0.45,
        "end_to_end_entity_f1": 0.33,
        "checkpoint_reload_passed": True,
    }

    result = regression_gate(candidate, baseline)

    assert result["eligible"] is True
    assert result["relation_regression_ok_or_benefit"] is True
