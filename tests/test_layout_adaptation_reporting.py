from __future__ import annotations

import pytest

from scripts.compile_layout_adaptation_trials import (
    _evaluation,
    downstream_selection_score,
    regression_gate,
    select_downstream_trial,
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


def test_evaluation_binding_refuses_wrong_layout_stream(
    tmp_path,
) -> None:
    report = tmp_path / "evaluation.json"
    report.write_text(
        (
            '{"split":"dev_select","rotation_angle":0.0,'
            '"token_sources":["ground_truth"]}'
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="token sources"):
        _evaluation(
            report,
            expected_rotation=0.0,
            expected_token_sources={"paddleocr"},
        )


def _selection_row(
    trial_id: str,
    *,
    score: float,
    cross_build: bool,
) -> dict:
    return {
        "trial_id": trial_id,
        "selection_score": score,
        "cross_build_comparison": cross_build,
        "reference_entity_f1": 0.90,
        "canonical_evidence_f1": 0.80,
        "document_accuracy": 0.95,
        "relation_f1": 0.50,
        "end_to_end_entity_f1": 0.30,
        "checkpoint_reload_passed": True,
    }


def test_cross_build_baseline_is_comparison_only() -> None:
    rows = [
        _selection_row("old_baseline", score=0.90, cross_build=True),
        _selection_row("continued_a", score=0.85, cross_build=False),
    ]

    selected = select_downstream_trial(
        rows,
        baseline_trial="old_baseline",
        selected_trial="continued_a",
        comparison_only_trials={"old_baseline"},
    )

    assert selected["selected"] is True
    assert selected["selection_candidate"] is True
    assert rows[0]["selected"] is False
    assert rows[0]["selection_candidate"] is False


def test_cross_build_trial_cannot_be_final_candidate() -> None:
    rows = [
        _selection_row("old_baseline", score=0.90, cross_build=True),
        _selection_row("continued_a", score=0.85, cross_build=False),
    ]

    with pytest.raises(ValueError, match="must be comparison-only"):
        select_downstream_trial(
            rows,
            baseline_trial="old_baseline",
            selected_trial="continued_a",
        )
