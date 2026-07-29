from __future__ import annotations

from scripts.run_ocr_preprocessing_ablation import _aggregate_setting, word_error_rate
from scripts.run_large_ocr_ablation import preprocessing_selection_score


def test_word_error_rate_handles_insert_delete_and_substitute() -> None:
    assert word_error_rate(["a", "b"], ["a", "b"]) == 0.0
    assert word_error_rate(["a", "b"], ["a"]) == 0.5
    assert word_error_rate(["a"], ["b", "c"]) == 2.0
    assert word_error_rate([], []) == 0.0


def test_successful_ablation_setting_is_explicitly_selectable() -> None:
    report = _aggregate_setting(
        "original",
        "original",
        {},
        [{
            "dataset": "funsd",
            "alignment_coverage": 0.75,
            "word_error_rate": 0.25,
            "mean_confidence": 0.9,
            "word_count": 12,
            "selected_orientation": 0.0,
        }],
    )

    assert report["status"] == "passed"
    assert report["mean_alignment_coverage"] == 0.75


def test_large_ablation_score_uses_quality_and_efficiency_terms() -> None:
    better = preprocessing_selection_score(
        detector_f1=0.7,
        coverage=0.8,
        wer=0.3,
        critical_exact=0.6,
        efficiency=0.8,
    )
    worse = preprocessing_selection_score(
        detector_f1=0.5,
        coverage=0.6,
        wer=0.5,
        critical_exact=0.4,
        efficiency=0.8,
    )
    assert 0.0 <= worse < better <= 1.0
