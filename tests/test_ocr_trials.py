from __future__ import annotations

from pathlib import Path

import pytest

from src.ocr.trials import (
    aggregate_detector_results,
    aggregate_recognition_results,
    detector_acceptance,
    detector_page_result,
    recognizer_acceptance,
    recognizer_selection_score,
    recognition_sample_result,
    sha256_tree,
)


def _region(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    token_id: str,
) -> dict[str, object]:
    return {
        "polygon": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        "token_ids": [token_id],
    }


def test_detector_page_result_tracks_size_and_critical_recall() -> None:
    references = [
        _region(0, 0, 20, 5, token_id="small"),
        _region(0, 20, 50, 40, token_id="critical"),
        _region(0, 60, 80, 95, token_id="large"),
    ]
    predictions = [
        {"polygon": [[0, 0], [20, 0], [20, 5], [0, 5]]},
        {"polygon": [[0, 20], [50, 20], [50, 40], [0, 40]]},
        {"polygon": [[90, 90], [99, 90], [99, 99], [90, 99]]},
    ]

    result = detector_page_result(
        references,
        predictions,
        page_width=100,
        page_height=100,
        critical_token_ids={"critical"},
    )

    assert result["true_positive"] == 2
    assert result["expected"] == 3
    assert result["predicted"] == 3
    assert result["small_expected"] == 1
    assert result["small_matched"] == 1
    assert result["medium_expected"] == 1
    assert result["medium_matched"] == 1
    assert result["large_expected"] == 1
    assert result["large_matched"] == 0
    assert result["critical_expected"] == 1
    assert result["critical_matched"] == 1


def test_detector_aggregation_and_acceptance_are_predeclared() -> None:
    aggregate = aggregate_detector_results(
        [
            {
                "true_positive": 6,
                "expected": 10,
                "predicted": 8,
                "small_expected": 4,
                "small_matched": 2,
                "medium_expected": 4,
                "medium_matched": 3,
                "large_expected": 2,
                "large_matched": 1,
                "critical_expected": 3,
                "critical_matched": 2,
                "duration_seconds": 0.4,
                "failed": False,
            },
            {
                "true_positive": 3,
                "expected": 5,
                "predicted": 4,
                "small_expected": 2,
                "small_matched": 1,
                "medium_expected": 2,
                "medium_matched": 1,
                "large_expected": 1,
                "large_matched": 1,
                "critical_expected": 1,
                "critical_matched": 1,
                "duration_seconds": 0.6,
                "failed": False,
            },
        ]
    )

    assert aggregate["precision"] == pytest.approx(0.75)
    assert aggregate["recall"] == pytest.approx(0.6)
    assert aggregate["f1"] == pytest.approx(2 * 0.75 * 0.6 / 1.35)
    assert aggregate["small_text_recall"] == pytest.approx(0.5)
    assert aggregate["critical_region_recall"] == pytest.approx(0.75)
    assert aggregate["time_per_page_seconds"] == pytest.approx(0.5)

    accepted = detector_acceptance(
        {"f1": 0.48, "critical_region_recall": 0.70},
        {
            "f1": 0.60,
            "critical_region_recall": 0.70,
            "failure_count": 0,
            "time_per_page_seconds": 0.7,
        },
        export_reload_passed=True,
        adapter_contract_passed=True,
        rotated_page_passed=True,
        blank_low_text_passed=True,
        private_row_count=0,
        maximum_time_per_page_seconds=2.0,
    )
    assert accepted["accepted"] is True
    assert accepted["criteria"]["f1_improvement_or_floor"] is True

    rejected = detector_acceptance(
        {"f1": 0.48, "critical_region_recall": 0.70},
        {
            "f1": 0.61,
            "critical_region_recall": 0.69,
            "failure_count": 0,
            "time_per_page_seconds": 0.7,
        },
        export_reload_passed=True,
        adapter_contract_passed=True,
        rotated_page_passed=True,
        blank_low_text_passed=True,
        private_row_count=0,
        maximum_time_per_page_seconds=2.0,
    )
    assert rejected["accepted"] is False
    assert rejected["criteria"]["critical_recall_no_regression"] is False


def test_recognition_metrics_cover_critical_and_language_slices() -> None:
    rows = [
        recognition_sample_result(
            "TOTAL 123.45 EUR",
            "TOTAL 123.45 EUR",
            confidence=0.9,
            critical_fields={"total_amount", "currency"},
        ),
        recognition_sample_result(
            "Tarih 24 Ağustos 2026",
            "Tarih 24 Agustos 2026",
            confidence=0.6,
            critical_fields={"date"},
        ),
        recognition_sample_result(
            "invoice@example.com",
            "",
            confidence=0.1,
            critical_fields={"email"},
        ),
    ]

    metrics = aggregate_recognition_results(rows)

    assert metrics["sample_count"] == 3
    assert metrics["exact_line_accuracy"] == pytest.approx(1 / 3)
    assert metrics["numeric_exact_match"] == pytest.approx(0.5)
    assert metrics["amount_exact_match"] == 1.0
    assert metrics["date_exact_match"] == 0.0
    assert metrics["currency_exact_match"] == 1.0
    assert metrics["email_exact_match"] == 0.0
    assert metrics["turkish_character_accuracy"] == 0.0
    assert 0.0 <= metrics["confidence_ece"] <= 1.0
    assert 0.0 <= metrics["confidence_brier"] <= 1.0


def test_recognizer_score_and_acceptance_require_critical_gain() -> None:
    baseline = {
        "cer": 0.50,
        "wer": 0.60,
        "amount_exact_match": 0.40,
        "identifier_exact_match": 0.30,
        "date_exact_match": 0.35,
        "currency_exact_match": 0.50,
        "numeric_exact_match": 0.45,
        "turkish_character_accuracy": 0.60,
        "english_exact_match": 0.40,
    }
    candidate = {
        "cer": 0.35,
        "wer": 0.45,
        "amount_exact_match": 0.55,
        "identifier_exact_match": 0.45,
        "date_exact_match": 0.50,
        "currency_exact_match": 0.60,
        "numeric_exact_match": 0.50,
        "turkish_character_accuracy": 0.65,
        "english_exact_match": 0.40,
        "confidence_all_finite": True,
        "time_per_sample_seconds": 0.01,
    }

    assert recognizer_selection_score(candidate) > recognizer_selection_score(baseline)
    decision = recognizer_acceptance(
        baseline,
        candidate,
        export_reload_passed=True,
        dictionary_match_passed=True,
        private_row_count=0,
        maximum_time_per_sample_seconds=0.1,
    )
    assert decision["accepted"] is True
    assert decision["critical_field_exact_gain"] >= 0.10

    candidate["amount_exact_match"] = 0.39
    decision = recognizer_acceptance(
        baseline,
        candidate,
        export_reload_passed=True,
        dictionary_match_passed=True,
        private_row_count=0,
        maximum_time_per_sample_seconds=0.1,
    )
    assert decision["accepted"] is False
    assert decision["criteria"]["numeric_and_amount_no_regression"] is False


def test_sha256_tree_is_path_and_content_bound(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "one.bin").write_bytes(b"one")
    first = sha256_tree(tmp_path)
    (tmp_path / "a" / "one.bin").write_bytes(b"two")
    second = sha256_tree(tmp_path)
    (tmp_path / "a" / "one.bin").write_bytes(b"one")
    (tmp_path / "two.bin").write_bytes(b"")
    third = sha256_tree(tmp_path)

    assert len(first) == 64
    assert first != second
    assert first != third
