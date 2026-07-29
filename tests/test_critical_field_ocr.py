from __future__ import annotations

import pytest

from src.evaluation.critical_field_ocr import evaluate_critical_fields


def test_field_aware_normalization_and_tolerance_metrics() -> None:
    reference = {
        "organization_name": "Assumption University",
        "date": "2026-07-24",
        "invoice_number": "INV-001-A",
        "total_amount": "1,234.50",
        "currency": "THB",
        "email": "Accounts@Example.com",
        "phone_number": "+66 2 123 4567",
        "address": "88 Bang Na-Trat Road Bangkok",
    }
    prediction = {
        "organization_name": "Assumption Univ.",
        "date": "24/07/2026",
        "invoice_number": "inv 001 a",
        "total_amount": "1234.49",
        "currency": "THB",
        "email": "accounts@example.com",
        "phone_number": "6621234567",
        "address": "88 Bang Na Trat Rd Bangkok",
    }

    metrics = evaluate_critical_fields(
        [reference],
        [prediction],
        amount_tolerance=0.02,
    )

    fields = metrics["fields"]
    assert fields["date"]["normalized_exact_match"] == pytest.approx(1.0)
    assert fields["invoice_number"]["alphanumeric_exact_match"] == pytest.approx(1.0)
    assert fields["total_amount"]["numeric_exact_match"] == pytest.approx(0.0)
    assert fields["total_amount"]["tolerance_match"] == pytest.approx(1.0)
    assert fields["currency"]["exact_match"] == pytest.approx(1.0)
    assert fields["email"]["normalized_exact_match"] == pytest.approx(1.0)
    assert fields["phone_number"]["normalized_exact_match"] == pytest.approx(1.0)
    assert fields["organization_name"]["token_recall"] >= 0.5
    assert fields["address"]["normalized_edit_similarity"] > 0.75


def test_missing_predictions_count_as_failures_without_division_errors() -> None:
    metrics = evaluate_critical_fields(
        [{"receipt_number": "RCPT-9", "tax": "7.00"}],
        [{}],
    )

    assert metrics["sample_count"] == 1
    assert metrics["fields"]["receipt_number"]["alphanumeric_exact_match"] == 0.0
    assert metrics["fields"]["tax"]["tolerance_match"] == 0.0
    assert metrics["aggregate"]["critical_exact_match"] == 0.0
