"""Field-aware OCR metrics for public development and locked evaluation."""
from __future__ import annotations

import re
import statistics
import unicodedata
from collections import Counter
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from typing import Any, Mapping, Sequence


FIELD_GROUPS = {
    "organization": {"organization", "organization_name"},
    "date": {"date"},
    "identifier": {"invoice_number", "receipt_number", "reference_number"},
    "amount": {"subtotal", "tax", "total_amount"},
    "currency": {"currency"},
    "email": {"email"},
    "phone": {"phone", "phone_number"},
    "address": {"address"},
}


def evaluate_critical_fields(
    references: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    *,
    amount_tolerance: float = 0.01,
) -> dict[str, Any]:
    """Evaluate matching fields, counting missing predictions as failures."""
    if len(references) != len(predictions):
        raise ValueError("references and predictions must have equal length")
    if amount_tolerance < 0:
        raise ValueError("amount tolerance must be non-negative")
    accumulators: dict[str, dict[str, list[float]]] = {}
    exact_values: list[float] = []
    for reference, prediction in zip(references, predictions):
        for field, raw_reference in reference.items():
            if raw_reference is None or str(raw_reference).strip() == "":
                continue
            raw_prediction = prediction.get(field)
            metrics = _field_metrics(
                field,
                str(raw_reference),
                "" if raw_prediction is None else str(raw_prediction),
                amount_tolerance=amount_tolerance,
            )
            target = accumulators.setdefault(field, {})
            for name, value in metrics.items():
                target.setdefault(name, []).append(float(value))
            exact_values.append(float(_primary_exact(field, metrics)))
    fields = {
        field: {
            name: statistics.fmean(values) if values else 0.0
            for name, values in metrics.items()
        }
        for field, metrics in sorted(accumulators.items())
    }
    return {
        "sample_count": len(references),
        "evaluated_field_count": len(exact_values),
        "fields": fields,
        "aggregate": {
            "critical_exact_match": (
                statistics.fmean(exact_values) if exact_values else 0.0
            )
        },
    }


def _field_metrics(
    field: str,
    reference: str,
    prediction: str,
    *,
    amount_tolerance: float,
) -> dict[str, float]:
    group = _group(field)
    if group == "date":
        return {
            "normalized_exact_match": float(
                _normalize_date(reference) == _normalize_date(prediction)
                and bool(_normalize_date(reference))
            )
        }
    if group == "amount":
        reference_number = _decimal(reference)
        prediction_number = _decimal(prediction)
        numeric_exact = (
            reference_number is not None
            and prediction_number is not None
            and reference_number == prediction_number
        )
        tolerance_match = (
            reference_number is not None
            and prediction_number is not None
            and abs(reference_number - prediction_number) <= Decimal(str(amount_tolerance))
        )
        return {
            "numeric_exact_match": float(numeric_exact),
            "tolerance_match": float(tolerance_match),
            "decimal_separator_correctness": float(
                _decimal_separator(reference) == _decimal_separator(prediction)
                and _decimal_separator(reference) is not None
            ),
        }
    if group == "identifier":
        return {
            "case_insensitive_exact_match": float(
                _normalize_space(reference).casefold()
                == _normalize_space(prediction).casefold()
            ),
            "alphanumeric_exact_match": float(
                _alphanumeric(reference) == _alphanumeric(prediction)
                and bool(_alphanumeric(reference))
            ),
        }
    if group == "currency":
        return {
            "exact_match": float(
                _normalize_currency(reference) == _normalize_currency(prediction)
                and bool(_normalize_currency(reference))
            )
        }
    if group == "email":
        return {
            "normalized_exact_match": float(
                reference.strip().casefold() == prediction.strip().casefold()
            )
        }
    if group == "phone":
        return {
            "normalized_exact_match": float(
                _phone(reference) == _phone(prediction) and bool(_phone(reference))
            )
        }
    if group in {"organization", "address"}:
        return {
            "normalized_edit_similarity": _edit_similarity(reference, prediction),
            "token_recall": _token_recall(reference, prediction),
        }
    return {
        "normalized_exact_match": float(
            _normalize_space(reference).casefold()
            == _normalize_space(prediction).casefold()
        )
    }


def _primary_exact(field: str, metrics: Mapping[str, float]) -> float:
    group = _group(field)
    if group == "amount":
        return float(metrics["numeric_exact_match"])
    if group == "identifier":
        return float(metrics["alphanumeric_exact_match"])
    if group in {"organization", "address"}:
        return float(
            metrics["normalized_edit_similarity"] >= 0.90
            and metrics["token_recall"] >= 0.80
        )
    return float(next(iter(metrics.values())))


def _group(field: str) -> str:
    normalized = str(field).casefold()
    for group, names in FIELD_GROUPS.items():
        if normalized in names:
            return group
    return "generic"


def _normalize_space(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def _normalize_date(value: str) -> str:
    numbers = re.findall(r"\d+", unicodedata.normalize("NFKC", value))
    if len(numbers) != 3:
        return ""
    first, second, third = numbers
    if len(first) == 4:
        year, month, day = first, second, third
    elif len(third) == 4:
        day, month, year = first, second, third
    else:
        return ""
    try:
        year_i, month_i, day_i = int(year), int(month), int(day)
    except ValueError:
        return ""
    if not (1 <= month_i <= 12 and 1 <= day_i <= 31):
        return ""
    return f"{year_i:04d}-{month_i:02d}-{day_i:02d}"


def _decimal(value: str) -> Decimal | None:
    normalized = unicodedata.normalize("NFKC", value)
    match = re.search(r"[-+]?\d[\d\s.,]*", normalized)
    if not match:
        return None
    number = match.group(0).replace(" ", "")
    if "," in number and "." in number:
        decimal_mark = "," if number.rfind(",") > number.rfind(".") else "."
        thousands = "." if decimal_mark == "," else ","
        number = number.replace(thousands, "").replace(decimal_mark, ".")
    elif "," in number:
        suffix = number.rsplit(",", 1)[1]
        number = number.replace(",", "." if len(suffix) in {1, 2} else "")
    elif number.count(".") > 1:
        final = number.rsplit(".", 1)
        number = final[0].replace(".", "") + "." + final[1]
    try:
        return Decimal(number)
    except InvalidOperation:
        return None


def _decimal_separator(value: str) -> str | None:
    match = re.search(r"\d([.,])\d{1,2}(?!\d)", value)
    return match.group(1) if match else None


def _alphanumeric(value: str) -> str:
    return "".join(character.casefold() for character in value if character.isalnum())


def _normalize_currency(value: str) -> str:
    normalized = _normalize_space(value).upper()
    aliases = {
        "฿": "THB",
        "BAHT": "THB",
        "$": "USD",
        "US$": "USD",
        "€": "EUR",
        "£": "GBP",
        "₺": "TRY",
    }
    return aliases.get(normalized, normalized)


def _phone(value: str) -> str:
    return "".join(character for character in value if character.isdigit())


def _tokens(value: str) -> list[str]:
    return re.findall(r"\w+", unicodedata.normalize("NFKC", value).casefold())


def _token_recall(reference: str, prediction: str) -> float:
    expected = Counter(_tokens(reference))
    observed = Counter(_tokens(prediction))
    if not expected:
        return 0.0
    matched = sum(min(count, observed[token]) for token, count in expected.items())
    return matched / sum(expected.values())


def _edit_similarity(reference: str, prediction: str) -> float:
    expected = _normalize_space(reference).casefold()
    observed = _normalize_space(prediction).casefold()
    return SequenceMatcher(None, expected, observed).ratio() if expected else 0.0
