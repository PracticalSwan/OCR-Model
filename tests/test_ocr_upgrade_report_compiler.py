from __future__ import annotations

import pytest

from scripts.compile_ocr_upgrade_reports import (
    ERROR_CATEGORIES,
    _selected_ocr_profile,
    error_category_rows,
    render_error_analysis,
)


def _evidence() -> dict:
    locked_counts = {
        category.key: index
        for index, category in enumerate(
            (
                item
                for item in ERROR_CATEGORIES
                if item.source == "locked_ocr"
            ),
            start=1,
        )
    }
    orientation_counts = {
        category.key: index
        for index, category in enumerate(
            (
                item
                for item in ERROR_CATEGORIES
                if item.source == "orientation"
            ),
            start=1,
        )
    }
    return {
        "locked_ocr": {
            "sample_count": 100,
            "error_counts": locked_counts,
        },
        "orientation": {
            "sample_count": 20,
            "metrics": {"error_counts": orientation_counts},
        },
    }


def test_error_analysis_quantifies_every_required_category() -> None:
    evidence = _evidence()
    rows = error_category_rows(evidence)
    report = render_error_analysis(evidence)

    assert len(rows) == 26
    assert {row["category"] for row in rows} == {
        category.label for category in ERROR_CATEGORIES
    }
    for category in ERROR_CATEGORIES:
        assert category.label in report
    assert "frequency per evaluated page × downstream impact" in report
    assert "contains no private filenames" in report


def test_error_analysis_refuses_missing_executed_category() -> None:
    evidence = _evidence()
    evidence["locked_ocr"]["error_counts"].pop(
        "recognition_turkish_character_errors"
    )

    with pytest.raises(
        KeyError,
        match="recognition_turkish_character_errors",
    ):
        error_category_rows(evidence)


def test_selected_ocr_profile_uses_actual_selection_schema() -> None:
    assert (
        _selected_ocr_profile({"selected_ocr_profile": "adaptive"})
        == "adaptive"
    )
