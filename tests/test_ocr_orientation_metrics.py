from __future__ import annotations

import pytest

from scripts.evaluate_ocr_orientation import (
    _multi_orientation_signal,
    _orientation_error_categories,
    _selected_preprocessing_profile,
    _selected_profile,
    circular_error,
)


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [(359, 1, 2), (1, 359, 2), (90, 270, 180), (12, 12, 0)],
)
def test_circular_error_wraps_at_360(
    left: float,
    right: float,
    expected: float,
) -> None:
    assert circular_error(left, right) == pytest.approx(expected)


def test_orientation_error_categories_are_aggregate_only() -> None:
    rows = [
        {
            "wrong_cardinal_orientation": True,
            "wrong_deskew": False,
            "close_candidate_ambiguity": True,
            "multi_orientation_signal": False,
        },
        {
            "wrong_cardinal_orientation": False,
            "wrong_deskew": True,
            "close_candidate_ambiguity": False,
            "multi_orientation_signal": True,
        },
    ]

    assert _orientation_error_categories(rows) == {
        "wrong_cardinal_orientation": 1,
        "wrong_deskew": 1,
        "close_candidate_ambiguity": 1,
        "multi_orientation_page": 1,
    }


def test_multi_orientation_signal_requires_distinct_text_directions() -> None:
    horizontal = {
        "polygon": [[0, 0], [100, 0], [100, 20], [0, 20]]
    }
    vertical = {
        "polygon": [[0, 0], [0, 100], [20, 100], [20, 0]]
    }

    assert _multi_orientation_signal([horizontal] * 4) is False
    assert _multi_orientation_signal(
        [horizontal, horizontal, vertical, vertical]
    ) is True


def test_orientation_evaluation_uses_frozen_profile_and_preprocessing() -> None:
    cfg = {
        "ocr": {
            "default_profile": "adaptive",
            "preprocessing_profile": "original",
            "adaptive_preprocessing_profile": "grayscale_normalized",
        }
    }
    profile = _selected_profile(cfg, "auto")
    assert profile == "adaptive"
    assert (
        _selected_preprocessing_profile(
            cfg,
            selected_profile=profile,
            requested="auto",
        )
        == "grayscale_normalized"
    )
