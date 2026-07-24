from __future__ import annotations

import pytest

from scripts.evaluate_ocr_orientation import circular_error


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
