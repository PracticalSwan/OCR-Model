from __future__ import annotations

import random

from PIL import ImageFont

from src.ocr.synthetic import (
    general_financial_text,
    render_synthetic_line,
    synthetic_count_for_fraction,
    thai_financial_text,
)


def test_synthetic_fraction_is_of_combined_real_and_synthetic_data() -> None:
    count = synthetic_count_for_fraction(80, 0.20)

    assert count == 20
    assert count / (80 + count) == 0.20


def test_general_templates_cover_financial_and_turkish_characters() -> None:
    values = [general_financial_text(index, random.Random(index)) for index in range(80)]
    joined = " ".join(values)

    assert "TOTAL" in joined
    assert "VAT" in joined
    assert "INV-" in joined
    assert "€" in joined
    assert "₺" in joined
    assert any(character in joined for character in "ÇĞİÖŞÜçğıöşü")


def test_thai_templates_cover_thai_digits_and_mixed_language() -> None:
    values = [thai_financial_text(index, random.Random(index)) for index in range(80)]
    joined = " ".join(values)

    assert "ใบกำกับภาษี" in joined
    assert "เลขประจำตัวผู้เสียภาษี" in joined
    assert "฿" in joined
    assert any(character in joined for character in "๐๑๒๓๔๕๖๗๘๙")
    assert "INV-" in joined


def test_rendered_line_is_deterministic_and_readable() -> None:
    font = ImageFont.load_default()

    first, first_metadata = render_synthetic_line(
        "TOTAL 1,250.00",
        font=font,
        seed=123,
    )
    second, second_metadata = render_synthetic_line(
        "TOTAL 1,250.00",
        font=font,
        seed=123,
    )

    assert first.tobytes() == second.tobytes()
    assert first_metadata == second_metadata
    assert first.width > first.height > 4
    assert first.convert("L").getextrema()[1] - first.convert("L").getextrema()[0] > 20
