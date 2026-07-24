from __future__ import annotations

import pytest

from scripts.evaluate_adaptive_pdf_rendering import _metric_deltas


def test_adaptive_pdf_metric_deltas_keep_error_direction() -> None:
    baseline = {
        "polygon_f1": 0.4,
        "recognized_text_coverage": 0.5,
        "cer": 0.5,
        "wer": 0.6,
        "critical_field_exact_match": 0.2,
        "nonempty_output_rate": 0.9,
        "time_per_page_seconds": 1.0,
        "page_failure_rate": 0.0,
    }
    candidate = dict(baseline)
    candidate.update({"polygon_f1": 0.5, "wer": 0.4})
    deltas = _metric_deltas(baseline, candidate)
    assert deltas["polygon_f1"] == pytest.approx(0.1)
    assert deltas["wer"] == pytest.approx(-0.2)
