from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from src.ocr.adaptive import (
    AdaptiveRenderingConfig,
    page_quality_signals,
    rerender_reasons,
    select_render_candidate,
)
from src.ocr.crops import (
    PADDING_PROFILES,
    recognize_with_retries,
    rectify_polygon_crop,
)
from src.ocr.tiling import (
    generate_tiles,
    map_tile_result_to_page,
    merge_tiled_results,
)
from src.ocr.pipeline import MultilingualOCR


def _word(
    word_id: str,
    text: str,
    confidence: float,
    bbox: tuple[float, float, float, float],
) -> dict:
    x0, y0, x1, y1 = bbox
    return {
        "id": word_id,
        "text": text,
        "confidence": confidence,
        "bbox": [x0, y0, x1, y1],
        "polygon": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
    }


def _result(words: list[dict], *, duration: float = 0.1) -> dict:
    return {
        "full_text": "\n".join(word["text"] for word in words),
        "words": words,
        "lines": [],
        "mean_confidence": (
            sum(word["confidence"] for word in words) / len(words) if words else None
        ),
        "detector_model": "detector",
        "recognizer_model": "recognizer",
        "language_route": "general",
        "orientation": 0.0,
        "duration_seconds": duration,
        "warnings": [],
        "provenance_hash": "a" * 64,
    }


def test_weak_page_signals_trigger_bounded_300_dpi_rerender() -> None:
    weak = _result([_word("w1", "x", 0.30, (5, 5, 15, 10))])
    signals = page_quality_signals(
        weak,
        image_width=1000,
        image_height=1400,
        critical_fields_expected=True,
        table_like=True,
    )
    reasons = rerender_reasons(signals, AdaptiveRenderingConfig())

    assert "suspiciously_few_text_boxes" in reasons
    assert "low_recognition_confidence" in reasons
    assert "critical_text_missing" in reasons
    assert "sparse_table_detection" in reasons


def test_render_selection_records_both_scores_and_prefers_quality_gain() -> None:
    first = _result([_word("w1", "x", 0.25, (5, 5, 15, 10))], duration=0.2)
    second = _result(
        [
            _word("w1", "Invoice 123", 0.95, (10, 10, 210, 35)),
            _word("w2", "Total 45.00", 0.92, (10, 50, 190, 75)),
        ],
        duration=0.35,
    )

    selected, provenance = select_render_candidate(
        first,
        second,
        first_size=(1000, 1400),
        second_size=(1500, 2100),
        first_dpi=200,
        second_dpi=300,
        reasons=("low_recognition_confidence",),
    )

    assert selected is second
    assert provenance["selected_dpi"] == 300
    assert provenance["first_pass_score"] < provenance["second_pass_score"]
    assert provenance["processing_time_impact_seconds"] == pytest.approx(0.35)


def test_overlapping_tiles_cover_page_and_restore_coordinates() -> None:
    tiles = generate_tiles(1000, 800, grid=(2, 2), overlap=0.15)
    assert len(tiles) == 4
    assert tiles[0].x0 == 0 and tiles[0].y0 == 0
    assert tiles[-1].x1 == 1000 and tiles[-1].y1 == 800
    assert tiles[0].x1 > tiles[1].x0

    tile = tiles[-1]
    mapped = map_tile_result_to_page(
        _result([_word("local", "TOTAL", 0.9, (10, 20, 110, 50))]),
        tile,
        upscale=2.0,
    )
    assert mapped["words"][0]["bbox"] == pytest.approx(
        [tile.x0 + 5, tile.y0 + 10, tile.x0 + 55, tile.y0 + 25]
    )


def test_tile_merge_removes_duplicate_and_preserves_reading_order() -> None:
    first = _result(
        [
            _word("a", "Invoice", 0.80, (20, 20, 120, 45)),
            _word("b", "Total", 0.85, (20, 100, 90, 125)),
        ]
    )
    second = _result(
        [
            _word("c", "Invoice", 0.96, (22, 21, 122, 46)),
            _word("d", "45.00", 0.90, (100, 100, 180, 125)),
        ]
    )

    merged = merge_tiled_results(
        [first, second],
        image_width=500,
        image_height=700,
        polygon_iou_threshold=0.5,
    )

    assert [word["text"] for word in merged["words"]] == [
        "Invoice",
        "Total",
        "45.00",
    ]
    assert merged["words"][0]["confidence"] == pytest.approx(0.96)
    assert merged["tiling"]["duplicates_removed"] == 1


def test_rectified_crop_uses_proportional_padding_and_rejects_invalid() -> None:
    image = Image.new("RGB", (300, 160), "white")
    polygon = [[50, 50], [250, 45], [255, 95], [45, 100]]
    crop, metadata = rectify_polygon_crop(image, polygon, PADDING_PROFILES["B"])

    assert crop.width > crop.height
    assert metadata["padding_profile"] == "B"
    assert metadata["horizontal_padding_ratio"] == pytest.approx(0.05)
    assert metadata["vertical_padding_ratio"] == pytest.approx(0.12)
    with pytest.raises(ValueError, match="four finite points"):
        rectify_polygon_crop(image, [[0, 0], [1, 1]], PADDING_PROFILES["A"])


def test_low_confidence_retry_is_bounded_and_retains_evidence() -> None:
    image = Image.new("RGB", (300, 160), "white")
    polygon = [[50, 50], [250, 50], [250, 100], [50, 100]]
    predictions = iter(
        [
            {"text": "T0tal 4S.00", "confidence": 0.35},
            {"text": "Total 45.00", "confidence": 0.93},
            {"text": "Total 45.00", "confidence": 0.88},
            {"text": "Total 45.00", "confidence": 0.86},
            {"text": "Total 45.00", "confidence": 0.84},
        ]
    )

    decision = recognize_with_retries(
        image,
        polygon,
        lambda _: next(predictions),
        route="general",
        expected_pattern=r"\d+[.,]\d{2}",
        max_candidates=5,
        confidence_threshold=0.70,
    )

    assert decision["text"] == "Total 45.00"
    assert decision["candidate_count"] <= 5
    assert decision["retry_performed"] is True
    assert decision["original_candidate"]["text"] == "T0tal 4S.00"
    assert all("score" in item and "variant" in item for item in decision["candidates"])


def test_multilingual_pipeline_activates_tiles_only_for_weak_full_page() -> None:
    class AreaSensitiveBackend:
        def predict(self, image: Image.Image, *, orientation: float = 0.0) -> dict:
            count = 1 if image.width * image.height > 500_000 else 3
            words = [
                _word(
                    f"{orientation}-{index}",
                    f"TEXT-{index}",
                    0.92,
                    (20 + index * 90, 20, 90 + index * 90, 50),
                )
                for index in range(count)
            ]
            return _result(words)

    backend = AreaSensitiveBackend()
    pipeline = MultilingualOCR(
        general_backend=backend,
        thai_backend=backend,
        cardinal_angles=(0,),
        enable_fine_deskew=False,
        enable_tiling=True,
        tile_grid=(2, 2),
        tile_overlap=0.15,
    )

    output = pipeline.extract_page(
        Image.new("RGB", (1000, 800), "white"),
        language_mode="general",
    )

    assert output["tiling"]["triggered"] is True
    assert output["tiling"]["selected"] is True
    assert output["tiling"]["tile_count"] == 4
    assert len(output["words"]) > 1
