from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from PIL import Image

from src.ocr.training_data import (
    assert_split_isolation,
    build_line_groups,
    convert_fatura_detection,
    convert_funsd_detection,
    convert_sroie_detection,
    critical_fields_by_token,
    rectify_polygon_crop,
    repaired_annotation_tokens,
    validate_transcription,
)


def _annotation(dataset: str) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "document_id": f"{dataset}-document",
        "page_id": f"{dataset}-page",
        "dataset": dataset,
        "image_path": f"data/raw/public/{dataset}/page.png",
        "page": {"width": 100, "height": 80},
        "tokens": [
            {
                "id": "token-1",
                "text": "TOTAL",
                "polygon": [[5, 5], [45, 5], [45, 20], [5, 20]],
                "source_id": "region-1",
            },
            {
                "id": "token-2",
                "text": "12.50",
                "bbox": [50, 5, 90, 20],
                "source_id": "region-2",
            },
        ],
        "canonical_fields": {
            "total_amount": {
                "value": "12.50",
                "token_ids": ["token-2"],
            }
        },
        "is_private": False,
    }


@pytest.mark.parametrize(
    ("converter", "dataset"),
    [
        (convert_fatura_detection, "fatura"),
        (convert_funsd_detection, "funsd"),
        (convert_sroie_detection, "sroie"),
    ],
)
def test_each_detection_converter_preserves_polygon_and_converts_bbox(
    converter, dataset: str
) -> None:
    converted = converter(_annotation(dataset))

    assert converted[0]["points"] == [[5.0, 5.0], [45.0, 5.0], [45.0, 20.0], [5.0, 20.0]]
    assert converted[1]["points"] == [
        [50.0, 5.0],
        [90.0, 5.0],
        [90.0, 20.0],
        [50.0, 20.0],
    ]
    assert [row["transcription"] for row in converted] == ["TOTAL", "12.50"]


def test_detection_converter_collapses_duplicate_source_region() -> None:
    annotation = _annotation("fatura")
    annotation["tokens"].append(  # type: ignore[union-attr]
        {
            "id": "token-3",
            "text": "DUE",
            "polygon": [[5, 5], [45, 5], [45, 20], [5, 20]],
            "source_id": "region-1",
        }
    )

    converted = convert_fatura_detection(annotation)

    assert len(converted) == 2
    assert converted[0]["transcription"] == "TOTAL DUE"
    assert converted[0]["token_ids"] == ["token-1", "token-3"]


def test_detection_converter_rejects_self_intersection() -> None:
    annotation = _annotation("sroie")
    annotation["tokens"][0]["polygon"] = [[5, 5], [45, 20], [5, 20], [45, 5]]  # type: ignore[index]

    with pytest.raises(ValueError, match="self-intersection"):
        convert_sroie_detection(annotation)


def test_sroie_unmatched_quote_continuations_are_recovered() -> None:
    annotation = _annotation("sroie")
    annotation["tokens"][0]["text"] = (  # type: ignore[index]
        "FIRST PHYSICAL LINE\r\n"
        "6,25,46,25,46,40,6,40,SECOND PHYSICAL LINE"
    )

    repaired = repaired_annotation_tokens(annotation)
    converted = convert_sroie_detection(annotation)

    assert [token["text"] for token in repaired[:2]] == [
        "FIRST PHYSICAL LINE",
        "SECOND PHYSICAL LINE",
    ]
    assert repaired[1]["polygon"] == [
        [6.0, 25.0],
        [46.0, 25.0],
        [46.0, 40.0],
        [6.0, 40.0],
    ]
    assert [region["transcription"] for region in converted[:2]] == [
        "FIRST PHYSICAL LINE",
        "SECOND PHYSICAL LINE",
    ]
    assert converted[2]["transcription"] == "12.50"


def test_split_isolation_rejects_duplicate_and_hash_leakage() -> None:
    rows = [
        {
            "page_id": "train-page",
            "project_split": "train",
            "duplicate_group_id": "dup-1",
            "sha256": "a" * 64,
        },
        {
            "page_id": "validation-page",
            "project_split": "dev_select",
            "duplicate_group_id": "dup-1",
            "sha256": "b" * 64,
        },
    ]

    with pytest.raises(ValueError, match="duplicate_group_id"):
        assert_split_isolation(rows)

    rows[1]["duplicate_group_id"] = "dup-2"
    rows[1]["sha256"] = "a" * 64
    with pytest.raises(ValueError, match="source hash"):
        assert_split_isolation(rows)


def test_line_grouping_orders_words_and_retains_provenance() -> None:
    annotation = _annotation("funsd")

    groups = build_line_groups(annotation["tokens"])  # type: ignore[arg-type]

    assert len(groups) == 1
    assert groups[0]["transcription"] == "TOTAL 12.50"
    assert groups[0]["token_ids"] == ["token-1", "token-2"]
    assert groups[0]["polygon"] == [[5.0, 5.0], [90.0, 5.0], [90.0, 20.0], [5.0, 20.0]]


def test_critical_field_mapping_uses_canonical_token_ids() -> None:
    mapping = critical_fields_by_token(_annotation("fatura"))

    assert mapping == {"token-2": ("total_amount",)}


def test_polygon_crop_is_rectified_with_padding(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (100, 80), "white").save(source)
    with Image.open(source) as image:
        crop = rectify_polygon_crop(
            image,
            [[10, 10], [60, 12], [58, 32], [8, 30]],
            padding_ratio=0.1,
        )

    assert crop.width > 0
    assert crop.height > 0
    output = tmp_path / "crop.png"
    crop.save(output)
    assert hashlib.sha256(output.read_bytes()).hexdigest()


@pytest.mark.parametrize("value", ["bad\ttext", "bad\ntext", "", "   "])
def test_transcription_validation_rejects_list_corruption(value: str) -> None:
    with pytest.raises(ValueError, match="transcription"):
        validate_transcription(value)
