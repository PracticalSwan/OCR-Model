from __future__ import annotations

import copy

from src.information_extraction.ocr_noise import (
    OCRNoiseConfig,
    build_noisy_example,
)


def _example() -> dict:
    tokens = [
        {
            "id": f"t{index}",
            "text": text,
            "bbox": [index * 20, 10, index * 20 + 18, 24],
            "polygon": [
                [index * 20, 10],
                [index * 20 + 18, 10],
                [index * 20 + 18, 24],
                [index * 20, 24],
            ],
            "confidence": 0.9,
        }
        for index, text in enumerate(
            ["INVOICE", "INV-001", "TOTAL", "1,250.00", "USD", "2026/07/24"]
        )
    ]
    return {
        "example_id": "page-1__ground_truth",
        "page_id": "page-1",
        "project_split": "train",
        "token_source": "ground_truth",
        "tokens": tokens,
        "labels": ["KEY", "VALUE", "KEY", "VALUE", "VALUE", "VALUE"],
        "token_loss_mask": [True] * len(tokens),
        "entities": [{"id": "e1", "token_ids": ["t1"]}],
        "relations": [{"id": "r1", "source_id": "e1", "target_id": "e1"}],
        "canonical_fields": {
            "invoice_number": {"token_ids": ["t1"], "value": "INV-001"}
        },
        "page": {"width": 300, "height": 100},
        "is_private": False,
    }


def test_noise_is_seeded_bounded_and_preserves_original() -> None:
    source = _example()
    untouched = copy.deepcopy(source)
    config = OCRNoiseConfig(
        example_probability=1.0,
        maximum_token_fraction=0.34,
        maximum_transformations=2,
        seed=42,
    )

    first = build_noisy_example(source, config=config)
    second = build_noisy_example(source, config=config)

    assert source == untouched
    assert first == second
    assert first is not None
    assert first["token_source"] == "ocr_noise"
    assert 1 <= len(first["ocr_noise"]["transformations"]) <= 2
    assert first["ocr_noise"]["private_row_count"] == 0
    assert first["relations"] == source["relations"]
    assert first["canonical_fields"] == source["canonical_fields"]


def test_noise_probability_can_preserve_only_the_unmodified_example() -> None:
    assert (
        build_noisy_example(
            _example(),
            config=OCRNoiseConfig(example_probability=0.0),
        )
        is None
    )
