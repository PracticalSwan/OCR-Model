from __future__ import annotations

from dataclasses import dataclass

from scripts.download_ocr_training_checkpoints import compare_state_shapes


@dataclass
class _Tensor:
    shape: tuple[int, ...]


def test_checkpoint_shape_comparison_reports_exact_compatibility() -> None:
    checkpoint = {
        "backbone.weight": _Tensor((4, 3, 3, 3)),
        "head.bias": _Tensor((8,)),
    }
    model = {
        "backbone.weight": _Tensor((4, 3, 3, 3)),
        "head.bias": _Tensor((8,)),
    }

    result = compare_state_shapes(checkpoint, model)

    assert result["model_parameter_match_ratio"] == 1.0
    assert result["shape_mismatch_count"] == 0
    assert result["missing_parameter_count"] == 0
    assert result["unexpected_parameter_count"] == 0


def test_checkpoint_shape_comparison_surfaces_mismatch_and_missing_keys() -> None:
    checkpoint = {
        "backbone.weight": _Tensor((4, 3, 3, 3)),
        "head.bias": _Tensor((9,)),
        "unused": _Tensor((1,)),
    }
    model = {
        "backbone.weight": _Tensor((4, 3, 3, 3)),
        "head.bias": _Tensor((8,)),
        "head.weight": _Tensor((8, 4)),
    }

    result = compare_state_shapes(checkpoint, model)

    assert result["model_parameter_match_ratio"] == 1 / 3
    assert result["shape_mismatch_examples"] == ["head.bias"]
    assert result["missing_parameter_examples"] == ["head.weight"]
    assert result["unexpected_parameter_examples"] == ["unused"]
