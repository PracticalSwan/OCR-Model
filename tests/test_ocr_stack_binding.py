from __future__ import annotations

from dataclasses import dataclass

from src.ocr.stack_binding import build_ocr_stack_binding


@dataclass(frozen=True)
class _Artifact:
    name: str
    model_id: str
    artifact_hash: str


class _Registry:
    detector = _Artifact("det", "det_original", "a" * 64)
    general = _Artifact("general", "general_original", "b" * 64)
    thai = _Artifact("thai", "thai_custom", "c" * 64)

    def route_models(self, route: str):
        return (
            self.detector,
            self.general if route == "general" else self.thai,
        )


def test_ocr_stack_binding_changes_with_adaptive_policy() -> None:
    cfg = {
        "ocr": {
            "preprocessing_version": "3.0",
            "preprocessing_profile": "original",
            "adaptive_preprocessing_profile": "quality_auto",
            "adaptive_rendering": {"enabled": True, "base_dpi": 200},
            "tiling": {"enabled": True, "grid": [2, 2]},
            "recognition_retries": {"enabled": True, "maximum_candidates": 5},
        }
    }

    original = build_ocr_stack_binding(
        cfg, _Registry(), ocr_profile="original"
    )
    adaptive = build_ocr_stack_binding(
        cfg, _Registry(), ocr_profile="adaptive"
    )

    assert original["detector_sha256"] == "a" * 64
    assert original["general_recognizer_sha256"] == "b" * 64
    assert original["thai_recognizer_sha256"] == "c" * 64
    assert original["recognizer_sha256"] != "b" * 64
    assert original["preprocessing_sha256"] != adaptive["preprocessing_sha256"]
    assert original["preprocessing_policy"]["tiling"]["enabled"] is False
    assert adaptive["preprocessing_policy"]["tiling"]["enabled"] is True
