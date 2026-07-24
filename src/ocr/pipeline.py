"""Multilingual, rotation-aware OCR orchestration independent of K-Means."""
from __future__ import annotations

import time
import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from src.ocr.cache import OCRCache, OCRCacheKey
from src.ocr.adaptive import page_quality_signals
from src.ocr.fine_deskew import estimate_residual_deskew
from src.ocr.language_router import (
    normalize_language_mode,
    route_for_mode,
    select_route_result,
    should_try_thai,
)
from src.ocr.model_registry import ModelRegistry
from src.ocr.orientation_candidates import (
    build_orientation_candidates,
    make_orientation_candidate,
    restore_original_coordinates,
)
from src.ocr.paddleocr_adapter import PaddleOCRAdapter
from src.ocr.preprocessing import preprocess_for_ocr
from src.ocr.scoring import score_ocr_candidate
from src.ocr.tiling import (
    generate_tiles,
    map_tile_result_to_page,
    merge_tiled_results,
    tiling_trigger_reasons,
)


class OCRBackend(Protocol):
    def predict(self, image: Image.Image, *, orientation: float = 0.0) -> dict[str, Any]: ...


class MultilingualOCR:
    """Evaluate OCR views/routes and return the auditable best result."""

    def __init__(
        self,
        registry: ModelRegistry | None = None,
        *,
        device: str = "cpu",
        general_backend: OCRBackend | None = None,
        thai_backend: OCRBackend | None = None,
        cardinal_angles: tuple[float, ...] = (0, 90, 180, 270),
        adapter_options: Mapping[str, Any] | None = None,
        cache: OCRCache | None = None,
        preprocessing_version: str = "1.0",
        preprocessing_profile: str = "original",
        enable_fine_deskew: bool = True,
        fine_deskew_reliability: float = 0.55,
        maximum_fine_candidates: int = 2,
        enable_tiling: bool = False,
        tile_grid: tuple[int, int] = (2, 2),
        tile_overlap: float = 0.15,
        tile_upscale: float = 1.0,
        tile_iou_threshold: float = 0.50,
        tile_minimum_score_gain: float = 0.0,
    ) -> None:
        options = dict(adapter_options or {})
        if general_backend is None:
            if registry is None:
                raise ValueError("registry is required when no general backend is injected")
            general_backend = PaddleOCRAdapter(registry, "general", device=device, **options)
        if thai_backend is None:
            if registry is None:
                raise ValueError("registry is required when no Thai backend is injected")
            thai_backend = PaddleOCRAdapter(registry, "thai", device=device, **options)
        self.backends = {"general": general_backend, "thai": thai_backend}
        self.cardinal_angles = tuple(float(value) for value in cardinal_angles)
        self.cache = cache
        self.preprocessing_version = preprocessing_version
        self.preprocessing_profile = preprocessing_profile
        self.enable_fine_deskew = bool(enable_fine_deskew)
        self.fine_deskew_reliability = float(fine_deskew_reliability)
        self.maximum_fine_candidates = int(maximum_fine_candidates)
        self.enable_tiling = bool(enable_tiling)
        self.tile_grid = tuple(map(int, tile_grid))
        self.tile_overlap = float(tile_overlap)
        self.tile_upscale = float(tile_upscale)
        self.tile_iou_threshold = float(tile_iou_threshold)
        self.tile_minimum_score_gain = float(tile_minimum_score_gain)
        # Validate geometry and thresholds before any expensive model initialization.
        generate_tiles(100, 100, grid=self.tile_grid, overlap=self.tile_overlap)
        if self.tile_upscale <= 0.0:
            raise ValueError("tile upscale must be positive")
        if not 0.0 <= self.tile_iou_threshold <= 1.0:
            raise ValueError("tile IoU threshold must be in [0, 1]")

    def extract_path(
        self,
        image_path: str | Path,
        *,
        language_mode: str = "auto",
        language_hint: str | None = None,
        metadata_language: str | None = None,
        deskew_angle: float | None = None,
        private: bool = False,
    ) -> dict[str, Any]:
        """Extract a local image with provenance-complete public caching."""
        path = Path(image_path)
        key = self._cache_key(
            path,
            language_mode=language_mode,
            language_hint=language_hint,
            metadata_language=metadata_language,
            deskew_angle=deskew_angle,
        ) if self.cache is not None else None
        if self.cache is not None and key is not None:
            cached = self.cache.get(key, private=private)
            if cached is not None:
                return cached
        with Image.open(path) as image:
            result = self.extract_page(
                image.convert("RGB"), language_mode=language_mode,
                language_hint=language_hint, metadata_language=metadata_language,
                deskew_angle=deskew_angle,
            )
        if self.cache is not None and key is not None:
            self.cache.put(key, result, private=private)
        return result

    def extract_page(
        self,
        image: Image.Image,
        *,
        language_mode: str = "auto",
        language_hint: str | None = None,
        metadata_language: str | None = None,
        deskew_angle: float | None = None,
    ) -> dict[str, Any]:
        """Run OCR without any K-Means input or dependency."""
        started = time.perf_counter()
        image, preprocessing = preprocess_for_ocr(image, self.preprocessing_profile)
        selected = self._extract_preprocessed(
            image,
            language_mode=language_mode,
            language_hint=language_hint,
            metadata_language=metadata_language,
            deskew_angle=deskew_angle,
        )
        selected = self._maybe_apply_tiling(
            image,
            selected,
            language_mode=language_mode,
            language_hint=language_hint,
            metadata_language=metadata_language,
            deskew_angle=deskew_angle,
        )
        selected["preprocessing"] = preprocessing
        selected["duration_seconds"] = time.perf_counter() - started
        return selected

    def _extract_preprocessed(
        self,
        image: Image.Image,
        *,
        language_mode: str,
        language_hint: str | None,
        metadata_language: str | None,
        deskew_angle: float | None,
    ) -> dict[str, Any]:
        """Run route/orientation selection on an already-preprocessed image."""
        mode = normalize_language_mode(language_mode)
        forced_route = route_for_mode(mode)
        candidates = build_orientation_candidates(
            image, cardinal_angles=self.cardinal_angles, deskew_angle=deskew_angle
        )
        general_best = None
        thai_best = None
        route_reasons: list[str] = []
        if forced_route in {None, "general"}:
            general_best = self._best_route("general", candidates, image)
        if forced_route == "thai":
            thai_best = self._best_route("thai", candidates, image)
        elif forced_route is None:
            assert general_best is not None
            run_thai, route_reasons = should_try_thai(
                general_best[0], general_best[1],
                language_hint=language_hint, metadata_language=metadata_language,
            )
            if run_thai:
                thai_best = self._best_route("thai", candidates, image)
        hints = {str(language_hint or "").lower(), str(metadata_language or "").lower()}
        preferred_route = "thai" if hints & {"th", "thai", "th-th"} else None
        selected, route_decision = select_route_result(
            general_best, thai_best, preferred_route=preferred_route
        )
        selected = dict(selected)
        selected["route_decision"] = {**route_decision, "thai_evaluation_reasons": route_reasons}
        selected["candidate_scores"] = [
            candidate for pair in (general_best, thai_best) if pair is not None
            for candidate in pair[0].get("all_candidate_scores", [])
        ]
        return selected

    def _maybe_apply_tiling(
        self,
        image: Image.Image,
        full_page: dict[str, Any],
        *,
        language_mode: str,
        language_hint: str | None,
        metadata_language: str | None,
        deskew_angle: float | None,
    ) -> dict[str, Any]:
        signals = page_quality_signals(
            full_page,
            image_width=image.width,
            image_height=image.height,
        )
        reasons = tiling_trigger_reasons(
            signals,
            image_width=image.width,
            image_height=image.height,
        )
        if not self.enable_tiling or not reasons:
            output = dict(full_page)
            output["tiling"] = {
                "enabled": self.enable_tiling,
                "triggered": False,
                "selected": False,
                "trigger_reasons": reasons,
                "tile_count": 0,
            }
            return output
        tiles = generate_tiles(
            image.width,
            image.height,
            grid=self.tile_grid,
            overlap=self.tile_overlap,
        )
        mapped_results = []
        tile_metadata = []
        for tile in tiles:
            crop = image.crop((tile.x0, tile.y0, tile.x1, tile.y1))
            if self.tile_upscale != 1.0:
                crop = crop.resize(
                    (
                        max(2, round(crop.width * self.tile_upscale)),
                        max(2, round(crop.height * self.tile_upscale)),
                    ),
                    Image.Resampling.LANCZOS,
                )
            tile_result = self._extract_preprocessed(
                crop,
                language_mode=language_mode,
                language_hint=language_hint,
                metadata_language=metadata_language,
                deskew_angle=deskew_angle,
            )
            mapped_results.append(
                map_tile_result_to_page(
                    tile_result,
                    tile,
                    upscale=self.tile_upscale,
                )
            )
            tile_metadata.append(
                {**tile.as_dict(), "upscale": self.tile_upscale}
            )
        merged = merge_tiled_results(
            mapped_results,
            image_width=image.width,
            image_height=image.height,
            polygon_iou_threshold=self.tile_iou_threshold,
        )
        full_score = float(
            score_ocr_candidate(full_page, image.width, image.height)["total"]
        )
        tiled_score = float(
            score_ocr_candidate(merged, image.width, image.height)["total"]
        )
        selected = tiled_score > full_score + self.tile_minimum_score_gain
        output = dict(merged if selected else full_page)
        merge_metadata = dict(merged.get("tiling") or {})
        output["tiling"] = {
            **merge_metadata,
            "enabled": True,
            "triggered": True,
            "selected": selected,
            "trigger_reasons": reasons,
            "tile_count": len(tiles),
            "grid": list(self.tile_grid),
            "overlap": self.tile_overlap,
            "upscale": self.tile_upscale,
            "full_page_score": full_score,
            "tiled_score": tiled_score,
            "minimum_score_gain": self.tile_minimum_score_gain,
            "tiles": tile_metadata,
            "selection_reason": (
                "tiled_quality_gain" if selected else "full_page_not_worse"
            ),
        }
        return output

    def _best_route(
        self, route: str, candidates: list, source_image: Image.Image
    ) -> tuple[dict[str, Any], dict[str, float]]:
        backend = self.backends[route]
        evaluated: list[tuple[dict[str, Any], dict[str, float], Any, dict[str, Any]]] = []
        candidate_scores: list[dict[str, Any]] = []
        seen_angles = {round(float(candidate.angle), 6) for candidate in candidates}

        def evaluate(candidate: Any, deskew: Mapping[str, Any] | None = None) -> None:
            result = backend.predict(candidate.image, orientation=candidate.angle)
            score = score_ocr_candidate(
                result, source_image.width, source_image.height
            )
            restored = restore_original_coordinates(result, candidate)
            entry = {
                "route": route,
                "orientation": candidate.angle,
                "candidate_kind": candidate.kind,
                **score,
            }
            if deskew is not None:
                entry["fine_deskew"] = dict(deskew)
                restored["fine_deskew"] = dict(deskew)
            candidate_scores.append(entry)
            evaluated.append((restored, score, candidate, result))

        for candidate in candidates:
            evaluate(candidate)
        if self.enable_fine_deskew and self.maximum_fine_candidates > 0:
            initial = sorted(
                evaluated,
                key=lambda value: (float(value[1]["total"]), -float(value[0]["orientation"])),
                reverse=True,
            )[: self.maximum_fine_candidates]
            for _, _, candidate, raw_result in initial:
                estimate = estimate_residual_deskew(raw_result.get("words", []))
                correction = float(estimate["correction_degrees"])
                if (
                    float(estimate["reliability"]) < self.fine_deskew_reliability
                    or abs(correction) < 0.25
                    or abs(correction) > 45.0
                ):
                    continue
                refined_angle = (float(candidate.angle) + correction) % 360.0
                key = round(refined_angle, 6)
                if key in seen_angles:
                    continue
                seen_angles.add(key)
                evaluate(
                    make_orientation_candidate(
                        source_image, refined_angle, kind="fine_deskew"
                    ),
                    estimate,
                )
        best_result, best_score, _, _ = max(
            evaluated,
            key=lambda value: (value[1]["total"], -float(value[0]["orientation"])),
        )
        best_result["all_candidate_scores"] = candidate_scores
        best_result["source_width"], best_result["source_height"] = source_image.size
        return best_result, best_score

    def _cache_key(self, path: Path, **configuration: Any) -> OCRCacheKey | None:
        provenance = {}
        for route, backend in self.backends.items():
            method = getattr(backend, "provenance", None)
            if not callable(method):
                return None
            provenance[route] = dict(method())
        general = provenance["general"]
        thai = provenance["thai"]
        recognizer_hash = hashlib.sha256(
            (str(general["recognizer_artifact_hash"]) + str(thai["recognizer_artifact_hash"])).encode("ascii")
        ).hexdigest()
        return OCRCacheKey.from_image(
            path,
            detector_model=str(general["detector_model"]),
            detector_artifact_hash=str(general["detector_artifact_hash"]),
            recognizer_model=f"{general['recognizer_model']}+{thai['recognizer_model']}",
            recognizer_artifact_hash=recognizer_hash,
            language_route_configuration=configuration,
            orientation_configuration={
                "cardinal_angles": self.cardinal_angles,
                "enable_fine_deskew": self.enable_fine_deskew,
                "fine_deskew_reliability": self.fine_deskew_reliability,
                "maximum_fine_candidates": self.maximum_fine_candidates,
                "preprocessing_profile": self.preprocessing_profile,
                "enable_tiling": self.enable_tiling,
                "tile_grid": self.tile_grid,
                "tile_overlap": self.tile_overlap,
                "tile_upscale": self.tile_upscale,
                "tile_iou_threshold": self.tile_iou_threshold,
                "tile_minimum_score_gain": self.tile_minimum_score_gain,
            },
            paddleocr_version=str(general["paddleocr_version"]),
            preprocessing_version=self.preprocessing_version,
        )
