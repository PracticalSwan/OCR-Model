"""Lazy PaddleOCR 3.7 adapter with explicit exact local model paths."""
from __future__ import annotations

import importlib.metadata
import hashlib
import json
import time
from collections.abc import Mapping
from typing import Any, Callable

import numpy as np
from PIL import Image

from src.ocr.errors import OCRInferenceError, OCRModelMismatch
from src.ocr.crops import recognize_with_retries
from src.ocr.environment import configure_external_environment
from src.ocr.model_registry import ModelRegistry
from src.ocr.result_normalizer import normalize_paddle_result
from src.rotation_common import canonical_json, stable_id


class PaddleOCRAdapter:
    """One explicit detector+recognizer route; no automatic substitution."""

    def __init__(
        self,
        registry: ModelRegistry,
        route: str,
        *,
        device: str = "cpu",
        use_doc_orientation_classify: bool = False,
        use_doc_unwarping: bool = False,
        use_textline_orientation: bool = False,
        enable_mkldnn: bool | None = None,
        enable_recognition_retries: bool = False,
        retry_confidence_threshold: float = 0.65,
        retry_max_candidates: int = 5,
        retry_padding_profile: str = "B",
        crop_recognizer: Callable[[Image.Image], Mapping[str, Any]] | None = None,
    ) -> None:
        self.registry = registry
        self.route = route
        self.device = device
        self.detector, self.recognizer = registry.route_models(route)
        self.options = {
            "use_doc_orientation_classify": bool(use_doc_orientation_classify),
            "use_doc_unwarping": bool(use_doc_unwarping),
            "use_textline_orientation": bool(use_textline_orientation),
            # PaddleOCR enables oneDNN/MKLDNN on CPU by default. PaddlePaddle
            # 3.3's Linux executor rejects an ArrayAttribute used by the pinned
            # PP-OCRv6 artifacts, so prefer the portable plain-CPU path.
            "enable_mkldnn": (
                not str(device).casefold().startswith("cpu")
                if enable_mkldnn is None
                else bool(enable_mkldnn)
            ),
        }
        self._pipeline: Any = None
        self.enable_recognition_retries = bool(enable_recognition_retries)
        self.retry_confidence_threshold = float(retry_confidence_threshold)
        self.retry_max_candidates = int(retry_max_candidates)
        self.retry_padding_profile = str(retry_padding_profile)
        self._crop_recognizer = crop_recognizer
        self._recognizer_pipeline: Any = None
        if not 0.0 <= self.retry_confidence_threshold <= 1.0:
            raise ValueError("retry confidence threshold must be in [0, 1]")
        if not 1 <= self.retry_max_candidates <= 5:
            raise ValueError("retry max candidates must be within [1, 5]")

    @property
    def paddleocr_version(self) -> str:
        return importlib.metadata.version("paddleocr")

    def initialize(self) -> None:
        if self._pipeline is not None:
            return
        configure_external_environment()
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:  # pragma: no cover - depends on external env
            raise OCRInferenceError(f"PaddleOCR or a required dependency failed to import: {exc}") from exc
        kwargs = {
            "text_detection_model_name": getattr(
                self.detector, "runtime_name", self.detector.name
            ),
            "text_detection_model_dir": str(self.detector.path),
            "text_recognition_model_name": getattr(
                self.recognizer, "runtime_name", self.recognizer.name
            ),
            "text_recognition_model_dir": str(self.recognizer.path),
            "device": self.device,
            **self.options,
        }
        try:
            self._pipeline = PaddleOCR(**kwargs)
        except TypeError as exc:
            raise OCRInferenceError(
                "installed PaddleOCR constructor is incompatible with the verified 3.7 adapter arguments"
            ) from exc
        model_names = str(getattr(self._pipeline, "paddlex_config", "")) + repr(self._pipeline)
        # Constructor paths are authoritative. If the installed object exposes
        # its configuration, a conflicting model name is a hard failure.
        for expected in (
            getattr(self.detector, "runtime_name", self.detector.name),
            getattr(self.recognizer, "runtime_name", self.recognizer.name),
        ):
            if model_names and "model_name" in model_names and expected not in model_names:
                raise OCRModelMismatch(f"initialized PaddleOCR pipeline does not expose expected model {expected}")

    def predict(self, image: Image.Image, *, orientation: float = 0.0) -> dict[str, Any]:
        self.initialize()
        started = time.perf_counter()
        try:
            raw = self._pipeline.predict(np.asarray(image.convert("RGB")))
            result = list(raw)
        except Exception as exc:  # pragma: no cover - external engine details
            raise OCRInferenceError(f"PaddleOCR {self.route} inference failed: {exc}") from exc
        normalized = normalize_paddle_result(
            result,
            detector_model=self.detector.name,
            recognizer_model=self.recognizer.name,
            route=self.route,
            orientation=orientation,
            duration_seconds=time.perf_counter() - started,
        )
        if self.enable_recognition_retries:
            normalized = self._apply_recognition_retries(image, normalized)
        normalized["duration_seconds"] = time.perf_counter() - started
        return normalized

    def _apply_recognition_retries(
        self,
        image: Image.Image,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        words = [dict(word) for word in result.get("words") or []]
        evidence: list[dict[str, Any]] = []
        for index, word in enumerate(words):
            confidence = word.get("confidence")
            if confidence is not None and float(confidence) >= self.retry_confidence_threshold:
                continue
            try:
                decision = recognize_with_retries(
                    image,
                    word["polygon"],
                    self._recognize_crop,
                    route=self.route,
                    max_candidates=self.retry_max_candidates,
                    confidence_threshold=self.retry_confidence_threshold,
                    initial_profile=self.retry_padding_profile,
                    initial_prediction={
                        "text": word.get("text", ""),
                        "confidence": confidence or 0.0,
                    },
                )
            except Exception as exc:
                result.setdefault("warnings", []).append(
                    f"low-confidence recognition retry failed for word {index}: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue
            evidence.append(
                {
                    "word_index": index,
                    "source_word_id": word.get("id"),
                    **decision,
                }
            )
            if float(decision["score"]) <= 0.0:
                continue
            word["text"] = str(decision["text"])
            word["confidence"] = float(decision["confidence"])
            word["id"] = stable_id(
                "ocr",
                self.route,
                result.get("orientation", 0.0),
                index,
                word["text"],
            )
        if not evidence:
            return result
        lines = [
            {
                "id": stable_id("line", word["id"]),
                "text": word["text"],
                "word_ids": [word["id"]],
                "polygon": word["polygon"],
                "bbox": word["bbox"],
                "confidence": word.get("confidence"),
            }
            for word in words
        ]
        confidences = [
            float(word["confidence"])
            for word in words
            if word.get("confidence") is not None
        ]
        result = dict(result)
        result["words"] = words
        result["lines"] = lines
        result["full_text"] = "\n".join(line["text"] for line in lines)
        result["mean_confidence"] = (
            sum(confidences) / len(confidences) if confidences else None
        )
        result["recognition_retries"] = {
            "enabled": True,
            "confidence_threshold": self.retry_confidence_threshold,
            "maximum_candidates_per_crop": self.retry_max_candidates,
            "padding_profile": self.retry_padding_profile,
            "retried_word_count": len(evidence),
            "items": evidence,
        }
        result["provenance_hash"] = hashlib.sha256(
            canonical_json(
                {
                    "detector_model": result.get("detector_model"),
                    "recognizer_model": result.get("recognizer_model"),
                    "words": words,
                    "recognition_retries": result["recognition_retries"],
                }
            ).encode("utf-8")
        ).hexdigest()
        return result

    def _recognize_crop(self, crop: Image.Image) -> Mapping[str, Any]:
        if self._crop_recognizer is not None:
            return self._crop_recognizer(crop)
        if self._recognizer_pipeline is None:
            configure_external_environment()
            try:
                from paddleocr import TextRecognition
            except ImportError as exc:  # pragma: no cover - external environment
                raise OCRInferenceError(
                    f"PaddleOCR text recognizer failed to import: {exc}"
                ) from exc
            self._recognizer_pipeline = TextRecognition(
                model_name=getattr(self.recognizer, "runtime_name", self.recognizer.name),
                model_dir=str(self.recognizer.path),
                device=self.device,
                enable_mkldnn=self.options["enable_mkldnn"],
            )
        try:
            values = list(
                self._recognizer_pipeline.predict(np.asarray(crop.convert("RGB")))
            )
        except Exception as exc:  # pragma: no cover - external engine details
            raise OCRInferenceError(f"recognizer retry inference failed: {exc}") from exc
        if not values:
            return {"text": "", "confidence": 0.0}
        payload = _mapping(values[0])
        if isinstance(payload.get("res"), Mapping):
            payload = dict(payload["res"])
        text = payload.get("rec_text", payload.get("text", ""))
        confidence = payload.get(
            "rec_score",
            payload.get("score", payload.get("confidence", 0.0)),
        )
        if isinstance(text, (list, tuple)):
            text = text[0] if text else ""
        if isinstance(confidence, (list, tuple)):
            confidence = confidence[0] if confidence else 0.0
        return {"text": str(text), "confidence": confidence}

    def provenance(self) -> Mapping[str, Any]:
        return {
            "detector_model": self.detector.name,
            "detector_artifact_hash": self.detector.artifact_hash,
            "recognizer_model": self.recognizer.name,
            "recognizer_artifact_hash": self.recognizer.artifact_hash,
            "paddleocr_version": self.paddleocr_version,
            "device": self.device,
            **self.options,
            "enable_recognition_retries": self.enable_recognition_retries,
            "retry_confidence_threshold": self.retry_confidence_threshold,
            "retry_max_candidates": self.retry_max_candidates,
            "retry_padding_profile": self.retry_padding_profile,
        }


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    candidate = getattr(value, "json", None)
    if callable(candidate):
        candidate = candidate()
    if isinstance(candidate, str):
        candidate = json.loads(candidate)
    if isinstance(candidate, Mapping):
        return dict(candidate)
    candidate = getattr(value, "res", None)
    if isinstance(candidate, Mapping):
        return dict(candidate)
    raise OCRInferenceError(
        f"unsupported recognition retry result type: {type(value).__name__}"
    )
