"""Versioned OCR model registry with exact paths, hashes, and fallbacks."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from src.ocr.errors import OCRModelMismatch, OCRModelUnavailable
from src.rotation_common import sha256_file

DETECTOR_MODEL = "PP-OCRv6_medium_det"
GENERAL_RECOGNIZER_MODEL = "PP-OCRv6_medium_rec"
THAI_RECOGNIZER_MODEL = "th_PP-OCRv5_mobile_rec"
REQUIRED_MODEL_NAMES = (
    DETECTOR_MODEL,
    GENERAL_RECOGNIZER_MODEL,
    THAI_RECOGNIZER_MODEL,
)
ORIGINAL_MODEL_IDS = {
    DETECTOR_MODEL: "PP-OCRv6_medium_det_original",
    GENERAL_RECOGNIZER_MODEL: "PP-OCRv6_medium_rec_original",
    THAI_RECOGNIZER_MODEL: "th_PP-OCRv5_mobile_rec_original",
}


@dataclass(frozen=True)
class ModelArtifact:
    """One reloadable inference artifact and its public-safe registry metadata."""

    name: str
    role: str
    language: str
    path: Path
    files: tuple[dict[str, Any], ...]
    runtime_name: str
    variant: str = "original"
    model_id: str | None = None
    upstream_base: str | None = None
    license: str | None = None
    intended_domain: str | None = None
    development_metrics: Mapping[str, Any] = field(default_factory=dict)
    registry_metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def artifact_hash(self) -> str:
        """Match the path-and-content tree digest used by OCR trial reports."""
        digest = hashlib.sha256()
        for item in sorted(
            self.files, key=lambda value: str(value["path"]).casefold()
        ):
            digest.update(str(item["path"]).encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(item["sha256"]).encode("ascii"))
            digest.update(b"\n")
        return digest.hexdigest()


class ModelRegistry:
    """Resolve original/custom/auto choices without silently substituting models."""

    def __init__(
        self,
        artifacts: Mapping[str, ModelArtifact],
        metadata: Mapping[str, Any],
        *,
        selection: Mapping[str, str] | None = None,
    ) -> None:
        self.artifacts = dict(artifacts)
        self.metadata = dict(metadata)
        self.selection = dict(
            selection
            or {
                "detector": DETECTOR_MODEL,
                "general_recognizer": GENERAL_RECOGNIZER_MODEL,
                "thai_recognizer": THAI_RECOGNIZER_MODEL,
            }
        )

    @classmethod
    def from_setup(
        cls,
        path: str | Path,
        *,
        verify_hashes: bool = True,
        upgrade_registry: str | Path | None = None,
        detector_choice: str = "auto",
        general_choice: str = "auto",
        thai_choice: str = "auto",
    ) -> "ModelRegistry":
        setup_path = Path(path)
        if not setup_path.is_file():
            raise OCRModelUnavailable(f"model setup manifest not found: {setup_path}")
        payload = json.loads(setup_path.read_text(encoding="utf-8"))
        models = payload.get("models")
        if not isinstance(models, Mapping):
            raise OCRModelMismatch("model setup manifest has no models mapping")
        artifacts: dict[str, ModelArtifact] = {}
        for expected in REQUIRED_MODEL_NAMES:
            item = models.get(expected)
            if not isinstance(item, Mapping):
                raise OCRModelUnavailable(
                    f"required OCR model missing from registry: {expected}"
                )
            if item.get("requested_name") != expected or item.get("resolved_name") != expected:
                raise OCRModelMismatch(f"model identity mismatch for {expected}")
            model_path = _resolve_model_path(
                item.get("resolved_path"), setup_path.parent
            )
            files = _validate_files(
                model_path,
                item.get("files"),
                verify_hashes=verify_hashes,
                model_name=expected,
            )
            artifact = ModelArtifact(
                name=expected,
                model_id=ORIGINAL_MODEL_IDS[expected],
                runtime_name=expected,
                role=str(item.get("role", "unknown")),
                language=str(item.get("language", "unknown")),
                path=model_path,
                files=files,
                variant="original",
                upstream_base=expected,
                license=str(item.get("license", "Apache-2.0")),
                intended_domain=str(item.get("intended_domain", "upstream general-purpose OCR")),
                development_metrics=dict(item.get("development_metrics") or {}),
                registry_metadata=dict(item),
            )
            artifacts[expected] = artifact
            artifacts[ORIGINAL_MODEL_IDS[expected]] = artifact

        upgrade_payload: dict[str, Any] = {}
        registry_path: Path | None = None
        if upgrade_registry is not None:
            registry_path = Path(upgrade_registry)
            if not registry_path.is_absolute():
                registry_path = (setup_path.parent / registry_path).resolve()
            if not registry_path.is_file():
                raise OCRModelUnavailable(
                    f"OCR upgrade registry not found: {registry_path}"
                )
            upgrade_payload = json.loads(registry_path.read_text(encoding="utf-8"))
            upgrade_models = upgrade_payload.get("models")
            if not isinstance(upgrade_models, Mapping):
                raise OCRModelMismatch("OCR upgrade registry has no models mapping")
            for model_id, raw in upgrade_models.items():
                if not isinstance(raw, Mapping):
                    raise OCRModelMismatch(
                        f"OCR registry entry {model_id} must be an object"
                    )
                if not bool(raw.get("available", True)):
                    continue
                role = str(raw.get("role", ""))
                variant = str(raw.get("variant", "")).casefold()
                if role not in {"detector", "recognizer"} or variant not in {
                    "original",
                    "custom",
                }:
                    raise OCRModelMismatch(
                        f"OCR registry entry {model_id} has invalid role/variant"
                    )
                upstream = str(raw.get("upstream_base", ""))
                if upstream not in REQUIRED_MODEL_NAMES:
                    raise OCRModelMismatch(
                        f"OCR registry entry {model_id} has unsupported upstream base {upstream!r}"
                    )
                model_path = _resolve_model_path(
                    raw.get("local_path"), registry_path.parent
                )
                files = _validate_files(
                    model_path,
                    raw.get("files"),
                    verify_hashes=verify_hashes,
                    model_name=str(model_id),
                )
                artifact = ModelArtifact(
                    name=str(model_id),
                    model_id=str(model_id),
                    runtime_name=upstream,
                    role=role,
                    language=str(raw.get("language", "unknown")),
                    path=model_path,
                    files=files,
                    variant=variant,
                    upstream_base=upstream,
                    license=str(raw.get("license", "unknown")),
                    intended_domain=str(raw.get("intended_domain", "unknown")),
                    development_metrics=dict(raw.get("development_metrics") or {}),
                    registry_metadata=dict(raw),
                )
                declared_hash = raw.get("model_sha256")
                if (
                    verify_hashes
                    and declared_hash is not None
                    and str(declared_hash).casefold() != artifact.artifact_hash
                ):
                    raise OCRModelMismatch(
                        f"model registry hash mismatch for {model_id}: "
                        f"declared {declared_hash}, actual {artifact.artifact_hash}"
                    )
                artifacts[str(model_id)] = artifact

        defaults = dict(upgrade_payload.get("defaults") or {})
        selection = {
            "detector": _select_model(
                artifacts,
                role="detector",
                language=None,
                choice=detector_choice,
                default_choice=str(defaults.get("detector", "original")),
                original_name=DETECTOR_MODEL,
            ),
            "general_recognizer": _select_model(
                artifacts,
                role="recognizer",
                language="general",
                choice=general_choice,
                default_choice=str(defaults.get("general_recognizer", "original")),
                original_name=GENERAL_RECOGNIZER_MODEL,
            ),
            "thai_recognizer": _select_model(
                artifacts,
                role="recognizer",
                language="thai",
                choice=thai_choice,
                default_choice=str(defaults.get("thai_recognizer", "original")),
                original_name=THAI_RECOGNIZER_MODEL,
            ),
        }
        return cls(
            artifacts,
            {
                **payload,
                "upgrade_registry": upgrade_payload or None,
                "upgrade_registry_path": str(registry_path) if registry_path else None,
            },
            selection=selection,
        )

    def require(self, name: str) -> ModelArtifact:
        try:
            return self.artifacts[name]
        except KeyError as exc:
            raise OCRModelUnavailable(
                f"registered OCR model is unavailable: {name}"
            ) from exc

    def route_models(self, route: str) -> tuple[ModelArtifact, ModelArtifact]:
        if route == "general":
            recognizer_key = "general_recognizer"
        elif route == "thai":
            recognizer_key = "thai_recognizer"
        else:
            raise ValueError(f"unsupported OCR route: {route}")
        return (
            self.require(self.selection["detector"]),
            self.require(self.selection[recognizer_key]),
        )


def _resolve_model_path(value: Any, base: Path) -> Path:
    model_path = Path(str(value or "")).expanduser()
    if not model_path.is_absolute():
        model_path = (base / model_path).resolve()
    if not model_path.is_dir():
        raise OCRModelUnavailable(
            f"required OCR model directory not found: {model_path}"
        )
    return model_path


def _validate_files(
    model_path: Path,
    value: Any,
    *,
    verify_hashes: bool,
    model_name: str,
) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list) or not value:
        raise OCRModelMismatch(f"model registry has no files for {model_name}")
    records: list[dict[str, Any]] = []
    for file_record in value:
        if not isinstance(file_record, Mapping):
            raise OCRModelMismatch(f"invalid file record for {model_name}")
        relative = Path(str(file_record.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise OCRModelMismatch(
                f"unsafe model-relative path for {model_name}: {relative}"
            )
        artifact_file = model_path / relative
        if not artifact_file.is_file():
            raise OCRModelUnavailable(f"registered model file missing: {artifact_file}")
        if verify_hashes and sha256_file(artifact_file) != file_record.get("sha256"):
            raise OCRModelMismatch(f"checksum mismatch: {artifact_file}")
        records.append(dict(file_record))
    return tuple(records)


def _select_model(
    artifacts: Mapping[str, ModelArtifact],
    *,
    role: str,
    language: str | None,
    choice: str,
    default_choice: str,
    original_name: str,
) -> str:
    requested = str(choice).casefold()
    if requested not in {"original", "custom", "auto"}:
        raise ValueError(f"unsupported OCR model choice: {choice}")
    desired = str(default_choice).casefold() if requested == "auto" else requested
    if desired not in {"original", "custom"}:
        raise OCRModelMismatch(f"invalid OCR registry default: {default_choice}")
    if desired == "original":
        return original_name
    candidates: dict[str, ModelArtifact] = {}
    for artifact in artifacts.values():
        if (
            artifact.variant == "custom"
            and artifact.role == role
            and (language is None or artifact.language.casefold().startswith(language))
        ):
            candidates[artifact.name] = artifact
    if not candidates:
        if requested == "auto":
            return original_name
        label = f"{language} {role}".strip()
        raise OCRModelUnavailable(
            f"custom {label} was explicitly requested but no accepted artifact is available"
        )
    selected = [
        artifact
        for artifact in candidates.values()
        if bool(artifact.registry_metadata.get("selected", False))
    ]
    pool = selected or list(candidates.values())
    pool.sort(key=lambda artifact: artifact.name)
    if len(pool) > 1 and not selected:
        raise OCRModelMismatch(
            f"multiple custom {role} artifacts exist without one selected: "
            + ", ".join(artifact.name for artifact in pool)
        )
    return pool[0].name
