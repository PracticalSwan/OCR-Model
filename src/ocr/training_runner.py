"""Validation and config resolution for bounded official PaddleOCR trials."""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.rotation_common import canonical_json


_TRIAL_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{2,63}$")
_KINDS = frozenset({"detector", "general_recognizer", "thai_recognizer"})
_FORBIDDEN_TRAINING_TOKENS = (
    "test_in_domain",
    "unseen_coru",
    "unseen_domain_test",
    "private_test",
    "private_operational",
    "gmail",
)


@dataclass(frozen=True)
class TrialDefinition:
    path: Path
    trial_id: str
    kind: str
    base_config: str
    source_checkpoint: Path
    overrides: dict[str, Any]
    metadata: dict[str, Any]


def deep_merge(
    base: dict[str, Any],
    patch: dict[str, Any],
) -> dict[str, Any]:
    """Recursively merge mappings while replacing scalar and list values."""
    result = copy.deepcopy(base)
    for key, value in patch.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_trial_definition(
    path: str | Path,
    *,
    expected_kind: str | None = None,
) -> TrialDefinition:
    """Load a repository trial definition and reject data-governance drift."""
    definition_path = Path(path)
    payload = yaml.safe_load(definition_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("trial definition must be a YAML mapping")
    if str(payload.get("schema_version")) != "1.0":
        raise ValueError("trial definition schema_version must be 1.0")
    trial_id = str(payload.get("trial_id", ""))
    if not _TRIAL_ID.fullmatch(trial_id):
        raise ValueError(f"invalid trial_id: {trial_id!r}")
    kind = str(payload.get("kind", ""))
    if kind not in _KINDS:
        raise ValueError(f"unsupported trial kind: {kind!r}")
    if expected_kind is not None and kind != expected_kind:
        raise ValueError(f"expected {expected_kind} trial, found {kind}")
    base_config = str(payload.get("base_config", "")).replace("\\", "/")
    if not base_config or Path(base_config).is_absolute() or ".." in Path(base_config).parts:
        raise ValueError("base_config must be a vendor-relative path")
    source_checkpoint = Path(str(payload.get("source_checkpoint", "")))
    if not source_checkpoint.is_absolute() or not source_checkpoint.is_file():
        raise ValueError(
            f"source_checkpoint must be an existing absolute file: {source_checkpoint}"
        )
    overrides = payload.get("overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError("trial overrides must be a mapping")
    _refuse_forbidden_training_data(overrides.get("Train", {}))
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("trial metadata must be a mapping")
    return TrialDefinition(
        path=definition_path.resolve(),
        trial_id=trial_id,
        kind=kind,
        base_config=base_config,
        source_checkpoint=source_checkpoint.resolve(),
        overrides=copy.deepcopy(overrides),
        metadata=copy.deepcopy(metadata),
    )


def resolve_paddle_config(
    definition: TrialDefinition,
    *,
    vendor_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Merge the pinned official config and inject external artifact paths."""
    vendor = Path(vendor_root).resolve()
    base_path = (vendor / definition.base_config).resolve()
    if vendor != base_path and vendor not in base_path.parents:
        raise ValueError(f"base config escapes vendor checkout: {base_path}")
    if not base_path.is_file():
        raise FileNotFoundError(base_path)
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if not isinstance(base, dict):
        raise ValueError("official PaddleOCR config must be a YAML mapping")
    resolved = deep_merge(base, definition.overrides)
    global_config = resolved.setdefault("Global", {})
    if not isinstance(global_config, dict):
        raise ValueError("resolved Global config must be a mapping")
    global_config["pretrained_model"] = definition.source_checkpoint.as_posix()
    global_config["checkpoints"] = None
    global_config["save_model_dir"] = (
        Path(output_root).resolve() / "training"
    ).as_posix()
    global_config["save_inference_dir"] = None
    global_config["distributed"] = False
    _refuse_forbidden_training_data(resolved.get("Train", {}))
    return resolved


def _refuse_forbidden_training_data(train_config: Any) -> None:
    serialized = canonical_json(train_config).casefold()
    found = [
        token for token in _FORBIDDEN_TRAINING_TOKENS if token in serialized
    ]
    if found:
        raise ValueError(
            "forbidden training data token(s): " + ", ".join(sorted(found))
        )
