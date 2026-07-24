#!/usr/bin/env python3
"""Build the public-safe versioned OCR registry from executed artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ocr.model_registry import ORIGINAL_MODEL_IDS, REQUIRED_MODEL_NAMES  # noqa: E402
from src.rotation_common import atomic_write_json, canonical_json, sha256_file  # noqa: E402


CUSTOM_IDS = {
    "detector": "PP-OCRv6_medium_det_csx4201_v1",
    "general": "PP-OCRv6_medium_rec_csx4201_v1",
    "thai": "th_PP-OCRv5_mobile_rec_csx4201_v1",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-setup",
        default=str(PROJECT_ROOT / "reports/ocr/model_setup.json"),
    )
    parser.add_argument(
        "--pretrained-report",
        default=str(PROJECT_ROOT / "reports/ocr_upgrade/pretrained_training_models.json"),
    )
    parser.add_argument(
        "--detector-trials",
        default=str(PROJECT_ROOT / "reports/ocr_upgrade/detector_trials.csv"),
    )
    parser.add_argument("--general-trial-root")
    parser.add_argument("--general-report")
    parser.add_argument(
        "--general-selected",
        choices=("original", "custom"),
        default="original",
    )
    parser.add_argument(
        "--thai-trial-root",
        default=(
            "D:/CSX4201/vision-info-extraction-assets/checkpoints/ocr_upgrade/"
            "thai_recognizer/rec_thai_synthetic"
        ),
    )
    parser.add_argument(
        "--thai-report",
        default=str(
            PROJECT_ROOT
            / "reports/ocr_upgrade/recognizer_trials/rec_thai_synthetic.json"
        ),
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "reports/ocr_upgrade/model_registry.json"),
    )
    args = parser.parse_args()

    registry = build_registry(
        model_setup=Path(args.model_setup),
        pretrained_report=Path(args.pretrained_report),
        detector_trials=Path(args.detector_trials),
        general_trial_root=Path(args.general_trial_root)
        if args.general_trial_root
        else None,
        general_report=Path(args.general_report) if args.general_report else None,
        general_selected=args.general_selected,
        thai_trial_root=Path(args.thai_trial_root),
        thai_report=Path(args.thai_report),
    )
    atomic_write_json(Path(args.output), registry)
    print(json.dumps(registry, indent=2, ensure_ascii=False))
    return 0


def build_registry(
    *,
    model_setup: Path,
    pretrained_report: Path,
    detector_trials: Path,
    general_trial_root: Path | None,
    general_report: Path | None,
    general_selected: str,
    thai_trial_root: Path,
    thai_report: Path,
) -> dict[str, Any]:
    setup = _json(model_setup)
    pretrained = _json(pretrained_report)
    setup_models = setup.get("models")
    if not isinstance(setup_models, Mapping):
        raise ValueError("model setup has no models mapping")
    entries: dict[str, Any] = {}
    for upstream in REQUIRED_MODEL_NAMES:
        source = setup_models.get(upstream)
        if not isinstance(source, Mapping):
            raise ValueError(f"model setup is missing {upstream}")
        files = [dict(value) for value in source.get("files") or []]
        model_hash = _manifest_hash(files)
        role = "detector" if upstream.endswith("_det") else "recognizer"
        language = (
            "thai"
            if upstream.startswith("th_")
            else "multilingual"
            if role == "detector"
            else "general"
        )
        pretrained_item = _pretrained_item(pretrained, upstream)
        entries[ORIGINAL_MODEL_IDS[upstream]] = {
            "available": True,
            "accepted": True,
            "selected": (
                role == "detector"
                or (language == "general" and general_selected == "original")
            ),
            "default": (
                role == "detector"
                or (language == "general" and general_selected == "original")
            ),
            "variant": "original",
            "role": role,
            "language": language,
            "model_name": ORIGINAL_MODEL_IDS[upstream],
            "upstream_base": upstream,
            "local_path": str(source["resolved_path"]),
            "files": files,
            "model_sha256": model_hash,
            "export_date": _report_date(setup, pretrained),
            "source_training_config": pretrained_item.get("config"),
            "training_data_manifest_sha256": None,
            "training_data_manifest_note": (
                "The upstream publisher did not provide a training-data manifest; "
                "this registry binds the downloaded official checkpoint and local inference export."
            ),
            "license": "Apache-2.0",
            "intended_domain": (
                "upstream multilingual document text detection"
                if role == "detector"
                else f"upstream {language} text recognition"
            ),
            "development_metrics": _baseline_metrics_for(upstream),
            "source_checkpoint_sha256": pretrained_item.get("sha256"),
        }

    entries[CUSTOM_IDS["detector"]] = {
        "available": False,
        "accepted": False,
        "selected": False,
        "default": False,
        "variant": "custom",
        "role": "detector",
        "language": "multilingual",
        "model_name": CUSTOM_IDS["detector"],
        "upstream_base": "PP-OCRv6_medium_det",
        "local_path": None,
        "files": [],
        "model_sha256": None,
        "export_date": None,
        "source_training_config": "configs/ocr_upgrade/detector",
        "training_data_manifest_sha256": _training_manifest_sha(
            PROJECT_ROOT / "reports/ocr_upgrade/training_data_summary.json"
        ),
        "license": "Apache-2.0",
        "intended_domain": "public financial document text detection",
        "development_metrics": {},
        "unavailable_reason": (
            "The bounded 8 GiB GPU trials produced no valid accepted checkpoint; "
            "the verified original detector remains selected."
        ),
        "trial_evidence": detector_trials.as_posix(),
    }

    if general_trial_root is not None or general_report is not None:
        if general_trial_root is None or general_report is None:
            raise ValueError(
                "general trial root and general report must be provided together"
            )
        entries[CUSTOM_IDS["general"]] = _custom_entry(
            model_id=CUSTOM_IDS["general"],
            upstream="PP-OCRv6_medium_rec",
            language="general",
            domain="public English/Turkish financial document recognition",
            trial_root=general_trial_root,
            report_path=general_report,
            selected=general_selected == "custom",
            accepted=general_selected == "custom",
            license_id="Apache-2.0",
        )

    entries[CUSTOM_IDS["thai"]] = _custom_entry(
        model_id=CUSTOM_IDS["thai"],
        upstream="th_PP-OCRv5_mobile_rec",
        language="thai",
        domain=(
            "Thai financial-domain recognition; trained and selected on "
            "OFL-font synthetic evidence only"
        ),
        trial_root=thai_trial_root,
        report_path=thai_report,
        selected=True,
        accepted=True,
        license_id="Apache-2.0; synthetic fonts OFL-1.1",
    )

    defaults = {
        "detector": "original",
        "general_recognizer": general_selected,
        "thai_recognizer": "custom",
    }
    generated = datetime.now(timezone.utc).isoformat()
    registry_material = {
        "schema_version": "2.0",
        "generated_at_utc": generated,
        "source_commit": _git_commit(),
        "defaults": defaults,
        "models": entries,
        "private_row_count": 0,
    }
    registry_material["configuration_sha256"] = hashlib.sha256(
        canonical_json(
            {"defaults": defaults, "models": entries}
        ).encode("utf-8")
    ).hexdigest()
    return registry_material


def _custom_entry(
    *,
    model_id: str,
    upstream: str,
    language: str,
    domain: str,
    trial_root: Path,
    report_path: Path,
    selected: bool,
    accepted: bool,
    license_id: str,
) -> dict[str, Any]:
    metadata = _json(trial_root / "run_metadata.json")
    report = _json(report_path)
    exported = trial_root / "exported"
    files = [
        {
            "path": value.name,
            "size_bytes": value.stat().st_size,
            "sha256": sha256_file(value),
        }
        for value in sorted(exported.glob("inference.*"))
        if value.is_file()
    ]
    if not files:
        raise ValueError(f"custom model export is empty: {exported}")
    if metadata.get("status") != "passed":
        raise ValueError(f"custom model trial did not pass: {trial_root}")
    if int(report.get("private_row_count", -1)) != 0:
        raise ValueError(f"custom model report has private rows: {report_path}")
    return {
        "available": True,
        "accepted": bool(accepted),
        "selected": bool(selected),
        "default": bool(selected),
        "variant": "custom",
        "role": "recognizer",
        "language": language,
        "model_name": model_id,
        "upstream_base": upstream,
        "local_path": exported.as_posix(),
        "files": files,
        "model_sha256": _manifest_hash(files),
        "export_date": report.get("generated_at_utc"),
        "source_training_config": metadata.get("definition_path"),
        "training_data_manifest_sha256": report.get("manifest_sha256"),
        "license": license_id,
        "intended_domain": domain,
        "development_metrics": dict(report.get("metrics") or {}),
        "checkpoint_sha256": metadata.get("selected_checkpoint_sha256"),
        "source_checkpoint_sha256": metadata.get("source_checkpoint_sha256"),
        "source_commit": metadata.get("source_commit"),
        "vendor_commit": metadata.get("vendor_commit"),
        "private_row_count": 0,
        "claim_boundary": (
            metadata.get("metadata", {}).get("claim_boundary")
            if isinstance(metadata.get("metadata"), Mapping)
            else None
        ),
    }


def _manifest_hash(files: list[dict[str, Any]]) -> str:
    """Return the same path-and-content digest as ``sha256_tree``."""
    digest = hashlib.sha256()
    for item in sorted(files, key=lambda value: str(value["path"]).casefold()):
        digest.update(str(item["path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(item["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _pretrained_item(report: Mapping[str, Any], upstream: str) -> dict[str, Any]:
    models = report.get("models") or {}
    for item in models.values():
        if isinstance(item, Mapping) and item.get("model_name") == upstream:
            return dict(item)
    return {}


def _baseline_metrics_for(upstream: str) -> dict[str, Any]:
    mapping = {
        "PP-OCRv6_medium_det": PROJECT_ROOT
        / "reports/ocr_upgrade/detector_trials/det_baseline_official.json",
        "PP-OCRv6_medium_rec": PROJECT_ROOT
        / "reports/ocr_upgrade/recognizer_trials/rec_general_baseline_official.json",
        "th_PP-OCRv5_mobile_rec": PROJECT_ROOT
        / "reports/ocr_upgrade/recognizer_trials/rec_thai_baseline_official.json",
    }
    path = mapping[upstream]
    return dict(_json(path).get("metrics") or {}) if path.is_file() else {}


def _training_manifest_sha(path: Path) -> str | None:
    if not path.is_file():
        return None
    payload = _json(path)
    return str(payload.get("manifest_sha256") or sha256_file(path))


def _report_date(*values: Mapping[str, Any]) -> str | None:
    for value in values:
        for key in ("generated_at_utc", "generated_at"):
            if value.get(key):
                return str(value[key])
    return None


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()


if __name__ == "__main__":
    raise SystemExit(main())
