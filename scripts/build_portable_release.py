#!/usr/bin/env python3
"""Build the private-data-free, relocatable OCR_Model distribution."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.release_provenance import require_clean_git_worktree  # noqa: E402

DEFAULT_ASSET_ROOT = Path("D:/OCR_Model_Assets")
INSTALLED_TARGET = Path("D:/OCR_Model")
DEFAULT_TARGET = (
    DEFAULT_ASSET_ROOT / "release-staging" / "v1.1.0-in-place" / "OCR_Model"
)
PORTABLE_LAYOUT_CHECKPOINT = "assets/checkpoints/layoutxlm_multitask/final"
OCR_MODEL_NAMES = (
    "PP-OCRv6_medium_det",
    "PP-OCRv6_medium_rec",
    "th_PP-OCRv5_mobile_rec",
)
SECRET_PATTERNS = (
    re.compile(
        rb"(?i)(?:api[_-]?key|client[_-]?secret|password|access[_-]?token)"
        rb"\s*[:=]\s*['\"]?[A-Za-z0-9_\-/.+=]{16,}"
    ),
    re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(rb"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
    re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def copy_file(source: Path, target: Path) -> None:
    _assert_no_reparse_components(source)
    if _is_reparse_point(source):
        raise ValueError(f"refusing to copy a reparse point: {source}")
    if not source.is_file():
        raise FileNotFoundError(source)
    source_size = source.stat().st_size
    source_hash = sha256_file(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if (
        source.stat().st_size != source_size
        or sha256_file(source) != source_hash
        or target.stat().st_size != source_size
        or sha256_file(target) != source_hash
    ):
        target.unlink(missing_ok=True)
        raise RuntimeError(f"source changed or copy verification failed: {source}")


def copy_tree(source: Path, target: Path) -> None:
    _assert_no_reparse_components(source)
    if _is_reparse_point(source):
        raise ValueError(f"refusing to copy a reparse directory: {source}")
    if not source.is_dir():
        raise FileNotFoundError(source)
    target.mkdir(parents=True, exist_ok=True)
    _copy_tree_no_follow(source, target)


def _copy_tree_no_follow(source: Path, target: Path) -> None:
    ignored_names = {"__pycache__", ".pytest_cache"}
    with os.scandir(source) as entries:
        ordered = sorted(entries, key=lambda entry: entry.name.casefold())
    for entry in ordered:
        if entry.name in ignored_names or entry.name.endswith(".pyc"):
            continue
        source_path = Path(entry.path)
        target_path = target / entry.name
        if entry.is_symlink() or _is_reparse_point(source_path):
            raise ValueError(f"refusing to follow a reparse point: {source_path}")
        if entry.is_dir(follow_symlinks=False):
            target_path.mkdir(parents=True, exist_ok=True)
            _copy_tree_no_follow(source_path, target_path)
        elif entry.is_file(follow_symlinks=False):
            copy_file(source_path, target_path)
        else:
            raise ValueError(f"refusing to copy a special file: {source_path}")


def selected_layout_checkpoint(
    asset_root: Path,
) -> tuple[Path, str, dict[str, Any]]:
    """Resolve and hash the frozen checkpoint bound to the project calibration."""
    cfg = yaml.safe_load(
        (PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8")
    )
    configured = str(
        cfg.get("layout_model", {}).get("inference_checkpoint", "")
    ).strip()
    if configured:
        checkpoint = Path(configured).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = PROJECT_ROOT / checkpoint
    else:
        checkpoint = asset_root / "checkpoints" / "layoutxlm_multitask" / "final"
    checkpoint = checkpoint.resolve()
    model_path = checkpoint / "model.safetensors"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    actual_hash = sha256_file(model_path)

    calibration_path = PROJECT_ROOT / "models" / "multitask_calibration.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    expected_hash = str(calibration.get("checkpoint_model_sha256", "")).strip()
    if expected_hash != actual_hash:
        raise ValueError(
            "selected checkpoint is not bound to the current calibration: "
            f"{actual_hash} != {expected_hash or '<missing>'}"
        )
    return checkpoint, actual_hash, calibration


def validate_build_location(target: Path, asset_root: Path) -> None:
    """Constrain release builds to the owner-authorized external staging root."""
    resolved = target.resolve()
    installed = INSTALLED_TARGET.resolve()
    assets = DEFAULT_ASSET_ROOT.resolve()
    if asset_root.resolve() != assets:
        raise ValueError(
            "portable asset root must be exactly "
            "D:\\OCR_Model_Assets"
        )
    if resolved == installed:
        raise ValueError(
            "release builder must never target the installed D:\\OCR_Model "
            "working copy; use external-assets release staging"
        )
    if assets not in resolved.parents:
        raise ValueError(
            "portable release builds are allowed only below "
            "D:\\OCR_Model_Assets"
        )


def require_unchanged_provenance(
    initial: dict[str, Any],
    final: dict[str, Any],
) -> None:
    """Fail if release-source state changed while the payload was assembled."""
    keys = (
        "source_commit",
        "source_tree_dirty",
        "source_tree_sha256",
        "source_candidate_file_count",
        "source_missing_candidate_paths",
    )
    changed = [key for key in keys if initial.get(key) != final.get(key)]
    if changed:
        raise RuntimeError(
            "release source changed during payload assembly: "
            + ", ".join(changed)
        )


def prepare_target(target: Path) -> None:
    resolved = Path(os.path.abspath(target))
    if resolved.name != "OCR_Model":
        raise ValueError("portable target directory must be named exactly OCR_Model")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_reparse_components(resolved.parent)
    if resolved.exists():
        raise FileExistsError(
            f"target already exists: {resolved}; choose a new isolated "
            "release-staging target"
        )
    resolved.mkdir(parents=True)


def copy_application(target: Path) -> None:
    for directory in ("src", "schemas", ".agents"):
        copy_tree(PROJECT_ROOT / directory, target / directory)
    copy_tree(PROJECT_ROOT / "models" / "kmeans_rotation", target / "models" / "kmeans_rotation")
    scripts_target = target / "scripts"
    scripts_target.mkdir(parents=True, exist_ok=True)
    for name in (
        "extract_document.py",
        "layout_entity_worker.py",
        "setup_portable_windows.ps1",
    ):
        copy_file(PROJECT_ROOT / "scripts" / name, scripts_target / name)
    for name in (
        "LICENSE",
        "CONTRIBUTING.md",
        "app.py",
        "doctor.py",
        "mcp_server.py",
        "run_ocr.py",
        "requirements-app.txt",
        "requirements-ie.txt",
        "requirements-layout.txt",
        "requirements-ocr.txt",
        "setup_windows.ps1",
        "setup_windows.bat",
        "launch_windows.bat",
        "run_cli.bat",
        "install_codex_integration.ps1",
        "Dockerfile",
        "docker-compose.yml",
        ".dockerignore",
        "launch_macos.command",
        "install_codex_integration_macos.command",
    ):
        copy_file(PROJECT_ROOT / name, target / name)
    (target / "extract_document.py").write_text(
        "#!/usr/bin/env python3\n"
        '"""Run the complete local OCR + information-extraction model."""\n'
        "from src.portable.cli import main\n\n"
        'if __name__ == "__main__":\n'
        "    raise SystemExit(main())\n",
        encoding="utf-8",
    )
    for name in (
        "PORTABLE_USAGE.md",
        "CODEX_INTEGRATION.md",
        "THIRD_PARTY_NOTICES.md",
    ):
        copy_file(PROJECT_ROOT / "docs" / name, target / "docs" / name)
    copy_file(PROJECT_ROOT / "docs" / "PORTABLE_USAGE.md", target / "README.md")


def portable_config(target: Path) -> None:
    cfg = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    paths = cfg["paths"]
    paths.update(
        {
            "project_root": ".",
            "sroie": "data/not-included/public/sroie",
            "funsd": "data/not-included/public/funsd",
            "fatura": "data/not-included/public/fatura",
            "coru": "data/not-included/public/coru",
            "gmail_receipts": "data/not-included/private/gmail/receipts",
            "gmail_invoices": "data/not-included/private/gmail/invoices",
            "gmail_legal_financial": "data/not-included/private/gmail/legal_financial_docs",
            "gmail_unclassified": "data/not-included/private/gmail/unclassified",
            "metadata": "data/not-included/metadata",
            "processed": "data/not-included/processed",
            "page_images": "data/not-included/page_images",
            "private_page_images": "data/not-included/private/page_images",
            "rotated_images": "data/not-included/rotated_images",
            "features": "data/not-included/features",
            "splits": "data/not-included/splits",
            "rotation_models": "models/kmeans_rotation",
            "reports": "reports",
            "external_assets": "assets",
            "ocr_environment": ".runtime/ocr",
            "layout_environment": ".runtime/layout",
            "layout_python": ".runtime/layout/Scripts/python.exe",
            "paddle_cache": "assets/cache/paddlex",
            "huggingface_cache": "assets/cache/huggingface",
            "ocr_models": "assets/ocr_models",
            "layout_models": "assets/cache/layoutxlm",
            "ie_checkpoints": "assets/checkpoints",
            "ocr_cache": "assets/cache/ocr",
            "model_datasets": "data/not-included/model_datasets",
            "generated_documents": "outputs",
            "private_outputs": "outputs/private",
        }
    )
    cfg["ocr"]["device"] = "cpu"
    cfg["ocr"]["paddle_cache_home"] = "assets/cache/paddlex"
    cfg["ocr"]["detector"]["path"] = "assets/ocr_models/PP-OCRv6_medium_det"
    cfg["ocr"]["general_recognizer"]["path"] = "assets/ocr_models/PP-OCRv6_medium_rec"
    cfg["ocr"]["thai_recognizer"]["path"] = "assets/ocr_models/th_PP-OCRv5_mobile_rec"
    cfg["layout_model"]["inference_checkpoint"] = (
        PORTABLE_LAYOUT_CHECKPOINT
    )
    (target / "config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    write_json(
        target / "runtime.json",
        {
            "schema_version": "1.0",
            "config": "config.yaml",
            "ocr_python": ".runtime/ocr/Scripts/python.exe",
            "layout_python": ".runtime/layout/Scripts/python.exe",
            "model_setup": "reports/ocr/model_setup.json",
            "layout_checkpoint": PORTABLE_LAYOUT_CHECKPOINT,
            "asset_root": "assets",
            "output_root": "outputs",
            "device": "cpu",
            "uses_openai_api": False,
        },
    )


def copy_models(target: Path, asset_root: Path) -> list[dict[str, Any]]:
    checkpoint_source, actual_hash, calibration = selected_layout_checkpoint(
        asset_root
    )
    model_source = checkpoint_source / "model.safetensors"
    checkpoint_target = target / "assets" / "checkpoints" / "layoutxlm_multitask" / "final"
    checkpoint_target.mkdir(parents=True, exist_ok=True)
    for source in sorted(checkpoint_source.iterdir()):
        if source.is_file():
            if source.name == "training_state.json":
                state = json.loads(source.read_text(encoding="utf-8"))
                state["manifest_path"] = "not included; public training manifest"
                write_json(checkpoint_target / source.name, state)
            else:
                copy_file(source, checkpoint_target / source.name)

    setup_source = json.loads(
        (PROJECT_ROOT / "reports" / "ocr" / "model_setup.json").read_text(
            encoding="utf-8"
        )
    )
    setup_source.update(
        {
            "device": "cpu",
            "asset_root": "../../assets",
            "cache_root": "../../assets/ocr_models",
            "offline": True,
            "portable": True,
        }
    )
    for name in OCR_MODEL_NAMES:
        source = (
            asset_root
            / "cache"
            / "paddlex"
            / "official_models"
            / name
        )
        destination = target / "assets" / "ocr_models" / name
        copy_tree(source, destination)
        setup_source["models"][name]["resolved_path"] = (
            f"../../assets/ocr_models/{name}"
        )
        setup_source["models"][name]["device"] = "cpu"
    write_json(target / "reports" / "ocr" / "model_setup.json", setup_source)
    _copy_upgrade_registry(target)

    calibration["checkpoint"] = PORTABLE_LAYOUT_CHECKPOINT
    calibration["manifest_path"] = "not included; public training manifest"
    write_json(target / "models" / "multitask_calibration.json", calibration)

    records = []
    for path in sorted(
        [
            *checkpoint_target.rglob("*"),
            *(target / "assets" / "ocr_models").rglob("*"),
            *(target / "models" / "kmeans_rotation").rglob("*"),
            target / "models" / "multitask_calibration.json",
        ]
    ):
        if not path.is_file():
            continue
        relative = path.relative_to(target).as_posix()
        if relative.startswith("assets/ocr_models/"):
            role = "ocr_model"
            license_name = "Apache-2.0"
        elif relative.endswith("model.safetensors"):
            role = "layout_checkpoint"
            license_name = "CC-BY-NC-SA-4.0"
        elif relative.startswith("assets/checkpoints/"):
            role = "layout_checkpoint_metadata"
            license_name = "CC-BY-NC-SA-4.0"
        elif relative.startswith("models/kmeans_rotation/"):
            role = "display_only_rotation_artifact"
            license_name = "project"
        else:
            role = "calibration"
            license_name = "project"
        records.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "role": role,
                "license": license_name,
            }
        )
    write_json(
        target / "MODEL_MANIFEST.json",
        {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "final_layout_model_sha256": actual_hash,
            "files": records,
        },
    )
    return records


def _copy_upgrade_registry(target: Path) -> None:
    source_path = PROJECT_ROOT / "reports" / "ocr_upgrade" / "model_registry.json"
    registry = json.loads(source_path.read_text(encoding="utf-8"))
    models = registry.get("models")
    if not isinstance(models, dict):
        raise ValueError("OCR upgrade registry has no models mapping")
    included: list[str] = []
    for model_id, raw in models.items():
        if not isinstance(raw, dict):
            raise ValueError(f"invalid OCR registry entry: {model_id}")
        upstream = str(raw.get("upstream_base", ""))
        variant = str(raw.get("variant", "")).casefold()
        available = bool(raw.get("available", False))
        selected = bool(raw.get("selected", False))
        accepted = bool(raw.get("accepted", False))
        if variant == "original" and available:
            destination_name = upstream
        elif variant == "custom" and available and selected and accepted:
            source = Path(str(raw.get("local_path", "")))
            if not source.is_dir():
                raise FileNotFoundError(source)
            destination_name = str(model_id)
            copy_tree(
                source,
                target / "assets" / "ocr_models" / destination_name,
            )
        else:
            raw["available"] = False
            raw["selected"] = False
            raw["local_path"] = ""
            raw["portable_included"] = False
            raw["portable_exclusion_reason"] = (
                "not selected and accepted for final portable inference"
            )
            continue
        raw["local_path"] = f"../../assets/ocr_models/{destination_name}"
        raw["portable_included"] = True
        included.append(str(model_id))
    registry["portable"] = True
    registry["portable_included_model_ids"] = sorted(included)
    registry["portable_policy"] = (
        "all original fallbacks plus selected accepted custom inference models"
    )
    write_json(
        target / "reports" / "ocr_upgrade" / "model_registry.json",
        registry,
    )


def copy_samples(target: Path, asset_root: Path) -> None:
    fixture_root = asset_root / "generated" / "integration_smoke" / "fixtures"
    fixture = fixture_root / "unknown_upright.png"
    report_path = (
        PROJECT_ROOT
        / "reports"
        / "information_extraction"
        / "integration_smoke.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    record = (
        report.get("fixture_artifacts", {})
        .get("unknown_upright_image")
    )
    if not isinstance(record, dict):
        raise ValueError("integration evidence has no upright sample binding")
    recorded = Path(str(record.get("path", "")))
    if (
        Path(os.path.abspath(recorded)) != Path(os.path.abspath(fixture))
        or int(record.get("size_bytes", -1)) != fixture.stat().st_size
        or str(record.get("sha256", "")) != sha256_file(fixture)
    ):
        raise ValueError(
            "synthetic sample does not match integration evidence"
        )
    copy_file(fixture, target / "samples" / "unknown_upright.png")
    (target / "samples" / "README.md").write_text(
        "# Synthetic sample\n\n"
        "`unknown_upright.png` is a generated integration fixture containing "
        "no real person, account, or private document data. Use it for the "
        "first CLI/GUI run and Devpost screenshots.\n",
        encoding="utf-8",
    )


def privacy_audit(target: Path) -> dict[str, Any]:
    files, reparse_points, special_paths = _payload_inventory(target)
    prohibited: list[str] = []
    prohibited.extend(special_paths)
    for path in files:
        relative = path.relative_to(target)
        lowered = [part.casefold() for part in relative.parts]
        if relative.name.casefold() in {
            ".env",
            "runtime.local.json",
            "private_file_inventory.csv",
        }:
            prohibited.append(relative.as_posix())
        if lowered and lowered[0] == "data":
            prohibited.append(relative.as_posix())
        if any(part in {"private_outputs", "private-evaluation"} for part in lowered):
            prohibited.append(relative.as_posix())

    private_names = _private_inventory_names(
        PROJECT_ROOT / "data" / "metadata" / "private_file_inventory.csv"
    )
    private_needles = _private_name_needles(private_names)
    secret_hits: list[str] = []
    private_name_hits: list[str] = []
    for path in files:
        try:
            secret_hit, private_name_hit = _scan_payload_file(
                path,
                private_needles,
            )
        except OSError:
            prohibited.append(path.relative_to(target).as_posix())
            continue
        relative = path.relative_to(target).as_posix()
        if secret_hit:
            secret_hits.append(relative)
        if (
            private_name_hit
            or any(name in relative.casefold() for name in private_names)
        ):
            private_name_hits.append(relative)

    failed = bool(
        prohibited or reparse_points or secret_hits or private_name_hits
    )
    prohibited_lower = [value.casefold() for value in prohibited]
    audit = {
        "status": "fail" if failed else "pass",
        "checked_file_count": len(files),
        "prohibited_files": sorted(set(prohibited)),
        "reparse_points": sorted(set(reparse_points)),
        "secret_hit_files": sorted(set(secret_hits)),
        "private_filename_hit_files": sorted(set(private_name_hits)),
        "private_filename_inventory_count": len(private_names),
        "raw_data_included": any(
            value == "data" or value.startswith("data/")
            for value in prohibited_lower
        ),
        "private_gmail_data_included": bool(private_name_hits),
        "private_outputs_included": any(
            "private_outputs" in value or "private-evaluation" in value
            for value in prohibited_lower
        ),
        "credentials_included": bool(secret_hits),
        "safe_sample_count": 1,
        "method": (
            "allowlisted copy plus final-payload prohibited-path, reparse, "
            "streaming all-file secret-pattern and live private-filename scans"
        ),
    }
    write_json(target / "PRIVACY_AUDIT.json", audit)
    if failed:
        raise ValueError(
            "privacy audit failed; inspect PRIVACY_AUDIT.json for hit paths"
        )
    return audit


def _private_inventory_names(inventory_path: Path) -> set[str]:
    if not inventory_path.is_file():
        raise FileNotFoundError(
            "private filename inventory is required for the release scan"
        )
    names: set[str] = set()
    with inventory_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            for key in (
                "original_filename",
                "current_relative_path",
                "original_relative_path",
                "relative_path",
                "path",
                "filename",
            ):
                value = str(row.get(key) or "").strip()
                if value:
                    names.add(Path(value).name.casefold())
    names.discard("")
    if not names:
        raise ValueError(
            "private filename inventory contains no usable filename values"
        )
    return names


def _private_name_needles(names: set[str]) -> tuple[bytes, ...]:
    needles: set[bytes] = set()
    for name in names:
        for encoding in ("utf-8", "utf-16-le", "utf-16-be"):
            encoded = name.encode(encoding, errors="ignore").lower()
            if encoded:
                needles.add(encoded)
    return tuple(sorted(needles, key=len, reverse=True))


def _scan_payload_file(
    path: Path,
    private_needles: tuple[bytes, ...],
) -> tuple[bool, bool]:
    max_needle = max((len(value) for value in private_needles), default=0)
    overlap = max(8192, max_needle * 2)
    tail = b""
    secret_hit = False
    private_name_hit = False
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            content = tail + chunk
            lowered = content.lower()
            if not secret_hit:
                secret_hit = any(
                    pattern.search(content) for pattern in SECRET_PATTERNS
                )
            if not private_name_hit:
                private_name_hit = any(
                    needle in lowered for needle in private_needles
                )
            if secret_hit and private_name_hit:
                break
            tail = content[-overlap:]
    return secret_hit, private_name_hit


def _is_reparse_point(path: Path) -> bool:
    if path.is_symlink():
        return True
    attributes = getattr(path.lstat(), "st_file_attributes", 0)
    return bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _assert_no_reparse_components(path: Path) -> None:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if (current.exists() or current.is_symlink()) and _is_reparse_point(
            current
        ):
            raise ValueError(f"path contains a reparse point: {current}")


def _payload_inventory(
    target: Path,
) -> tuple[list[Path], list[str], list[str]]:
    files: list[Path] = []
    reparse_points: list[str] = []
    special_paths: list[str] = []
    pending = [target]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            ordered = sorted(entries, key=lambda entry: entry.name.casefold())
        for entry in ordered:
            path = Path(entry.path)
            relative = path.relative_to(target).as_posix()
            if entry.is_symlink() or _is_reparse_point(path):
                reparse_points.append(relative)
            elif entry.is_dir(follow_symlinks=False):
                pending.append(path)
            elif entry.is_file(follow_symlinks=False):
                files.append(path)
            else:
                special_paths.append(relative)
    files.sort(key=lambda path: path.relative_to(target).as_posix())
    return files, sorted(reparse_points), sorted(special_paths)


def write_payload_manifest(target: Path) -> dict[str, Any]:
    manifest_path = target / "PAYLOAD_MANIFEST.json"
    if manifest_path.exists():
        manifest_path.unlink()
    files, reparse_points, special_paths = _payload_inventory(target)
    if reparse_points or special_paths:
        raise ValueError(
            "cannot create payload manifest with reparse or special paths"
        )
    records = [
        {
            "path": path.relative_to(target).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    payload = {
        "schema_version": "1.0",
        "self_excluded": "PAYLOAD_MANIFEST.json",
        "file_count": len(records),
        "total_size_bytes": sum(record["size_bytes"] for record in records),
        "files": records,
    }
    write_json(manifest_path, payload)
    return payload


def validate_payload_manifest(target: Path) -> dict[str, Any]:
    manifest_path = target / "PAYLOAD_MANIFEST.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("payload manifest is missing or invalid") from exc
    records = manifest.get("files")
    if (
        not isinstance(records, list)
        or int(manifest.get("file_count", -1)) != len(records)
        or manifest.get("self_excluded") != "PAYLOAD_MANIFEST.json"
    ):
        raise ValueError("payload manifest structure is invalid")
    expected: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("payload manifest contains a non-object record")
        relative = str(record.get("path", ""))
        if (
            not relative
            or relative == "PAYLOAD_MANIFEST.json"
            or relative in expected
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise ValueError(f"payload manifest path is unsafe: {relative!r}")
        expected[relative] = record
    files, reparse_points, special_paths = _payload_inventory(target)
    if reparse_points or special_paths:
        raise ValueError(
            "payload contains reparse points or special files after manifesting"
        )
    actual_paths = {
        path.relative_to(target).as_posix()
        for path in files
        if path != manifest_path
    }
    if actual_paths != set(expected):
        raise ValueError("payload file set changed after manifest creation")
    for relative, record in expected.items():
        path = target / relative
        if (
            int(record.get("size_bytes", -1)) != path.stat().st_size
            or str(record.get("sha256", "")) != sha256_file(path)
        ):
            raise ValueError(
                f"payload file changed after manifest creation: {relative}"
            )
    expected_total = sum(
        int(record["size_bytes"]) for record in expected.values()
    )
    if int(manifest.get("total_size_bytes", -1)) != expected_total:
        raise ValueError("payload manifest total size is invalid")
    return manifest


def validate_zip_payload_manifest(
    archive: Path,
    *,
    expected_root: str,
    manifest: dict[str, Any],
    manifest_path: Path,
) -> dict[str, Any]:
    expected = {
        str(record["path"]): {
            "size_bytes": int(record["size_bytes"]),
            "sha256": str(record["sha256"]),
        }
        for record in manifest["files"]
    }
    expected["PAYLOAD_MANIFEST.json"] = {
        "size_bytes": manifest_path.stat().st_size,
        "sha256": sha256_file(manifest_path),
    }
    observed: dict[str, dict[str, Any]] = {}
    prefix = f"{expected_root}/"
    with zipfile.ZipFile(archive, "r") as source:
        for info in source.infolist():
            if info.is_dir():
                continue
            normalized = info.filename.replace("\\", "/")
            if not normalized.startswith(prefix):
                raise ValueError(
                    f"ZIP payload entry escaped expected root: {info.filename}"
                )
            relative = normalized[len(prefix):]
            digest = hashlib.sha256()
            size = 0
            with source.open(info, "r") as stream:
                for chunk in iter(
                    lambda: stream.read(4 * 1024 * 1024),
                    b"",
                ):
                    digest.update(chunk)
                    size += len(chunk)
            observed[relative] = {
                "size_bytes": size,
                "sha256": digest.hexdigest(),
            }
    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        changed = sorted(
            relative
            for relative in set(expected) & set(observed)
            if expected[relative] != observed[relative]
        )
        raise ValueError(
            "ZIP payload does not match PAYLOAD_MANIFEST.json: "
            f"missing={missing}, extra={extra}, changed={changed}"
        )
    return {
        "manifest_file_count": len(expected),
        "manifest_match": True,
    }


def build_info(
    target: Path,
    model_records: list[dict[str, Any]],
    provenance: dict[str, Any],
) -> None:
    write_json(
        target / "BUILD_INFO.json",
        {
            "schema_version": "1.0",
            "built_at": datetime.now(timezone.utc).isoformat(),
            "source_repository": "OCR Model",
            "source_commit": provenance["source_commit"],
            "source_tree_dirty_at_build": provenance["source_tree_dirty"],
            "source_tree_sha256": provenance["source_tree_sha256"],
            "source_candidate_file_count": provenance[
                "source_candidate_file_count"
            ],
            "model_file_count": len(model_records),
            "model_size_bytes": sum(item["size_bytes"] for item in model_records),
            "supported_hosts": {
                "windows": "native Python 3.10 setup; CPU or compatible NVIDIA GPU",
                "macos": "Docker Desktop CPU runtime using linux/amd64 emulation",
            },
            "physical_mac_tested": False,
            "uses_openai_api": False,
        },
    )


def validate_zip_archive(archive: Path, expected_root: str) -> dict[str, Any]:
    with zipfile.ZipFile(archive, "r") as source:
        names = source.namelist()
        corrupt = source.testzip()
    duplicates = sorted(
        name for name in set(names) if names.count(name) > 1
    )
    unsafe = []
    roots = set()
    for name in names:
        normalized = name.replace("\\", "/")
        parts = Path(normalized).parts
        if (
            not parts
            or normalized.startswith("/")
            or re.match(r"^[A-Za-z]:", normalized)
            or any(part in {"", ".", ".."} for part in parts)
        ):
            unsafe.append(name)
            continue
        roots.add(parts[0])
    if corrupt or duplicates or unsafe or roots != {expected_root}:
        raise ValueError(
            "ZIP validation failed: "
            f"corrupt={corrupt!r}, duplicates={len(duplicates)}, "
            f"unsafe={len(unsafe)}, roots={sorted(roots)}"
        )
    return {
        "entry_count": len(names),
        "root": expected_root,
        "crc_passed": True,
        "duplicate_count": 0,
        "unsafe_path_count": 0,
    }


def make_zip(target: Path) -> tuple[Path, str, dict[str, Any]]:
    payload_manifest = validate_payload_manifest(target)
    manifest_path = target / "PAYLOAD_MANIFEST.json"
    archive = target.with_suffix(".zip")
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(
        archive,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as output:
        manifest_files = [
            target / str(record["path"])
            for record in payload_manifest["files"]
        ]
        for path in [*manifest_files, manifest_path]:
            arcname = (Path(target.name) / path.relative_to(target)).as_posix()
            # ZipFile.write streams large model files instead of reading the
            # 1+ GiB checkpoint into memory. macOS instructions invoke the
            # launchers through `bash`, so executable-bit preservation is not
            # required.
            output.write(path, arcname)
    verification = validate_zip_archive(archive, target.name)
    verification.update(
        validate_zip_payload_manifest(
            archive,
            expected_root=target.name,
            manifest=payload_manifest,
            manifest_path=manifest_path,
        )
    )
    digest = sha256_file(archive)
    archive.with_suffix(".zip.sha256").write_text(
        f"{digest}  {archive.name}\n",
        encoding="ascii",
    )
    return archive, digest, verification


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--zip", action="store_true", dest="create_zip")
    args = parser.parse_args()

    target = Path(os.path.abspath(args.target.expanduser()))
    asset_root = Path(os.path.abspath(args.asset_root.expanduser()))
    provenance = require_clean_git_worktree(PROJECT_ROOT)
    validate_build_location(target, asset_root)
    prepare_target(target)
    copy_application(target)
    portable_config(target)
    model_records = copy_models(target, asset_root)
    copy_samples(target, asset_root)
    build_info(target, model_records, provenance)
    privacy = privacy_audit(target)
    payload_manifest = write_payload_manifest(target)
    final_provenance = require_clean_git_worktree(PROJECT_ROOT)
    require_unchanged_provenance(provenance, final_provenance)
    archive = None
    archive_hash = None
    archive_verification = None
    if args.create_zip:
        archive, archive_hash, archive_verification = make_zip(target)
    print(
        json.dumps(
            {
                "status": "complete",
                "target": str(target),
                "file_count": sum(1 for path in target.rglob("*") if path.is_file()),
                "model_size_bytes": sum(item["size_bytes"] for item in model_records),
                "privacy_audit": privacy["status"],
                "payload_manifest_file_count": payload_manifest["file_count"],
                "payload_manifest_total_size_bytes": payload_manifest[
                    "total_size_bytes"
                ],
                "archive": str(archive) if archive else None,
                "archive_sha256": archive_hash,
                "archive_verification": archive_verification,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
