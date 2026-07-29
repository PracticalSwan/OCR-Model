"""Safety and provenance helpers for large OCR corpora built on D:."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from src.ocr.benchmark import resolve_project_input
from src.ocr.training_data import (
    SUPPORTED_PUBLIC_DATASETS,
    TRAINING_SPLITS,
    assert_split_isolation,
)
from src.rotation_common import read_csv_rows, sha256_file


MINIMUM_C_FREE_BYTES = 15 * 1024**3
OUTPUT_MARGIN = 1.35


def load_public_training_rows(
    project_root: Path,
    manifest_path: Path,
) -> list[dict[str, str]]:
    rows = [
        row
        for row in read_csv_rows(manifest_path)
        if row.get("dataset", "").casefold() in SUPPORTED_PUBLIC_DATASETS
        and row.get("project_split", "").casefold() in TRAINING_SPLITS
        and row.get("is_private", "").casefold() == "false"
        and row.get("is_usable", "").casefold() == "true"
    ]
    if any(row.get("dataset", "").casefold() in {"coru", "gmail"} for row in rows):
        raise ValueError("CORU or Gmail entered OCR training rows")
    if any(
        row.get("project_split", "").casefold()
        in {"dev_calibration", "test_in_domain", "unseen_domain_test", "private_test"}
        for row in rows
    ):
        raise ValueError("non-training split entered OCR training rows")
    assert_split_isolation(rows)
    if not rows:
        raise ValueError("no public OCR training rows were selected")
    return sorted(
        rows,
        key=lambda row: (
            row["project_split"],
            row["dataset"],
            row["page_id"],
        ),
    )


def resolve_training_inputs(
    project_root: Path,
    row: dict[str, str],
) -> tuple[Path, Path]:
    _, image_path = resolve_project_input(
        project_root,
        row["image_path"],
        label="source image",
        required_prefix="data/raw/public",
    )
    _, annotation_path = resolve_project_input(
        project_root,
        row["normalized_annotation_path"],
        label="normalized annotation",
        required_prefix="data/processed/normalized_ie_annotations",
        allow_prefix_junction=True,
    )
    actual_hash = sha256_file(image_path)
    if actual_hash != row.get("sha256", "").casefold():
        raise ValueError(f"source hash mismatch for {row['page_id']}")
    return image_path, annotation_path


def training_build_id(
    *,
    kind: str,
    profile: str,
    source_manifest_sha256: str,
    rows: list[dict[str, str]],
    parameters: dict[str, Any],
) -> str:
    material = {
        "kind": kind,
        "profile": profile,
        "source_manifest_sha256": source_manifest_sha256,
        "sources": [
            {
                "page_id": row["page_id"],
                "dataset": row["dataset"],
                "split": row["project_split"],
                "sha256": row["sha256"],
            }
            for row in rows
        ],
        "parameters": parameters,
    }
    digest = hashlib.sha256(
        json.dumps(
            material,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    return f"ocr-{kind}-{digest[:16]}"


def check_storage_gate(
    *,
    project_root: Path,
    output_root: Path,
    anticipated_output_bytes: int,
) -> dict[str, Any]:
    c_free = shutil.disk_usage(project_root.anchor).free
    output_root.parent.mkdir(parents=True, exist_ok=True)
    asset_free = shutil.disk_usage(output_root.parent).free
    required_asset = int(anticipated_output_bytes * OUTPUT_MARGIN)
    errors: list[str] = []
    if c_free < MINIMUM_C_FREE_BYTES:
        errors.append("C: reserve would be below 15 GiB")
    if asset_free < required_asset:
        errors.append("asset volume lacks the required 35% free-space margin")
    result = {
        "minimum_c_free_gib": 15.0,
        "output_margin": OUTPUT_MARGIN,
        "anticipated_output_gib": round(anticipated_output_bytes / 1024**3, 3),
        "required_asset_free_gib": round(required_asset / 1024**3, 3),
        "c_free_gib": round(c_free / 1024**3, 3),
        "asset_free_gib": round(asset_free / 1024**3, 3),
        "passed": not errors,
        "errors": errors,
    }
    if errors:
        raise RuntimeError("; ".join(errors))
    return result


@contextmanager
def output_transaction(
    output_root: Path,
    *,
    expected_leaf: str,
    force: bool,
) -> Iterator[Path]:
    """Build beside a validated target and atomically promote on success."""
    target = output_root.absolute()
    if target.name.casefold() != expected_leaf.casefold():
        raise ValueError(
            f"refusing output root with unexpected leaf {target.name!r}; "
            f"expected {expected_leaf!r}"
        )
    if target.parent == target or str(target.parent) in {target.anchor, ""}:
        raise ValueError(f"refusing broad output root: {target}")
    if target.exists() and not force:
        raise FileExistsError(f"{target} already exists; pass --force to replace it")
    temporary = target.with_name(f".{target.name}.building-{os.getpid()}")
    backup = target.with_name(f".{target.name}.previous-{os.getpid()}")
    for path in (temporary, backup):
        if path.exists():
            shutil.rmtree(path)
    temporary.mkdir(parents=True)
    try:
        yield temporary
        if target.exists():
            target.rename(backup)
        temporary.rename(target)
        if backup.exists():
            shutil.rmtree(backup)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        if backup.exists() and not target.exists():
            backup.rename(target)
        raise


def write_checksums(root: Path) -> int:
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    )
    lines = [
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}"
        for path in paths
    ]
    (root / "checksums.sha256").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return len(paths)


def verify_checksums(root: Path) -> int:
    checksum_path = root / "checksums.sha256"
    if not checksum_path.is_file():
        raise FileNotFoundError(checksum_path)
    count = 0
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if not separator or len(digest) != 64:
            raise ValueError(f"malformed checksum line: {line!r}")
        path = root / relative
        if sha256_file(path) != digest:
            raise ValueError(f"checksum mismatch: {relative}")
        count += 1
    return count
