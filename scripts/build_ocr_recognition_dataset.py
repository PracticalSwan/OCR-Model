#!/usr/bin/env python3
"""Build public line and critical-value PaddleOCR recognition crops on D:."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config as cfgmod  # noqa: E402
from src.ocr.build_support import (  # noqa: E402
    check_storage_gate,
    load_public_training_rows,
    output_transaction,
    resolve_training_inputs,
    training_build_id,
    write_checksums,
)
from src.ocr.training_data import (  # noqa: E402
    build_line_groups,
    build_word_regions,
    critical_fields_by_token,
    rectify_polygon_crop,
    repaired_annotation_tokens,
    validate_transcription,
)
from src.rotation_common import (  # noqa: E402
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    sha256_file,
)


MANIFEST_COLUMNS = (
    "build_id",
    "crop_id",
    "dataset",
    "split",
    "crop_type",
    "critical_fields",
    "transcription",
    "source_page_id",
    "source_document_id",
    "source_image_path",
    "source_image_sha256",
    "normalized_annotation_path",
    "annotation_sha256",
    "token_ids",
    "source_ids",
    "crop_path",
    "crop_sha256",
    "crop_width",
    "crop_height",
    "padding_ratio",
    "is_private",
)
ERROR_COLUMNS = (
    "page_id",
    "dataset",
    "split",
    "crop_type",
    "crop_index",
    "error_type",
    "message",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--profile", choices=["final"], default="final")
    parser.add_argument(
        "--output-root",
        default=(
            "D:/CSX4201/vision-info-extraction-assets/data/recognition_training"
        ),
    )
    parser.add_argument("--line-crops", action="store_true")
    parser.add_argument("--critical-word-crops", action="store_true")
    parser.add_argument("--padding-ratio", type=float, default=0.10)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not args.line_crops and not args.critical_word_crops:
        parser.error("enable --line-crops and/or --critical-word-crops")
    if not 0.0 <= args.padding_ratio <= 0.5:
        parser.error("--padding-ratio must be between 0 and 0.5")

    started = time.monotonic()
    cfg = cfgmod.load_config(args.config)
    root = cfgmod.project_root(cfg)
    metadata = cfgmod.resolve_path(cfg, "metadata")
    source_manifest = metadata / "information_extraction_split_manifest.csv"
    rows = load_public_training_rows(root, source_manifest)
    output_root = Path(args.output_root)
    source_bytes = sum((root / row["image_path"]).stat().st_size for row in rows)
    storage_gate = check_storage_gate(
        project_root=root,
        output_root=output_root,
        anticipated_output_bytes=source_bytes * 3 + 512 * 1024**2,
    )
    parameters = {
        "line_crops": args.line_crops,
        "critical_word_crops": args.critical_word_crops,
        "padding_ratio": args.padding_ratio,
        "format": "PaddleOCR-tab-separated-v1",
        "deduplicate_exact_hash": True,
    }
    build_id = training_build_id(
        kind="recognition-data",
        profile=args.profile,
        source_manifest_sha256=sha256_file(source_manifest),
        rows=rows,
        parameters=parameters,
    )

    with output_transaction(
        output_root,
        expected_leaf="recognition_training",
        force=args.force,
    ) as temporary:
        for role in ("train", "validation"):
            (temporary / role).mkdir(parents=True)
        list_lines: dict[str, list[str]] = {"train": [], "validation": []}
        converted_manifest: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        character_counts: dict[str, Counter[str]] = {
            "train": Counter(),
            "validation": Counter(),
        }
        critical_counts: dict[str, Counter[str]] = {
            "train": Counter(),
            "validation": Counter(),
        }
        seen_hashes: dict[str, set[str]] = {"train": set(), "validation": set()}
        skipped_duplicate_count = 0
        for row in rows:
            role = "train" if row["project_split"] == "train" else "validation"
            source_hash = row["sha256"].casefold()
            if source_hash in seen_hashes[role]:
                skipped_duplicate_count += 1
                continue
            image_path, annotation_path = resolve_training_inputs(root, row)
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
            if annotation.get("page_id") != row["page_id"]:
                raise ValueError("page ID mismatch between split and annotation")
            if str(annotation.get("is_private", "")).casefold() == "true":
                raise ValueError("private annotation entered recognition data")
            tokens = repaired_annotation_tokens(annotation)
            field_map = critical_fields_by_token(annotation)
            candidates: list[dict[str, Any]] = []
            if args.line_crops:
                for group in build_line_groups(tokens):
                    candidates.append(
                        {
                            **group,
                            "crop_type": "line",
                            "critical_fields": sorted(
                                {
                                    field
                                    for token_id in group["token_ids"]
                                    for field in field_map.get(token_id, ())
                                }
                            ),
                        }
                    )
            if args.critical_word_crops:
                for region in build_word_regions(tokens):
                    fields = sorted(
                        {
                            field
                            for token_id in region["token_ids"]
                            for field in field_map.get(token_id, ())
                        }
                    )
                    if fields:
                        candidates.append(
                            {
                                **region,
                                "crop_type": "critical_word",
                                "critical_fields": fields,
                            }
                        )
            annotation_hash = sha256_file(annotation_path)
            with Image.open(image_path) as opened:
                opened.load()
                image = opened.convert("RGB")
                page = annotation.get("page", {})
                if (int(page.get("width", 0)), int(page.get("height", 0))) != image.size:
                    raise ValueError("annotation dimensions do not match image")
                for index, candidate in enumerate(candidates):
                    try:
                        transcription = validate_transcription(
                            candidate["transcription"]
                        )
                        polygon = _quad(candidate["polygon"])
                        crop = rectify_polygon_crop(
                            image,
                            polygon,
                            padding_ratio=args.padding_ratio,
                        )
                        _validate_crop(crop)
                    except (OSError, ValueError) as exc:
                        errors.append(
                            {
                                "page_id": row["page_id"],
                                "dataset": row["dataset"],
                                "split": row["project_split"],
                                "crop_type": candidate.get("crop_type", ""),
                                "crop_index": str(index),
                                "error_type": type(exc).__name__,
                                "message": str(exc),
                            }
                        )
                        continue
                    crop_material = "|".join(
                        (
                            build_id,
                            row["page_id"],
                            candidate["crop_type"],
                            str(index),
                            transcription,
                        )
                    )
                    crop_id = "rec_" + hashlib.sha256(
                        crop_material.encode("utf-8")
                    ).hexdigest()[:20]
                    relative_crop = (
                        Path(role)
                        / row["dataset"]
                        / row["page_id"]
                        / f"{crop_id}.png"
                    )
                    crop_path = temporary / relative_crop
                    crop_path.parent.mkdir(parents=True, exist_ok=True)
                    crop.save(crop_path, format="PNG", optimize=False)
                    with Image.open(crop_path) as reloaded:
                        reloaded.load()
                        if reloaded.size != crop.size:
                            raise RuntimeError(f"crop reload mismatch: {crop_id}")
                    crop_hash = sha256_file(crop_path)
                    list_lines[role].append(
                        f"{relative_crop.as_posix()}\t{transcription}"
                    )
                    character_counts[role].update(transcription)
                    for field in candidate["critical_fields"]:
                        critical_counts[role][field] += 1
                    converted_manifest.append(
                        {
                            "build_id": build_id,
                            "crop_id": crop_id,
                            "dataset": row["dataset"],
                            "split": role,
                            "crop_type": candidate["crop_type"],
                            "critical_fields": "|".join(
                                candidate["critical_fields"]
                            ),
                            "transcription": transcription,
                            "source_page_id": row["page_id"],
                            "source_document_id": row["document_id"],
                            "source_image_path": row["image_path"],
                            "source_image_sha256": source_hash,
                            "normalized_annotation_path": row[
                                "normalized_annotation_path"
                            ],
                            "annotation_sha256": annotation_hash,
                            "token_ids": "|".join(candidate["token_ids"]),
                            "source_ids": "|".join(candidate["source_ids"]),
                            "crop_path": relative_crop.as_posix(),
                            "crop_sha256": crop_hash,
                            "crop_width": crop.width,
                            "crop_height": crop.height,
                            "padding_ratio": args.padding_ratio,
                            "is_private": "false",
                        }
                    )
            seen_hashes[role].add(source_hash)

        atomic_write_text(
            temporary / "train_list.txt",
            "\n".join(list_lines["train"]) + "\n",
        )
        atomic_write_text(
            temporary / "validation_list.txt",
            "\n".join(list_lines["validation"]) + "\n",
        )
        atomic_write_csv(
            temporary / "dataset_manifest.csv",
            converted_manifest,
            MANIFEST_COLUMNS,
        )
        atomic_write_csv(temporary / "crop_errors.csv", errors, ERROR_COLUMNS)
        character_inventory = {
            "schema_version": "1.0",
            "build_id": build_id,
            "encoding": "UTF-8",
            "counts_by_split": {
                role: dict(sorted(counts.items()))
                for role, counts in character_counts.items()
            },
            "unique_characters": sorted(
                set().union(*(set(counts) for counts in character_counts.values()))
            ),
        }
        critical_inventory = {
            "schema_version": "1.0",
            "build_id": build_id,
            "counts_by_split": {
                role: dict(sorted(counts.items()))
                for role, counts in critical_counts.items()
            },
        }
        atomic_write_json(
            temporary / "character_inventory.json",
            character_inventory,
        )
        atomic_write_json(
            temporary / "critical_field_inventory.json",
            critical_inventory,
        )
        stats = {
            "schema_version": "1.0",
            "status": "passed",
            "build_id": build_id,
            "profile": args.profile,
            "parameters": parameters,
            "source_manifest": source_manifest.as_posix(),
            "source_manifest_sha256": sha256_file(source_manifest),
            "output_root": output_root.as_posix(),
            "input_page_count": len(rows),
            "processed_unique_page_count": sum(
                len(value) for value in seen_hashes.values()
            ),
            "skipped_exact_duplicate_count": skipped_duplicate_count,
            "output_crop_count": len(converted_manifest),
            "skipped_invalid_crop_count": len(errors),
            "private_row_count": 0,
            "counts_by_split": dict(
                sorted(Counter(row["split"] for row in converted_manifest).items())
            ),
            "counts_by_dataset": dict(
                sorted(Counter(row["dataset"] for row in converted_manifest).items())
            ),
            "counts_by_crop_type": dict(
                sorted(Counter(row["crop_type"] for row in converted_manifest).items())
            ),
            "storage_gate": storage_gate,
            "duration_seconds": time.monotonic() - started,
        }
        atomic_write_json(temporary / "statistics.json", stats)
        stats["checksum_entry_count"] = write_checksums(temporary)
        atomic_write_json(temporary / "statistics.json", stats)
        write_checksums(temporary)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


def _quad(polygon: list[list[float]]) -> list[list[float]]:
    if len(polygon) == 4:
        return polygon
    xs = [float(point[0]) for point in polygon]
    ys = [float(point[1]) for point in polygon]
    return [
        [min(xs), min(ys)],
        [max(xs), min(ys)],
        [max(xs), max(ys)],
        [min(xs), max(ys)],
    ]


def _validate_crop(crop: Image.Image) -> None:
    if crop.width < 2 or crop.height < 2:
        raise ValueError("crop dimensions are invalid")
    grayscale = crop.convert("L")
    extrema = grayscale.getextrema()
    if extrema is None or extrema[1] - extrema[0] < 2:
        raise ValueError("crop is near-empty")
    if not all(value == value for value in ImageStat.Stat(grayscale).mean):
        raise ValueError("crop statistics are not finite")


if __name__ == "__main__":
    raise SystemExit(main())
