#!/usr/bin/env python3
"""Build deterministic public PaddleOCR detector data on the asset volume."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

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
from src.ocr.training_data import convert_detection_annotation  # noqa: E402
from src.rotation_common import (  # noqa: E402
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    sha256_file,
)


MANIFEST_COLUMNS = (
    "build_id",
    "sample_id",
    "dataset",
    "split",
    "page_id",
    "document_id",
    "source_image_path",
    "source_image_sha256",
    "normalized_annotation_path",
    "annotation_sha256",
    "output_image_path",
    "output_image_sha256",
    "text_region_count",
    "split_group_id",
    "duplicate_group_id",
    "is_private",
)
ERROR_COLUMNS = ("page_id", "dataset", "split", "error_type", "message")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--profile", choices=["final"], default="final")
    parser.add_argument(
        "--output-root",
        default=(
            "D:/OCR_Model_Assets/data/detector_training"
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    started = time.monotonic()
    cfg = cfgmod.load_config(args.config)
    root = cfgmod.project_root(cfg)
    metadata = cfgmod.resolve_path(cfg, "metadata")
    source_manifest = metadata / "information_extraction_split_manifest.csv"
    rows = load_public_training_rows(root, source_manifest)
    output_root = Path(args.output_root)
    estimated_bytes = sum((root / row["image_path"]).stat().st_size for row in rows)
    storage_gate = check_storage_gate(
        project_root=root,
        output_root=output_root,
        anticipated_output_bytes=estimated_bytes + 100 * 1024**2,
    )
    build_id = training_build_id(
        kind="detector-data",
        profile=args.profile,
        source_manifest_sha256=sha256_file(source_manifest),
        rows=rows,
        parameters={"format": "PaddleOCR-DB-label-v1", "deduplicate_exact_hash": True},
    )

    with output_transaction(
        output_root,
        expected_leaf="detector_training",
        force=args.force,
    ) as temporary:
        for directory in ("train_images", "validation_images"):
            (temporary / directory).mkdir(parents=True)
        label_lines: dict[str, list[str]] = {"train": [], "validation": []}
        converted_manifest: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        seen_hashes: dict[str, set[str]] = {"train": set(), "validation": set()}
        skipped_duplicate_count = 0
        for row in rows:
            role = "train" if row["project_split"] == "train" else "validation"
            source_hash = row["sha256"].casefold()
            if source_hash in seen_hashes[role]:
                skipped_duplicate_count += 1
                continue
            try:
                image_path, annotation_path = resolve_training_inputs(root, row)
                annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
                if annotation.get("page_id") != row["page_id"]:
                    raise ValueError("page ID mismatch between split and annotation")
                regions = convert_detection_annotation(annotation)
                with ImageProbe(image_path) as dimensions:
                    page = annotation.get("page", {})
                    if (
                        int(page.get("width", 0)),
                        int(page.get("height", 0)),
                    ) != dimensions:
                        raise ValueError("annotation dimensions do not match image")
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                errors.append(
                    {
                        "page_id": row["page_id"],
                        "dataset": row["dataset"],
                        "split": row["project_split"],
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
                continue
            seen_hashes[role].add(source_hash)
            suffix = image_path.suffix.casefold() or ".png"
            relative_output = (
                Path(f"{role}_images")
                / row["dataset"]
                / f"{row['page_id']}{suffix}"
            )
            destination = temporary / relative_output
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(image_path, destination)
            output_hash = sha256_file(destination)
            if output_hash != source_hash:
                raise RuntimeError(f"copied image hash mismatch: {row['page_id']}")
            paddle_regions = [
                {
                    "transcription": region["transcription"],
                    "points": region["points"],
                }
                for region in regions
            ]
            label_lines[role].append(
                f"{relative_output.as_posix()}\t"
                f"{json.dumps(paddle_regions, ensure_ascii=False, separators=(',', ':'))}"
            )
            converted_manifest.append(
                {
                    "build_id": build_id,
                    "sample_id": f"det_{row['page_id']}",
                    "dataset": row["dataset"],
                    "split": role,
                    "page_id": row["page_id"],
                    "document_id": row["document_id"],
                    "source_image_path": row["image_path"],
                    "source_image_sha256": source_hash,
                    "normalized_annotation_path": row[
                        "normalized_annotation_path"
                    ],
                    "annotation_sha256": sha256_file(annotation_path),
                    "output_image_path": relative_output.as_posix(),
                    "output_image_sha256": output_hash,
                    "text_region_count": len(regions),
                    "split_group_id": row.get("split_group_id", ""),
                    "duplicate_group_id": row.get("duplicate_group_id", ""),
                    "is_private": "false",
                }
            )
        atomic_write_csv(
            temporary / "conversion_errors.csv",
            errors,
            ERROR_COLUMNS,
        )
        if errors:
            raise RuntimeError(
                f"detector conversion failed for {len(errors)} pages; "
                f"first error: {errors[0]}"
            )
        atomic_write_text(
            temporary / "train_labels.txt",
            "\n".join(label_lines["train"]) + "\n",
        )
        atomic_write_text(
            temporary / "validation_labels.txt",
            "\n".join(label_lines["validation"]) + "\n",
        )
        atomic_write_csv(
            temporary / "dataset_manifest.csv",
            converted_manifest,
            MANIFEST_COLUMNS,
        )
        stats = {
            "schema_version": "1.0",
            "status": "passed",
            "build_id": build_id,
            "profile": args.profile,
            "source_manifest": source_manifest.as_posix(),
            "source_manifest_sha256": sha256_file(source_manifest),
            "output_root": output_root.as_posix(),
            "input_page_count": len(rows),
            "output_page_count": len(converted_manifest),
            "skipped_exact_duplicate_count": skipped_duplicate_count,
            "failure_count": 0,
            "private_row_count": 0,
            "counts_by_split": dict(
                sorted(Counter(row["split"] for row in converted_manifest).items())
            ),
            "counts_by_dataset": dict(
                sorted(Counter(row["dataset"] for row in converted_manifest).items())
            ),
            "text_region_count": sum(
                int(row["text_region_count"]) for row in converted_manifest
            ),
            "storage_gate": storage_gate,
            "duration_seconds": time.monotonic() - started,
        }
        atomic_write_json(temporary / "statistics.json", stats)
        stats["checksum_entry_count"] = write_checksums(temporary)
        atomic_write_json(temporary / "statistics.json", stats)
        write_checksums(temporary)
    print(json.dumps(stats, indent=2))
    return 0


class ImageProbe:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.image = None

    def __enter__(self) -> tuple[int, int]:
        from PIL import Image

        self.image = Image.open(self.path)
        self.image.load()
        return self.image.size

    def __exit__(self, *_: object) -> None:
        if self.image is not None:
            self.image.close()


if __name__ == "__main__":
    raise SystemExit(main())
