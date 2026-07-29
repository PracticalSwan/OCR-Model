#!/usr/bin/env python3
"""Generate deterministic OFL-font financial and Thai OCR line corpora."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ocr.build_support import (  # noqa: E402
    check_storage_gate,
    output_transaction,
    write_checksums,
)
from src.ocr.synthetic import (  # noqa: E402
    GENERAL_CATEGORIES,
    THAI_CATEGORIES,
    general_financial_text,
    render_synthetic_line,
    synthetic_count_for_fraction,
    thai_financial_text,
)
from src.rotation_common import (  # noqa: E402
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    sha256_file,
)


GOOGLE_FONTS_COMMIT = "9fab8b6cc7b2f20376914fd765d918c698c66d75"
FONT_ASSETS = {
    "noto_sans": {
        "filename": "NotoSans[wdth,wght].ttf",
        "url": (
            "https://raw.githubusercontent.com/google/fonts/"
            f"{GOOGLE_FONTS_COMMIT}/ofl/notosans/"
            "NotoSans%5Bwdth%2Cwght%5D.ttf"
        ),
        "git_blob_sha1": "75575046c015ff623a848096a15779867ba71453",
        "license_filename": "NotoSans-OFL.txt",
        "license_url": (
            "https://raw.githubusercontent.com/google/fonts/"
            f"{GOOGLE_FONTS_COMMIT}/ofl/notosans/OFL.txt"
        ),
        "license_git_blob_sha1": "6843f31878c9945e1e71e1baa275546b4eefead8",
    },
    "noto_sans_thai": {
        "filename": "NotoSansThai[wdth,wght].ttf",
        "url": (
            "https://raw.githubusercontent.com/google/fonts/"
            f"{GOOGLE_FONTS_COMMIT}/ofl/notosansthai/"
            "NotoSansThai%5Bwdth%2Cwght%5D.ttf"
        ),
        "git_blob_sha1": "34b48ab6f74867dbfce19410a2f452abef34e3ff",
        "license_filename": "NotoSansThai-OFL.txt",
        "license_url": (
            "https://raw.githubusercontent.com/google/fonts/"
            f"{GOOGLE_FONTS_COMMIT}/ofl/notosansthai/OFL.txt"
        ),
        "license_git_blob_sha1": "7fa8dcb08ce4d2667b683ac4a8e79166e44e5277",
    },
}
MANIFEST_COLUMNS = (
    "build_id",
    "sample_id",
    "language_track",
    "split",
    "category",
    "transcription",
    "image_path",
    "image_sha256",
    "width",
    "height",
    "font_id",
    "font_sha256",
    "font_license",
    "generation_seed",
    "augmentation",
    "is_synthetic",
    "is_private",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real-corpus-root",
        default=(
            "D:/CSX4201/vision-info-extraction-assets/data/recognition_training"
        ),
    )
    parser.add_argument(
        "--output-root",
        default=(
            "D:/CSX4201/vision-info-extraction-assets/data/synthetic_recognition"
        ),
    )
    parser.add_argument(
        "--font-root",
        default=(
            "D:/CSX4201/vision-info-extraction-assets/fonts/google-fonts/"
            f"{GOOGLE_FONTS_COMMIT}"
        ),
    )
    parser.add_argument("--synthetic-fraction", type=float, default=0.20)
    parser.add_argument("--thai-train-count", type=int, default=12000)
    parser.add_argument("--thai-validation-count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not 0.15 <= args.synthetic_fraction <= 0.30:
        parser.error("--synthetic-fraction must be within [0.15, 0.30]")
    if args.thai_train_count <= 0 or args.thai_validation_count <= 0:
        parser.error("Thai split counts must be positive")

    started = time.monotonic()
    real_root = Path(args.real_corpus_root)
    real_stats_path = real_root / "statistics.json"
    real_manifest_path = real_root / "dataset_manifest.csv"
    real_stats = json.loads(real_stats_path.read_text(encoding="utf-8"))
    if real_stats.get("private_row_count") != 0:
        raise ValueError("real corpus reports private rows")
    real_train = int(real_stats["counts_by_split"]["train"])
    real_validation = int(real_stats["counts_by_split"]["validation"])
    general_train = synthetic_count_for_fraction(
        real_train,
        args.synthetic_fraction,
    )
    general_validation = synthetic_count_for_fraction(
        real_validation,
        args.synthetic_fraction,
    )
    total_synthetic = (
        general_train
        + general_validation
        + args.thai_train_count
        + args.thai_validation_count
    )
    output_root = Path(args.output_root)
    storage_gate = check_storage_gate(
        project_root=PROJECT_ROOT,
        output_root=output_root,
        anticipated_output_bytes=total_synthetic * 40 * 1024,
    )
    fonts = _obtain_fonts(Path(args.font_root))
    build_material = {
        "schema_version": "1.0",
        "seed": args.seed,
        "synthetic_fraction": args.synthetic_fraction,
        "real_manifest_sha256": sha256_file(real_manifest_path),
        "counts": {
            "general_train": general_train,
            "general_validation": general_validation,
            "thai_train": args.thai_train_count,
            "thai_validation": args.thai_validation_count,
        },
        "fonts": {
            key: {
                "sha256": value["font_sha256"],
                "license_sha256": value["license_sha256"],
            }
            for key, value in fonts.items()
        },
    }
    build_id = "ocr-synthetic-" + hashlib.sha256(
        json.dumps(
            build_material,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:16]

    with output_transaction(
        output_root,
        expected_leaf="synthetic_recognition",
        force=args.force,
    ) as temporary:
        rows: list[dict[str, Any]] = []
        list_lines: dict[tuple[str, str], list[str]] = {}
        category_counts: Counter[str] = Counter()
        font_cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}
        specifications = (
            ("general", "train", general_train),
            ("general", "validation", general_validation),
            ("thai", "train", args.thai_train_count),
            ("thai", "validation", args.thai_validation_count),
        )
        for track, split, count in specifications:
            list_lines[(track, split)] = []
            font_id = "noto_sans" if track == "general" else "noto_sans_thai"
            font_record = fonts[font_id]
            categories = GENERAL_CATEGORIES if track == "general" else THAI_CATEGORIES
            text_factory = (
                general_financial_text
                if track == "general"
                else thai_financial_text
            )
            for index in range(count):
                seed = _sample_seed(args.seed, track, split, index)
                rng = random.Random(seed)
                text = text_factory(index, rng)
                font_size = rng.randint(22, 52)
                cache_key = (font_id, font_size)
                font = font_cache.get(cache_key)
                if font is None:
                    font = ImageFont.truetype(
                        str(font_record["font_path"]),
                        font_size,
                    )
                    font_cache[cache_key] = font
                image, augmentation = render_synthetic_line(
                    text,
                    font=font,
                    seed=seed ^ 0xA5A5A5A5,
                )
                sample_id = f"syn_{track}_{split}_{index:06d}"
                relative_image = Path(track) / split / f"{sample_id}.png"
                image_path = temporary / relative_image
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(image_path, format="PNG", optimize=False)
                with Image.open(image_path) as reloaded:
                    reloaded.load()
                    if reloaded.size != image.size:
                        raise RuntimeError(f"synthetic reload mismatch: {sample_id}")
                image_hash = sha256_file(image_path)
                category = categories[index % len(categories)]
                category_counts[f"{track}:{split}:{category}"] += 1
                list_lines[(track, split)].append(
                    f"{relative_image.as_posix()}\t{text}"
                )
                rows.append(
                    {
                        "build_id": build_id,
                        "sample_id": sample_id,
                        "language_track": track,
                        "split": split,
                        "category": category,
                        "transcription": text,
                        "image_path": relative_image.as_posix(),
                        "image_sha256": image_hash,
                        "width": image.width,
                        "height": image.height,
                        "font_id": font_id,
                        "font_sha256": font_record["font_sha256"],
                        "font_license": "SIL-OFL-1.1",
                        "generation_seed": seed,
                        "augmentation": json.dumps(
                            augmentation,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        "is_synthetic": "true",
                        "is_private": "false",
                    }
                )
        for (track, split), values in list_lines.items():
            atomic_write_text(
                temporary / f"{track}_{split}_list.txt",
                "\n".join(values) + "\n",
            )
        atomic_write_csv(
            temporary / "dataset_manifest.csv",
            rows,
            MANIFEST_COLUMNS,
        )
        output_stats = {
            "schema_version": "1.0",
            "status": "passed",
            "build_id": build_id,
            "seed": args.seed,
            "output_root": output_root.as_posix(),
            "sample_count": len(rows),
            "failure_count": 0,
            "private_row_count": 0,
            "counts": build_material["counts"],
            "category_counts": dict(sorted(category_counts.items())),
            "real_corpus": {
                "root": real_root.as_posix(),
                "build_id": real_stats["build_id"],
                "manifest_sha256": build_material["real_manifest_sha256"],
                "train_count": real_train,
                "validation_count": real_validation,
            },
            "general_mixture": {
                "requested_synthetic_fraction": args.synthetic_fraction,
                "train_actual_synthetic_fraction": general_train
                / (real_train + general_train),
                "validation_actual_synthetic_fraction": general_validation
                / (real_validation + general_validation),
            },
            "fonts": {
                key: {
                    name: (
                        value.as_posix() if isinstance(value, Path) else value
                    )
                    for name, value in record.items()
                }
                for key, record in fonts.items()
            },
            "thai_dataset_decision": {
                "candidate": "openthaigpt/thai-ocr-evaluation",
                "source": (
                    "https://huggingface.co/datasets/"
                    "openthaigpt/thai-ocr-evaluation"
                ),
                "declared_license": "CC-BY-SA-4.0",
                "declared_size": 104,
                "card_description": (
                    "Images and text are described as derived from various "
                    "open-source websites without per-sample source/license mapping."
                ),
                "downloaded": False,
                "used_for_training_or_selection": False,
                "redistribution_allowed": "not established per sample",
                "derived_weights_distribution": "not relied upon",
                "decision": (
                    "Rejected for this upgrade because underlying image provenance "
                    "and redistribution rights are not sufficiently explicit."
                ),
                "fallback": (
                    "OFL-licensed Noto Sans Thai synthetic TRAIN and validation "
                    "splits; claims remain synthetic/integration-only."
                ),
            },
            "storage_gate": storage_gate,
            "duration_seconds": time.monotonic() - started,
        }
        atomic_write_json(temporary / "statistics.json", output_stats)
        output_stats["checksum_entry_count"] = write_checksums(temporary)
        atomic_write_json(temporary / "statistics.json", output_stats)
        write_checksums(temporary)

    report = {
        **output_stats,
        "manifest_path": (output_root / "dataset_manifest.csv").as_posix(),
        "manifest_sha256": sha256_file(output_root / "dataset_manifest.csv"),
        "checksums_sha256": sha256_file(output_root / "checksums.sha256"),
    }
    report_path = PROJECT_ROOT / "reports" / "ocr_upgrade" / "synthetic_data_manifest.json"
    atomic_write_json(report_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def _obtain_fonts(root: Path) -> dict[str, dict[str, Any]]:
    root.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, Any]] = {}
    for font_id, definition in FONT_ASSETS.items():
        font_path = root / definition["filename"]
        license_path = root / definition["license_filename"]
        _download_pinned(
            definition["url"],
            font_path,
            expected_git_blob_sha1=definition["git_blob_sha1"],
        )
        _download_pinned(
            definition["license_url"],
            license_path,
            expected_git_blob_sha1=definition["license_git_blob_sha1"],
        )
        records[font_id] = {
            "font_path": font_path,
            "font_sha256": sha256_file(font_path),
            "license_path": license_path,
            "license_sha256": sha256_file(license_path),
            "license": "SIL Open Font License 1.1",
            "source_commit": GOOGLE_FONTS_COMMIT,
            "source_url": definition["url"],
            "license_url": definition["license_url"],
            "redistributed_in_project": False,
        }
    return records


def _download_pinned(
    url: str,
    destination: Path,
    *,
    expected_git_blob_sha1: str,
) -> None:
    if destination.is_file():
        data = destination.read_bytes()
        if _git_blob_sha1(data) == expected_git_blob_sha1:
            return
        raise ValueError(f"existing pinned font asset hash mismatch: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: BaseException | None = None
    for attempt in range(1, 4):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "csx4201-ocr-upgrade/1.0"},
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
            if _git_blob_sha1(data) != expected_git_blob_sha1:
                raise ValueError(f"downloaded Git blob hash mismatch for {url}")
            partial = destination.with_suffix(destination.suffix + ".partial")
            partial.write_bytes(data)
            partial.replace(destination)
            return
        except (OSError, urllib.error.URLError, ValueError) as exc:
            last_error = exc
            if attempt == 3:
                break
            time.sleep(attempt)
    raise RuntimeError(f"failed to download pinned font asset: {url}") from last_error


def _git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(
        f"blob {len(data)}\0".encode("ascii") + data,
        usedforsecurity=False,
    ).hexdigest()


def _sample_seed(base_seed: int, track: str, split: str, index: int) -> int:
    return int.from_bytes(
        hashlib.sha256(
            f"{base_seed}|{track}|{split}|{index}".encode("utf-8")
        ).digest()[:8],
        "big",
    )


if __name__ == "__main__":
    raise SystemExit(main())
