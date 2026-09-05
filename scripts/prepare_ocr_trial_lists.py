#!/usr/bin/env python3
"""Materialize verified absolute PaddleOCR TRAIN/DEV_SELECT label lists on D:."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ocr.build_support import output_transaction, write_checksums  # noqa: E402
from src.ocr.training_data import validate_transcription  # noqa: E402
from src.rotation_common import (  # noqa: E402
    atomic_write_json,
    atomic_write_text,
    canonical_json,
    sha256_file,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real-root",
        default="D:/OCR_Model_Assets/data/recognition_training",
    )
    parser.add_argument(
        "--synthetic-root",
        default="D:/OCR_Model_Assets/data/synthetic_recognition",
    )
    parser.add_argument(
        "--vendor-root",
        default="D:/OCR_Model_Assets/vendor/PaddleOCR",
    )
    parser.add_argument(
        "--output-root",
        default="D:/OCR_Model_Assets/data/ocr_trial_lists",
    )
    parser.add_argument(
        "--report",
        default=str(PROJECT_ROOT / "reports/ocr_upgrade/trial_list_manifest.json"),
    )
    parser.add_argument(
        "--training-data-report",
        default=str(PROJECT_ROOT / "reports/ocr_upgrade/training_data_summary.json"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    real_root = Path(args.real_root)
    synthetic_root = Path(args.synthetic_root)
    vendor_root = Path(args.vendor_root)
    output_root = Path(args.output_root)
    real_manifest = real_root / "dataset_manifest.csv"
    synthetic_manifest = synthetic_root / "dataset_manifest.csv"
    training_data_report = Path(args.training_data_report)
    provenance = json.loads(training_data_report.read_text(encoding="utf-8"))
    _verify_corpus_provenance(
        provenance["recognition"],
        root=real_root,
        manifest=real_manifest,
    )
    _verify_corpus_provenance(
        provenance["synthetic"],
        root=synthetic_root,
        manifest=synthetic_manifest,
    )
    checksum_indices = {
        real_root: _read_checksum_index(real_root / "checksums.sha256"),
        synthetic_root: _read_checksum_index(
            synthetic_root / "checksums.sha256"
        ),
    }
    real_rows = _read_rows(real_manifest)
    synthetic_rows = _read_rows(synthetic_manifest)
    dictionaries = {
        "general": vendor_root / "ppocr/utils/dict/ppocrv6_dict.txt",
        "thai": vendor_root / "ppocr/utils/dict/ppocrv5_th_dict.txt",
    }
    dictionary_tokens = {
        name: set(path.read_text(encoding="utf-8").splitlines())
        for name, path in dictionaries.items()
    }
    exclusions: list[dict[str, Any]] = []

    real = {
        split: _real_lines(
            real_rows,
            real_root=real_root,
            split=split,
            checksum_index=checksum_indices[real_root],
            allowed_tokens=dictionary_tokens["general"],
            exclusions=exclusions,
        )
        for split in ("train", "validation")
    }
    general = {
        split: _synthetic_lines(
            synthetic_rows,
            synthetic_root=synthetic_root,
            track="general",
            split=split,
            checksum_index=checksum_indices[synthetic_root],
            allowed_tokens=dictionary_tokens["general"],
            exclusions=exclusions,
        )
        for split in ("train", "validation")
    }
    thai = {
        split: _synthetic_lines(
            synthetic_rows,
            synthetic_root=synthetic_root,
            track="thai",
            split=split,
            checksum_index=checksum_indices[synthetic_root],
            allowed_tokens=dictionary_tokens["thai"],
            exclusions=exclusions,
        )
        for split in ("train", "validation")
    }
    source_counts = {
        "real_train": sum(row["split"] == "train" for row in real_rows),
        "real_validation": sum(
            row["split"] == "validation" for row in real_rows
        ),
        "general_train": sum(
            row["split"] == "train" and row["language_track"] == "general"
            for row in synthetic_rows
        ),
        "general_validation": sum(
            row["split"] == "validation"
            and row["language_track"] == "general"
            for row in synthetic_rows
        ),
        "thai_train": sum(
            row["split"] == "train" and row["language_track"] == "thai"
            for row in synthetic_rows
        ),
        "thai_validation": sum(
            row["split"] == "validation"
            and row["language_track"] == "thai"
            for row in synthetic_rows
        ),
    }
    if source_counts != {
        "real_train": 119773,
        "real_validation": 18113,
        "general_train": 29943,
        "general_validation": 4528,
        "thai_train": 12000,
        "thai_validation": 2000,
    }:
        raise ValueError(f"recognition source corpus count drift: {source_counts}")
    if (len(real["train"]), len(real["validation"])) != (119772, 18111):
        raise ValueError("supported real recognition count drift")
    if (len(general["train"]), len(general["validation"])) != (28072, 4245):
        raise ValueError("supported general synthetic count drift")
    if (len(thai["train"]), len(thai["validation"])) != (12000, 2000):
        raise ValueError("Thai synthetic corpus count drift")
    if len(exclusions) != 2157:
        raise ValueError(f"unexpected dictionary exclusion count: {len(exclusions)}")

    lists = {
        "general_real_train.txt": real["train"],
        "general_real_dev_select.txt": real["validation"],
        "general_mixed_train.txt": [*real["train"], *general["train"]],
        "general_mixed_dev_select.txt": [
            *real["validation"],
            *general["validation"],
        ],
        "thai_train.txt": thai["train"],
        "thai_dev_select.txt": thai["validation"],
    }
    dictionary_audit = {
        name: _dictionary_audit(
            dictionary_path,
            [
                line.rsplit("\t", 1)[1]
                for list_name, values in lists.items()
                if (
                    name == "general"
                    and list_name
                    in {
                        "general_mixed_train.txt",
                        "general_mixed_dev_select.txt",
                    }
                )
                or (name == "thai" and list_name.startswith("thai"))
                for line in values
            ],
        )
        for name, dictionary_path in dictionaries.items()
    }

    source_hashes = {
        real_manifest.as_posix(): sha256_file(real_manifest),
        synthetic_manifest.as_posix(): sha256_file(synthetic_manifest),
        training_data_report.as_posix(): sha256_file(training_data_report),
        (real_root / "checksums.sha256").as_posix(): sha256_file(
            real_root / "checksums.sha256"
        ),
        (synthetic_root / "checksums.sha256").as_posix(): sha256_file(
            synthetic_root / "checksums.sha256"
        ),
        **{
            path.as_posix(): sha256_file(path)
            for path in dictionaries.values()
        },
    }
    build_id = "ocr-trial-lists-" + hashlib.sha256(
        canonical_json(
            {
                "sources": source_hashes,
                "counts": {name: len(values) for name, values in lists.items()},
            }
        ).encode("utf-8")
    ).hexdigest()[:16]
    with output_transaction(
        output_root,
        expected_leaf="ocr_trial_lists",
        force=args.force,
    ) as temporary:
        for name, values in lists.items():
            atomic_write_text(temporary / name, "\n".join(values) + "\n")
        payload: dict[str, Any] = {
            "schema_version": "1.0",
            "status": "passed",
            "build_id": build_id,
            "output_root": output_root.as_posix(),
            "source_hashes": source_hashes,
            "counts": {name: len(values) for name, values in lists.items()},
            "source_counts_before_dictionary_gate": source_counts,
            "general_mixed_synthetic_fraction": {
                "train": len(general["train"])
                / (len(real["train"]) + len(general["train"])),
                "dev_select": len(general["validation"])
                / (len(real["validation"]) + len(general["validation"])),
            },
            "dictionary_audit": dictionary_audit,
            "dictionary_exclusions": _summarize_exclusions(exclusions),
            "private_row_count": 0,
            "failure_count": 0,
        }
        atomic_write_json(temporary / "manifest.json", payload)
        payload["checksum_entry_count"] = write_checksums(temporary)
        atomic_write_json(temporary / "manifest.json", payload)
        write_checksums(temporary)
    payload["output_checksums_sha256"] = sha256_file(output_root / "checksums.sha256")
    atomic_write_json(Path(args.report), payload)
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    return 0


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty manifest: {path}")
    return rows


def _real_lines(
    rows: list[dict[str, str]],
    *,
    real_root: Path,
    split: str,
    checksum_index: dict[str, str],
    allowed_tokens: set[str],
    exclusions: list[dict[str, Any]],
) -> list[str]:
    result = []
    for row in rows:
        if row["split"] != split:
            continue
        _refuse_private(row)
        path = (real_root / row["crop_path"]).resolve()
        _verify_indexed_file(
            path,
            row["crop_sha256"],
            root=real_root,
            checksum_index=checksum_index,
        )
        text = validate_transcription(row["transcription"])
        if _exclude_unsupported(
            text,
            allowed_tokens=allowed_tokens,
            exclusions=exclusions,
            sample_id=row["crop_id"],
            source="real_public",
            split=split,
        ):
            continue
        result.append(f"{path.as_posix()}\t{text}")
    return result


def _synthetic_lines(
    rows: list[dict[str, str]],
    *,
    synthetic_root: Path,
    track: str,
    split: str,
    checksum_index: dict[str, str],
    allowed_tokens: set[str],
    exclusions: list[dict[str, Any]],
) -> list[str]:
    result = []
    for row in rows:
        if row["language_track"] != track or row["split"] != split:
            continue
        _refuse_private(row)
        path = (synthetic_root / row["image_path"]).resolve()
        _verify_indexed_file(
            path,
            row["image_sha256"],
            root=synthetic_root,
            checksum_index=checksum_index,
        )
        text = validate_transcription(row["transcription"])
        if _exclude_unsupported(
            text,
            allowed_tokens=allowed_tokens,
            exclusions=exclusions,
            sample_id=row["sample_id"],
            source=f"synthetic_{track}",
            split=split,
        ):
            continue
        result.append(f"{path.as_posix()}\t{text}")
    return result


def _refuse_private(row: dict[str, str]) -> None:
    if str(row.get("is_private", "")).casefold() != "false":
        raise ValueError(f"private training row refused: {row}")


def _verify_indexed_file(
    path: Path,
    expected_sha: str,
    *,
    root: Path,
    checksum_index: dict[str, str],
) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    resolved_root = root.resolve()
    if resolved_root not in path.parents:
        raise ValueError(f"training crop escapes corpus root: {path}")
    relative = path.relative_to(resolved_root).as_posix()
    indexed_sha = checksum_index.get(relative)
    if indexed_sha != expected_sha.casefold():
        raise ValueError(
            f"training crop manifest/checksum-index mismatch: {path}"
        )


def _read_checksum_index(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if not separator or len(digest) != 64 or not relative:
            raise ValueError(f"malformed checksum line in {path}: {line!r}")
        if relative in result:
            raise ValueError(f"duplicate checksum path in {path}: {relative}")
        result[relative] = digest.casefold()
    return result


def _verify_corpus_provenance(
    record: dict[str, Any],
    *,
    root: Path,
    manifest: Path,
) -> None:
    if record.get("status") != "passed":
        raise ValueError(f"corpus provenance is not passed: {record}")
    if record.get("independent_checksum_verification") != "passed":
        raise ValueError("corpus has no recorded independent checksum replay")
    if Path(record["output_root"]).resolve() != root.resolve():
        raise ValueError("corpus output-root provenance mismatch")
    if sha256_file(manifest) != record["manifest_sha256"]:
        raise ValueError("corpus manifest digest drift")
    if sha256_file(root / "checksums.sha256") != record["checksums_sha256"]:
        raise ValueError("corpus checksum-index digest drift")


def _dictionary_audit(
    path: Path,
    transcriptions: list[str],
) -> dict[str, Any]:
    tokens = set(path.read_text(encoding="utf-8").splitlines())
    characters = Counter(
        character
        for transcription in transcriptions
        for character in transcription
        if character != " "
    )
    unsupported = {
        character: count
        for character, count in characters.items()
        if character not in tokens
    }
    return {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "dictionary_token_count": len(tokens),
        "observed_character_count": sum(characters.values()),
        "observed_unique_character_count": len(characters),
        "unsupported_unique_character_count": len(unsupported),
        "unsupported_occurrence_count": sum(unsupported.values()),
        "unsupported_characters": dict(
            sorted(unsupported.items(), key=lambda item: (-item[1], item[0]))[:50]
        ),
        "compatible": not unsupported,
    }


def _exclude_unsupported(
    text: str,
    *,
    allowed_tokens: set[str],
    exclusions: list[dict[str, Any]],
    sample_id: str,
    source: str,
    split: str,
) -> bool:
    unsupported = sorted(
        {character for character in text if character != " " and character not in allowed_tokens}
    )
    if not unsupported:
        return False
    exclusions.append(
        {
            "sample_id": sample_id,
            "source": source,
            "split": split,
            "reason": "unsupported_by_pinned_official_dictionary",
            "unsupported_codepoints": [
                f"U+{ord(character):04X}" for character in unsupported
            ],
            "transcription_sha256": hashlib.sha256(
                text.encode("utf-8")
            ).hexdigest(),
        }
    )
    return True


def _summarize_exclusions(
    exclusions: list[dict[str, Any]],
) -> dict[str, Any]:
    by_source = Counter(row["source"] for row in exclusions)
    by_split = Counter(row["split"] for row in exclusions)
    by_codepoint = Counter(
        codepoint
        for row in exclusions
        for codepoint in row["unsupported_codepoints"]
    )
    return {
        "count": len(exclusions),
        "reason": "Samples whose target text cannot be represented by the pinned official model dictionary are excluded rather than silently mutated.",
        "by_source": dict(sorted(by_source.items())),
        "by_split": dict(sorted(by_split.items())),
        "by_codepoint": dict(sorted(by_codepoint.items())),
        "sample_ids_sha256": hashlib.sha256(
            "\n".join(sorted(row["sample_id"] for row in exclusions)).encode(
                "utf-8"
            )
        ).hexdigest(),
        "examples": exclusions[:10],
    }


if __name__ == "__main__":
    raise SystemExit(main())
