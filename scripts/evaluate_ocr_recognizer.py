#!/usr/bin/env python3
"""Evaluate a general or Thai recognizer on frozen DEV_SELECT crop data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.ocr_reporting import build_metric_report  # noqa: E402
from src.ocr.trials import (  # noqa: E402
    aggregate_recognition_results,
    recognition_sample_result,
    recognizer_selection_score,
    sha256_tree,
)
from src.rotation_common import (  # noqa: E402
    atomic_write_json,
    atomic_write_text,
    canonical_json,
    sha256_file,
)


REAL_ROOT = Path(
    "D:/CSX4201/vision-info-extraction-assets/data/recognition_training"
)
SYNTHETIC_ROOT = Path(
    "D:/CSX4201/vision-info-extraction-assets/data/synthetic_recognition"
)
VENDOR_ROOT = Path("D:/CSX4201/vision-info-extraction-assets/vendor/PaddleOCR")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--track", choices=("general", "thai"), required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--detector-sha256", required=True)
    parser.add_argument("--real-root", default=str(REAL_ROOT))
    parser.add_argument("--synthetic-root", default=str(SYNTHETIC_ROOT))
    parser.add_argument("--dictionary")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    real_root = Path(args.real_root)
    synthetic_root = Path(args.synthetic_root)
    dictionary_path = Path(args.dictionary) if args.dictionary else (
        VENDOR_ROOT
        / (
            "ppocr/utils/dict/ppocrv6_dict.txt"
            if args.track == "general"
            else "ppocr/utils/dict/ppocrv5_th_dict.txt"
        )
    )
    rows, source_manifests, exclusions = _load_samples(
        track=args.track,
        real_root=real_root,
        synthetic_root=synthetic_root,
        dictionary_path=dictionary_path,
    )
    manifest_sha = _combined_manifest_sha(source_manifests)
    model_dir = Path(args.model_dir)
    recognizer_sha = sha256_tree(model_dir)
    configuration_sha = sha256_file(Path(args.config))
    source_commit = _git_commit()
    build_id = "ocr-recognizer-eval-" + hashlib.sha256(
        canonical_json(
            {
                "trial_id": args.trial_id,
                "track": args.track,
                "manifest": manifest_sha,
                "recognizer": recognizer_sha,
                "config": configuration_sha,
            }
        ).encode("utf-8")
    ).hexdigest()[:16]

    from paddleocr import TextRecognition

    model = TextRecognition(
        model_name=args.model_name,
        model_dir=str(model_dir),
        device=args.device,
    )
    _reset_peak_memory()
    sample_results: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for offset in range(0, len(rows), args.batch_size):
        group = rows[offset : offset + args.batch_size]
        paths = [str(row["image_path"]) for row in group]
        started = time.perf_counter()
        try:
            outputs = list(model.predict(input=paths, batch_size=len(paths)))
            if len(outputs) != len(group):
                raise RuntimeError(
                    f"recognizer returned {len(outputs)} results for {len(group)} crops"
                )
            duration = (time.perf_counter() - started) / len(group)
            for row, output in zip(group, outputs):
                text, confidence = _recognizer_payload(output)
                _append_sample(
                    row,
                    prediction=text,
                    confidence=confidence,
                    duration_seconds=duration,
                    error=None,
                    sample_results=sample_results,
                    prediction_rows=prediction_rows,
                )
        except Exception as batch_error:
            for row in group:
                single_started = time.perf_counter()
                try:
                    output = list(
                        model.predict(input=str(row["image_path"]), batch_size=1)
                    )
                    if len(output) != 1:
                        raise RuntimeError(
                            f"recognizer returned {len(output)} results for one crop"
                        )
                    text, confidence = _recognizer_payload(output[0])
                    error = None
                except Exception as single_error:
                    text, confidence = "", 0.0
                    error = (
                        f"batch={type(batch_error).__name__}: {batch_error}; "
                        f"single={type(single_error).__name__}: {single_error}"
                    )
                _append_sample(
                    row,
                    prediction=text,
                    confidence=confidence,
                    duration_seconds=time.perf_counter() - single_started,
                    error=error,
                    sample_results=sample_results,
                    prediction_rows=prediction_rows,
                )
        completed = min(offset + len(group), len(rows))
        if completed == len(rows) or completed % 1024 < len(group):
            print(
                f"{args.track} recognizer DEV_SELECT: {completed}/{len(rows)}",
                flush=True,
            )

    metrics = aggregate_recognition_results(sample_results)
    metrics["selection_score"] = recognizer_selection_score(metrics)
    metrics["track"] = args.track
    metrics["trial_id"] = args.trial_id
    metrics["peak_gpu_memory_bytes"] = _peak_memory_bytes()
    metrics["source_manifests"] = [
        {"path": path.as_posix(), "sha256": sha256_file(path)}
        for path in source_manifests
    ]
    metrics["dictionary"] = {
        "path": dictionary_path.as_posix(),
        "sha256": sha256_file(dictionary_path),
        "compatible_evaluated_samples": len(rows),
        "excluded_sample_count": len(exclusions),
        "excluded_by_codepoint": _exclusion_codepoint_counts(exclusions),
        "excluded_sample_ids_sha256": hashlib.sha256(
            "\n".join(sorted(row["sample_id"] for row in exclusions)).encode(
                "utf-8"
            )
        ).hexdigest(),
        "policy": "Unrepresentable targets are excluded and reported; target text is never silently mutated.",
    }
    predictions_path = Path(args.predictions)
    atomic_write_text(
        predictions_path,
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in prediction_rows
        ),
    )
    metrics["predictions_path"] = predictions_path.as_posix()
    metrics["predictions_sha256"] = sha256_file(predictions_path)
    metrics["model_directory"] = model_dir.as_posix()
    report = build_metric_report(
        build_id=build_id,
        split="dev_select",
        manifest_sha256=manifest_sha,
        detector_sha256=args.detector_sha256,
        recognizer_sha256=recognizer_sha,
        checkpoint_sha256=args.checkpoint_sha256,
        calibration_sha256=None,
        configuration_sha256=configuration_sha,
        source_commit=source_commit,
        device=args.device,
        sample_count=len(rows),
        failure_count=int(metrics["failure_count"]),
        duration_seconds=float(metrics["duration_seconds"]),
        private_row_count=0,
        metrics=metrics,
    )
    report["model_name"] = args.model_name
    report["environment"] = _environment_versions()
    atomic_write_json(Path(args.output), report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if int(metrics["failure_count"]) == 0 else 1


def _load_samples(
    *,
    track: str,
    real_root: Path,
    synthetic_root: Path,
    dictionary_path: Path,
) -> tuple[list[dict[str, Any]], list[Path], list[dict[str, Any]]]:
    samples: list[dict[str, Any]] = []
    manifests: list[Path] = []
    exclusions: list[dict[str, Any]] = []
    allowed_tokens = set(
        dictionary_path.read_text(encoding="utf-8").splitlines()
    )
    if track == "general":
        real_manifest = real_root / "dataset_manifest.csv"
        manifests.append(real_manifest)
        with real_manifest.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["split"] != "validation":
                    continue
                _refuse_private(row)
                if _record_dictionary_exclusion(
                    row["transcription"],
                    allowed_tokens=allowed_tokens,
                    sample_id=row["crop_id"],
                    source="real_public",
                    exclusions=exclusions,
                ):
                    continue
                path = _checked_crop(
                    real_root / row["crop_path"], row["crop_sha256"]
                )
                samples.append(
                    {
                        "sample_id": row["crop_id"],
                        "dataset": row["dataset"],
                        "source": "real_public",
                        "image_path": path,
                        "reference": row["transcription"],
                        "critical_fields": set(
                            filter(None, row["critical_fields"].split("|"))
                        ),
                        "category": "",
                    }
                )
    synthetic_manifest = synthetic_root / "dataset_manifest.csv"
    manifests.append(synthetic_manifest)
    with synthetic_manifest.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != "validation" or row["language_track"] != track:
                continue
            _refuse_private(row)
            if _record_dictionary_exclusion(
                row["transcription"],
                allowed_tokens=allowed_tokens,
                sample_id=row["sample_id"],
                source=f"synthetic_{track}",
                exclusions=exclusions,
            ):
                continue
            path = _checked_crop(
                synthetic_root / row["image_path"], row["image_sha256"]
            )
            samples.append(
                {
                    "sample_id": row["sample_id"],
                    "dataset": "synthetic_financial",
                    "source": "synthetic_public",
                    "image_path": path,
                    "reference": row["transcription"],
                    "critical_fields": set(),
                    "category": row["category"],
                }
            )
    expected = 22356 if track == "general" else 2000
    if len(samples) != expected:
        raise ValueError(
            f"{track} frozen recognition benchmark must contain {expected} crops, "
            f"found {len(samples)}"
        )
    identifiers = [row["sample_id"] for row in samples]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("recognition benchmark sample IDs are not unique")
    expected_exclusions = 285 if track == "general" else 0
    if len(exclusions) != expected_exclusions:
        raise ValueError(
            f"{track} dictionary exclusion count drift: {len(exclusions)}"
        )
    return (
        sorted(samples, key=lambda row: (row["source"], row["sample_id"])),
        manifests,
        exclusions,
    )


def _checked_crop(path: Path, expected_sha: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != expected_sha.casefold():
        raise ValueError(f"recognition crop hash drift: {path}")
    return path.resolve()


def _refuse_private(row: dict[str, str]) -> None:
    if str(row.get("is_private", "")).casefold() != "false":
        raise ValueError(f"private recognition row refused: {row}")


def _record_dictionary_exclusion(
    text: str,
    *,
    allowed_tokens: set[str],
    sample_id: str,
    source: str,
    exclusions: list[dict[str, Any]],
) -> bool:
    unsupported = sorted(
        {
            character
            for character in text
            if character != " " and character not in allowed_tokens
        }
    )
    if not unsupported:
        return False
    exclusions.append(
        {
            "sample_id": sample_id,
            "source": source,
            "unsupported_codepoints": [
                f"U+{ord(character):04X}" for character in unsupported
            ],
        }
    )
    return True


def _exclusion_codepoint_counts(
    exclusions: list[dict[str, Any]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in exclusions:
        for codepoint in row["unsupported_codepoints"]:
            counts[codepoint] = counts.get(codepoint, 0) + 1
    return dict(sorted(counts.items()))


def _recognizer_payload(output: Any) -> tuple[str, float]:
    payload = getattr(output, "json", output)
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise TypeError(f"unsupported recognizer result: {type(payload).__name__}")
    result = payload.get("res", payload)
    text = str(result.get("rec_text", ""))
    confidence = float(result.get("rec_score", 0.0))
    return text, confidence


def _append_sample(
    row: dict[str, Any],
    *,
    prediction: str,
    confidence: float,
    duration_seconds: float,
    error: str | None,
    sample_results: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
) -> None:
    result = recognition_sample_result(
        row["reference"],
        prediction,
        confidence=confidence,
        critical_fields=row["critical_fields"],
        category=row["category"],
        duration_seconds=duration_seconds,
    )
    result["failed"] = error is not None
    sample_results.append(result)
    prediction_rows.append(
        {
            "sample_id": row["sample_id"],
            "dataset": row["dataset"],
            "source": row["source"],
            "reference": row["reference"],
            "prediction": prediction,
            "confidence": confidence,
            "critical_fields": sorted(row["critical_fields"]),
            "category": row["category"],
            "duration_seconds": duration_seconds,
            "error": error,
            "private": False,
        }
    )


def _combined_manifest_sha(paths: list[Path]) -> str:
    return hashlib.sha256(
        canonical_json(
            [
                {"path": path.as_posix(), "sha256": sha256_file(path)}
                for path in paths
            ]
        ).encode("utf-8")
    ).hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()


def _environment_versions() -> dict[str, str]:
    import importlib.metadata

    return {
        "python": sys.version.split()[0],
        "paddle": _package_version("paddlepaddle-gpu", importlib.metadata),
        "paddleocr": _package_version("paddleocr", importlib.metadata),
        "paddlex": _package_version("paddlex", importlib.metadata),
    }


def _package_version(name: str, metadata: Any) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unavailable"


def _reset_peak_memory() -> None:
    try:
        import paddle

        paddle.device.cuda.reset_max_memory_allocated()
        paddle.device.cuda.reset_max_memory_reserved()
    except Exception:
        pass


def _peak_memory_bytes() -> int | None:
    try:
        import paddle

        return int(paddle.device.cuda.max_memory_allocated())
    except Exception:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
