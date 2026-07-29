#!/usr/bin/env python3
"""Download and validate exact official PaddleOCR training checkpoints."""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.verify_ocr_training_environment import (  # noqa: E402
    EXPECTED_VENDOR_COMMIT,
    prepare_architecture_for_training,
    resolve_vendor_config_paths,
)
from src.rotation_common import atomic_write_json, sha256_file  # noqa: E402


OFFICIAL_BASE = (
    "https://paddle-model-ecology.bj.bcebos.com/"
    "paddlex/official_pretrained_model"
)
MODELS = {
    "detector": {
        "model_name": "PP-OCRv6_medium_det",
        "filename": "PP-OCRv6_medium_det_pretrained.pdparams",
        "config": "configs/det/PP-OCRv6/PP-OCRv6_medium_det.yml",
        "model_type": "det",
        "documentation": (
            "https://github.com/PaddlePaddle/PaddleOCR/blob/v3.7.0/"
            "docs/version3.x/module_usage/text_detection.en.md"
        ),
        "documentation_training_link_present": False,
    },
    "general_recognizer": {
        "model_name": "PP-OCRv6_medium_rec",
        "filename": "PP-OCRv6_medium_rec_pretrained.pdparams",
        "config": "configs/rec/PP-OCRv6/PP-OCRv6_medium_rec.yml",
        "model_type": "rec",
        "documentation": (
            "https://github.com/PaddlePaddle/PaddleOCR/blob/v3.7.0/"
            "docs/version3.x/module_usage/text_recognition.en.md"
        ),
        "documentation_training_link_present": True,
    },
    "thai_recognizer": {
        "model_name": "th_PP-OCRv5_mobile_rec",
        "filename": "th_PP-OCRv5_mobile_rec_pretrained.pdparams",
        "config": (
            "configs/rec/PP-OCRv5/multi_language/"
            "th_PP-OCRv5_mobile_rec.yaml"
        ),
        "model_type": "rec",
        "documentation": (
            "https://github.com/PaddlePaddle/PaddleOCR/blob/v3.7.0/"
            "docs/version3.x/algorithm/PP-OCRv5/"
            "PP-OCRv5_multi_languages.en.md"
        ),
        "documentation_training_link_present": True,
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default=(
            "D:/CSX4201/vision-info-extraction-assets/"
            "models/paddleocr/training"
        ),
    )
    parser.add_argument(
        "--vendor-root",
        default="D:/CSX4201/vision-info-extraction-assets/vendor/PaddleOCR",
    )
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.retries <= 8:
        parser.error("--retries must be in [1, 8]")
    started = time.monotonic()
    output_root = Path(args.output_root).resolve()
    vendor_root = Path(args.vendor_root).resolve()
    if output_root.name.casefold() != "training":
        raise ValueError(f"unexpected checkpoint output root: {output_root}")
    if vendor_root.name.casefold() != "paddleocr":
        raise ValueError(f"unexpected vendor root: {vendor_root}")
    vendor_commit = _git(vendor_root, "rev-parse", "HEAD")
    if vendor_commit != EXPECTED_VENDOR_COMMIT:
        raise RuntimeError(f"unexpected PaddleOCR commit: {vendor_commit}")
    if _git(vendor_root, "status", "--porcelain"):
        raise RuntimeError("PaddleOCR vendor checkout is not clean")
    output_root.mkdir(parents=True, exist_ok=True)

    import paddle
    import yaml

    vendor_text = str(vendor_root)
    if vendor_text not in sys.path:
        sys.path.insert(0, vendor_text)
    from ppocr.modeling.architectures import build_model
    from ppocr.postprocess import build_post_process

    records: dict[str, dict[str, Any]] = {}
    for role, definition in MODELS.items():
        url = f"{OFFICIAL_BASE}/{definition['filename']}"
        destination = output_root / definition["filename"]
        download = download_with_resume(
            url,
            destination,
            retries=args.retries,
            force=args.force,
        )
        config_path = vendor_root / definition["config"]
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        resolve_vendor_config_paths(config, vendor_root)
        prepare_architecture_for_training(config, build_post_process)
        if config["Global"]["model_name"] != definition["model_name"]:
            raise RuntimeError(f"{role} config model-name mismatch")
        if config["Architecture"]["model_type"] != definition["model_type"]:
            raise RuntimeError(f"{role} config model-type mismatch")
        model = build_model(config["Architecture"])
        checkpoint_state = paddle.load(str(destination))
        if not isinstance(checkpoint_state, Mapping):
            raise RuntimeError(f"{role} checkpoint is not a state mapping")
        compatibility = compare_state_shapes(
            checkpoint_state,
            model.state_dict(),
        )
        if compatibility["model_parameter_match_ratio"] < 0.95:
            raise RuntimeError(
                f"{role} checkpoint architecture match is too low: {compatibility}"
            )
        if compatibility["shape_mismatch_count"]:
            raise RuntimeError(
                f"{role} checkpoint contains shape mismatches: {compatibility}"
            )
        matched_state = {
            key: value
            for key, value in checkpoint_state.items()
            if key in model.state_dict()
        }
        model.set_state_dict(matched_state)
        records[role] = {
            **definition,
            "source_url": url,
            "official_source": True,
            "path": destination.as_posix(),
            "size_bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
            "download": download,
            "config_sha256": sha256_file(config_path),
            "vendor_commit": vendor_commit,
            "compatibility": compatibility,
            "paddle_load": True,
            "model_set_state_dict": True,
            "partial_file_remaining": destination.with_suffix(
                destination.suffix + ".partial"
            ).exists(),
        }
        del model, checkpoint_state, matched_state

    source_manifest = (
        PROJECT_ROOT / "data" / "metadata" / "information_extraction_split_manifest.csv"
    )
    aggregate_hash = hashlib.sha256(
        json.dumps(
            {role: record["sha256"] for role, record in records.items()},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    source_commit = _git(PROJECT_ROOT, "rev-parse", "HEAD")
    report = {
        "schema_version": "1.0",
        "status": "passed",
        "build_id": f"ocr-pretrained-models-{aggregate_hash[:16]}",
        "split": "official_pretrained",
        "manifest_path": source_manifest.as_posix(),
        "manifest_sha256": sha256_file(source_manifest),
        "detector_sha256": records["detector"]["sha256"],
        "recognizer_sha256": records["general_recognizer"]["sha256"],
        "thai_recognizer_sha256": records["thai_recognizer"]["sha256"],
        "checkpoint_sha256": aggregate_hash,
        "calibration_sha256": None,
        "configuration_hash": hashlib.sha256(
            json.dumps(
                {
                    role: record["config_sha256"]
                    for role, record in records.items()
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "source_commit": source_commit,
        "device": "cpu-load-validation",
        "sample_count": len(records),
        "failure_count": 0,
        "duration_seconds": time.monotonic() - started,
        "private_row_count": 0,
        "paddle_version": paddle.__version__,
        "paddleocr_version": importlib.metadata.version("paddleocr"),
        "vendor": {
            "path": vendor_root.as_posix(),
            "tag": "v3.7.0",
            "commit": vendor_commit,
            "clean": True,
        },
        "models": records,
        "availability_note": (
            "The v3.7.0 detector documentation leaves its training-model "
            "anchor empty, but the exact official model-ecology object exists "
            "under the same naming convention and passed full architecture "
            "state-shape validation. No substitute architecture was used."
        ),
    }
    atomic_write_json(
        PROJECT_ROOT
        / "reports"
        / "ocr_upgrade"
        / "pretrained_training_models.json",
        report,
    )
    print(json.dumps(report, indent=2))
    return 0


def download_with_resume(
    url: str,
    destination: Path,
    *,
    retries: int,
    force: bool,
) -> dict[str, Any]:
    headers = _head(url)
    expected_size = int(headers["content_length"])
    expected_md5 = headers.get("content_md5")
    if destination.is_file() and not force:
        if destination.stat().st_size != expected_size:
            raise RuntimeError(
                f"existing checkpoint has unexpected size: {destination}"
            )
        _verify_md5(destination, expected_md5)
        return {
            **headers,
            "downloaded": False,
            "resumed": False,
            "attempts": 0,
            "existing_file_verified": True,
        }
    partial = destination.with_suffix(destination.suffix + ".partial")
    if force:
        partial.unlink(missing_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    attempts = 0
    resumed = False
    last_error: BaseException | None = None
    while attempts < retries:
        attempts += 1
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > expected_size:
                raise RuntimeError("partial download exceeds expected size")
            request_headers = {"User-Agent": "csx4201-ocr-upgrade/1.0"}
            mode = "wb"
            if 0 < offset < expected_size:
                request_headers["Range"] = f"bytes={offset}-"
                mode = "ab"
                resumed = True
            request = urllib.request.Request(url, headers=request_headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                status = getattr(response, "status", response.getcode())
                if offset and status != 206:
                    partial.unlink(missing_ok=True)
                    raise RuntimeError("server refused the requested byte range")
                with partial.open(mode) as handle:
                    while True:
                        chunk = response.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
            if partial.stat().st_size != expected_size:
                raise RuntimeError(
                    f"partial size {partial.stat().st_size} != {expected_size}"
                )
            _verify_md5(partial, expected_md5)
            os.replace(partial, destination)
            return {
                **headers,
                "downloaded": True,
                "resumed": resumed,
                "attempts": attempts,
                "existing_file_verified": False,
            }
        except (
            OSError,
            RuntimeError,
            urllib.error.HTTPError,
            urllib.error.URLError,
        ) as exc:
            last_error = exc
            if attempts >= retries:
                break
            time.sleep(attempts)
    raise RuntimeError(f"failed to download {url}") from last_error


def compare_state_shapes(
    checkpoint_state: Mapping[str, Any],
    model_state: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint_shapes = {
        key: tuple(int(value) for value in tensor.shape)
        for key, tensor in checkpoint_state.items()
        if hasattr(tensor, "shape")
    }
    model_shapes = {
        key: tuple(int(value) for value in tensor.shape)
        for key, tensor in model_state.items()
        if hasattr(tensor, "shape")
    }
    matched = sorted(
        key
        for key, shape in model_shapes.items()
        if checkpoint_shapes.get(key) == shape
    )
    mismatched = sorted(
        key
        for key in set(checkpoint_shapes) & set(model_shapes)
        if checkpoint_shapes[key] != model_shapes[key]
    )
    missing = sorted(set(model_shapes) - set(checkpoint_shapes))
    unexpected = sorted(set(checkpoint_shapes) - set(model_shapes))
    return {
        "checkpoint_tensor_count": len(checkpoint_shapes),
        "model_parameter_count": len(model_shapes),
        "matched_parameter_count": len(matched),
        "model_parameter_match_ratio": (
            len(matched) / len(model_shapes) if model_shapes else 0.0
        ),
        "shape_mismatch_count": len(mismatched),
        "missing_parameter_count": len(missing),
        "unexpected_parameter_count": len(unexpected),
        "shape_mismatch_examples": mismatched[:10],
        "missing_parameter_examples": missing[:10],
        "unexpected_parameter_examples": unexpected[:10],
    }


def _head(url: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        method="HEAD",
        headers={"User-Agent": "csx4201-ocr-upgrade/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        content_length = response.headers.get("Content-Length")
        if not content_length or int(content_length) < 1024 * 1024:
            raise RuntimeError(f"official checkpoint size is implausible: {url}")
        return {
            "http_status": getattr(response, "status", response.getcode()),
            "content_length": int(content_length),
            "content_type": response.headers.get("Content-Type"),
            "content_md5": response.headers.get("Content-MD5"),
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
            "accept_ranges": response.headers.get("Accept-Ranges"),
        }


def _verify_md5(path: Path, expected_base64: str | None) -> None:
    if not expected_base64:
        return
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    actual = base64.b64encode(digest.digest()).decode("ascii")
    if actual != expected_base64:
        raise RuntimeError(f"Content-MD5 mismatch for {path}")


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
