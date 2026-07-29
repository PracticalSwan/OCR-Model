#!/usr/bin/env python3
"""Run, evaluate, export, and fingerprint one bounded official PaddleOCR trial."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ocr.build_support import check_storage_gate  # noqa: E402
from src.ocr.training_runner import (  # noqa: E402
    load_trial_definition,
    resolve_paddle_config,
)
from src.ocr.trials import sha256_tree  # noqa: E402
from src.rotation_common import (  # noqa: E402
    atomic_write_json,
    atomic_write_text,
    canonical_json,
    sha256_file,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--definition", required=True)
    parser.add_argument(
        "--kind",
        required=True,
        choices=("detector", "general_recognizer", "thai_recognizer"),
    )
    parser.add_argument(
        "--vendor-root",
        default="D:/CSX4201/vision-info-extraction-assets/vendor/PaddleOCR",
    )
    parser.add_argument(
        "--environment-root",
        default="D:/CSX4201/vision-info-extraction-assets/environments/ie-ocr-train",
    )
    parser.add_argument(
        "--artifact-root",
        default="D:/CSX4201/vision-info-extraction-assets/checkpoints/ocr_upgrade",
    )
    args = parser.parse_args()

    started = time.monotonic()
    definition = load_trial_definition(
        args.definition,
        expected_kind=args.kind,
    )
    vendor_root = Path(args.vendor_root).resolve()
    environment_root = Path(args.environment_root).resolve()
    python = environment_root / "Scripts/python.exe"
    if not python.is_file():
        raise FileNotFoundError(python)
    artifact_parent = (Path(args.artifact_root).resolve() / args.kind)
    artifact_parent.mkdir(parents=True, exist_ok=True)
    output_root = (artifact_parent / definition.trial_id).resolve()
    if output_root.parent != artifact_parent:
        raise ValueError(f"trial output escapes artifact parent: {output_root}")
    if output_root.exists():
        raise FileExistsError(
            f"trial output already exists; use a new immutable trial ID: {output_root}"
        )
    check_storage_gate(
        project_root=PROJECT_ROOT,
        output_root=artifact_parent,
        anticipated_output_bytes=(
            5 * 1024**3 if args.kind == "detector" else 8 * 1024**3
        ),
    )
    output_root.mkdir()
    training_root = output_root / "training"
    export_root = output_root / "exported"
    logs_root = output_root / "logs"
    logs_root.mkdir()
    resolved = resolve_paddle_config(
        definition,
        vendor_root=vendor_root,
        output_root=output_root,
    )
    resolved_path = output_root / "resolved_config.yml"
    atomic_write_text(
        resolved_path,
        yaml.safe_dump(
            resolved,
            sort_keys=False,
            allow_unicode=True,
            width=120,
        ),
    )
    shutil.copy2(definition.path, output_root / "trial_definition.yml")
    environment_report = PROJECT_ROOT / "reports/ocr_upgrade/training_environment.json"
    if environment_report.is_file():
        shutil.copy2(environment_report, output_root / "training_environment.json")

    source_commit = _git_commit(PROJECT_ROOT)
    vendor_commit = _git_commit(vendor_root)
    if _git_status(vendor_root):
        raise RuntimeError("pinned PaddleOCR vendor checkout is dirty")
    runtime_environment = _runtime_environment(environment_root, vendor_root)
    train_command = [
        str(python),
        "tools/train.py",
        "-c",
        str(resolved_path),
    ]
    metadata: dict[str, Any] = {
        "schema_version": "1.0",
        "status": "running",
        "trial_id": definition.trial_id,
        "kind": definition.kind,
        "definition_path": definition.path.as_posix(),
        "definition_sha256": sha256_file(definition.path),
        "resolved_config_path": resolved_path.as_posix(),
        "resolved_config_sha256": sha256_file(resolved_path),
        "source_checkpoint": definition.source_checkpoint.as_posix(),
        "source_checkpoint_sha256": sha256_file(definition.source_checkpoint),
        "source_commit": source_commit,
        "vendor_root": vendor_root.as_posix(),
        "vendor_commit": vendor_commit,
        "environment_root": environment_root.as_posix(),
        "output_root": output_root.as_posix(),
        "training_command": train_command,
        "metadata": definition.metadata,
        "private_row_count": 0,
    }
    atomic_write_json(output_root / "run_metadata.json", metadata)

    train_result = _run_phase(
        "train",
        train_command,
        cwd=vendor_root,
        environment=runtime_environment,
        log_path=logs_root / "train.log",
    )
    metadata["training"] = train_result
    if train_result["return_code"] != 0:
        metadata["status"] = "failed"
        metadata["duration_seconds"] = time.monotonic() - started
        atomic_write_json(output_root / "run_metadata.json", metadata)
        raise SystemExit(train_result["return_code"])

    checkpoint_prefix = _select_checkpoint(training_root)
    checkpoint_files = sorted(
        path
        for path in checkpoint_prefix.parent.glob(checkpoint_prefix.name + ".*")
        if path.is_file()
    )
    metadata["selected_checkpoint_prefix"] = checkpoint_prefix.as_posix()
    metadata["selected_checkpoint_files"] = [
        {
            "path": path.as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in checkpoint_files
    ]
    params_path = checkpoint_prefix.with_suffix(".pdparams")
    metadata["selected_checkpoint_sha256"] = sha256_file(params_path)

    eval_command = [
        str(python),
        "tools/eval.py",
        "-c",
        str(resolved_path),
        "-o",
        f"Global.checkpoints={checkpoint_prefix.as_posix()}",
    ]
    eval_result = _run_phase(
        "eval",
        eval_command,
        cwd=vendor_root,
        environment=runtime_environment,
        log_path=logs_root / "eval.log",
    )
    metadata["official_eval"] = eval_result
    if eval_result["return_code"] != 0:
        metadata["status"] = "failed"
        metadata["duration_seconds"] = time.monotonic() - started
        atomic_write_json(output_root / "run_metadata.json", metadata)
        raise SystemExit(eval_result["return_code"])

    export_command = [
        str(python),
        "tools/export_model.py",
        "-c",
        str(resolved_path),
        "-o",
        f"Global.pretrained_model={checkpoint_prefix.as_posix()}",
        f"Global.save_inference_dir={export_root.as_posix()}",
    ]
    export_result = _run_phase(
        "export",
        export_command,
        cwd=vendor_root,
        environment=runtime_environment,
        log_path=logs_root / "export.log",
    )
    metadata["export"] = export_result
    if export_result["return_code"] != 0:
        metadata["status"] = "failed"
        metadata["duration_seconds"] = time.monotonic() - started
        atomic_write_json(output_root / "run_metadata.json", metadata)
        raise SystemExit(export_result["return_code"])
    _verify_export(export_root)

    metadata["export_sha256"] = sha256_tree(export_root)
    metadata["peak_gpu_memory_mib"] = max(
        int(train_result["peak_gpu_memory_mib"]),
        int(eval_result["peak_gpu_memory_mib"]),
        int(export_result["peak_gpu_memory_mib"]),
    )
    metadata["duration_seconds"] = time.monotonic() - started
    metadata["status"] = "passed"
    metadata["vendor_clean_after"] = not bool(_git_status(vendor_root))
    metadata["artifact_manifest"] = _artifact_manifest(output_root)
    metadata["artifact_manifest_sha256"] = hashlib.sha256(
        canonical_json(metadata["artifact_manifest"]).encode("utf-8")
    ).hexdigest()
    atomic_write_json(output_root / "run_metadata.json", metadata)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


def _run_phase(
    name: str,
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    log_path: Path,
) -> dict[str, Any]:
    print(f"{name}: {' '.join(command)}", flush=True)
    started = time.monotonic()
    peak_memory = 0
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        last_update = started
        while process.poll() is None:
            peak_memory = max(peak_memory, _gpu_memory_mib())
            now = time.monotonic()
            if now - last_update >= 60:
                print(
                    f"{name}: {int(now - started)}s elapsed; "
                    f"peak GPU memory {peak_memory} MiB",
                    flush=True,
                )
                last_update = now
            time.sleep(2.0)
        return_code = int(process.returncode)
    duration = time.monotonic() - started
    print(
        f"{name}: return_code={return_code}, duration={duration:.1f}s, "
        f"peak_gpu_memory={peak_memory} MiB",
        flush=True,
    )
    return {
        "command": command,
        "return_code": return_code,
        "duration_seconds": duration,
        "peak_gpu_memory_mib": peak_memory,
        "log_path": log_path.as_posix(),
        "log_sha256": sha256_file(log_path),
    }


def _select_checkpoint(training_root: Path) -> Path:
    for name in ("best_accuracy", "latest"):
        prefix = training_root / name
        if prefix.with_suffix(".pdparams").is_file():
            return prefix
    candidates = sorted(
        training_root.glob("iter_epoch_*.pdparams"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0].with_suffix("")
    raise FileNotFoundError(f"no exported training checkpoint under {training_root}")


def _verify_export(path: Path) -> None:
    required = ("inference.json", "inference.pdiparams", "inference.yml")
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise RuntimeError(f"incomplete Paddle inference export: {missing}")
    if any((path / name).stat().st_size <= 0 for name in required):
        raise RuntimeError("Paddle inference export contains an empty file")


def _runtime_environment(
    environment_root: Path,
    vendor_root: Path,
) -> dict[str, str]:
    environment = dict(os.environ)
    cuda_paths = [
        environment_root / "Lib/site-packages/nvidia/cu13/bin/x86_64",
        environment_root / "Lib/site-packages/nvidia/cudnn/bin",
        environment_root / "Scripts",
    ]
    environment["PATH"] = os.pathsep.join(
        [*(str(path) for path in cuda_paths if path.is_dir()), environment["PATH"]]
    )
    environment["PYTHONPATH"] = str(vendor_root)
    environment["FLAGS_allocator_strategy"] = "auto_growth"
    environment["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    return environment


def _gpu_memory_mib() -> int:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        return max(int(line.strip()) for line in output.splitlines() if line.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0


def _artifact_manifest(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        if path.name == "run_metadata.json":
            continue
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def _git_commit(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
    ).strip()


def _git_status(root: Path) -> str:
    return subprocess.check_output(
        ["git", "status", "--short"],
        cwd=root,
        text=True,
    ).strip()


if __name__ == "__main__":
    raise SystemExit(main())
