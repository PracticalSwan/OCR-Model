#!/usr/bin/env python3
"""Verify the isolated PaddleOCR training environment and GPU lifecycle."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ocr.environment import configure_windows_nvidia_dlls  # noqa: E402
from src.rotation_common import atomic_write_json, sha256_file  # noqa: E402


EXPECTED_VENDOR_COMMIT = "b03f46425e8ff4442b268ce449e3eef758146cd4"
EXPECTED_PACKAGES = {
    "paddlepaddle-gpu": "3.3.0",
    "paddleocr": "3.7.0",
    "paddlex": "3.7.2",
    "nvidia-cuda-nvrtc": "13.0.48",
}
CONFIG_PATHS = {
    "detector": "configs/det/PP-OCRv6/PP-OCRv6_medium_det.yml",
    "general_recognizer": "configs/rec/PP-OCRv6/PP-OCRv6_medium_rec.yml",
    "thai_recognizer": (
        "configs/rec/PP-OCRv5/multi_language/th_PP-OCRv5_mobile_rec.yaml"
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--environment-root",
        default=(
            "D:/CSX4201/vision-info-extraction-assets/environments/ie-ocr-train"
        ),
    )
    parser.add_argument(
        "--vendor-root",
        default="D:/CSX4201/vision-info-extraction-assets/vendor/PaddleOCR",
    )
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()

    environment_root = Path(args.environment_root).resolve()
    vendor_root = Path(args.vendor_root).resolve()
    _require_expected_leaf(environment_root, "ie-ocr-train")
    _require_expected_leaf(vendor_root, "PaddleOCR")
    runtime_dll_directories = configure_training_runtime(environment_root)
    vendor_commit = _git(vendor_root, "rev-parse", "HEAD")
    if vendor_commit != EXPECTED_VENDOR_COMMIT:
        raise RuntimeError(f"unexpected PaddleOCR commit: {vendor_commit}")
    if _git(vendor_root, "status", "--porcelain"):
        raise RuntimeError("PaddleOCR vendor checkout is not clean")

    package_versions = {
        package: importlib.metadata.version(package)
        for package in EXPECTED_PACKAGES
    }
    if package_versions != EXPECTED_PACKAGES:
        raise RuntimeError(
            f"training package mismatch: {package_versions} != {EXPECTED_PACKAGES}"
        )

    import paddle
    import yaml

    if not paddle.is_compiled_with_cuda():
        raise RuntimeError("PaddlePaddle is not CUDA-enabled")
    if paddle.device.cuda.device_count() < 1:
        raise RuntimeError("PaddlePaddle cannot see a CUDA device")
    paddle.set_device(args.device)
    vendor_text = str(vendor_root)
    if vendor_text not in sys.path:
        sys.path.insert(0, vendor_text)
    from ppocr.losses import build_loss
    from ppocr.modeling.architectures import build_model
    from ppocr.optimizer import build_optimizer
    from ppocr.postprocess import build_post_process
    from tools import program

    del build_loss, build_optimizer, program
    model_summaries: dict[str, dict[str, Any]] = {}
    config_hashes: dict[str, str] = {}
    dictionary_hashes: dict[str, str] = {}
    for role, relative in CONFIG_PATHS.items():
        config_path = vendor_root / relative
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config_hashes[role] = sha256_file(config_path)
        resolve_vendor_config_paths(config, vendor_root)
        prepare_architecture_for_training(config, build_post_process)
        model = build_model(config["Architecture"])
        parameter_count = sum(
            _shape_product(parameter.shape) for parameter in model.parameters()
        )
        if parameter_count <= 0:
            raise RuntimeError(f"{role} initialized with no parameters")
        expected_type = "det" if role == "detector" else "rec"
        if config["Architecture"]["model_type"] != expected_type:
            raise RuntimeError(f"{role} model-type mismatch")
        character_dict = config.get("Global", {}).get("character_dict_path")
        if character_dict:
            dictionary_path = (vendor_root / character_dict).resolve()
            dictionary_hashes[role] = sha256_file(dictionary_path)
        model_summaries[role] = {
            "config_path": relative,
            "config_sha256": config_hashes[role],
            "model_name": config["Global"].get("model_name"),
            "model_type": config["Architecture"]["model_type"],
            "algorithm": config["Architecture"]["algorithm"],
            "parameter_count": parameter_count,
            "initialized": True,
        }
        del model

    lifecycle = _run_training_lifecycle(paddle, environment_root, args.device)
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    pip_check = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source_commit = _git(PROJECT_ROOT, "rev-parse", "HEAD")
    source_manifest = (
        PROJECT_ROOT / "data" / "metadata" / "information_extraction_split_manifest.csv"
    )
    report_material = {
        "vendor_commit": vendor_commit,
        "packages": package_versions,
        "configs": config_hashes,
        "device": args.device,
        "lifecycle_checkpoint_sha256": lifecycle["checkpoint_sha256"],
    }
    build_id = "ocr-training-environment-" + hashlib.sha256(
        json.dumps(
            report_material,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:16]
    report = {
        "schema_version": "1.0",
        "status": "passed",
        "build_id": build_id,
        "split": "environment_smoke",
        "manifest_path": source_manifest.as_posix(),
        "manifest_sha256": sha256_file(source_manifest),
        "detector_sha256": None,
        "recognizer_sha256": None,
        "checkpoint_sha256": lifecycle["checkpoint_sha256"],
        "calibration_sha256": None,
        "configuration_hash": hashlib.sha256(
            json.dumps(
                config_hashes,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "source_commit": source_commit,
        "device": args.device,
        "sample_count": 1,
        "failure_count": 0,
        "duration_seconds": time.monotonic() - started,
        "private_row_count": 0,
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "environment_root": environment_root.as_posix(),
            "runtime_dll_directories": runtime_dll_directories,
        },
        "packages": package_versions,
        "paddle": {
            "version": paddle.__version__,
            "compiled_with_cuda": paddle.is_compiled_with_cuda(),
            "cuda_version": paddle.version.cuda(),
            "cudnn_version": paddle.version.cudnn(),
            "device_count": paddle.device.cuda.device_count(),
            "device_name": paddle.device.cuda.get_device_name(),
            "device_capability": list(paddle.device.cuda.get_device_capability()),
            "active_device": paddle.device.get_device(),
        },
        "yaml": {
            "version": importlib.metadata.version("PyYAML"),
            "parser": "yaml.safe_load",
        },
        "vendor": {
            "path": vendor_root.as_posix(),
            "remote": _git(vendor_root, "remote", "get-url", "origin"),
            "tag": "v3.7.0",
            "commit": vendor_commit,
            "clean": True,
            "requirements_sha256": sha256_file(vendor_root / "requirements.txt"),
        },
        "official_model_initialization": model_summaries,
        "dictionary_hashes": dictionary_hashes,
        "training_lifecycle": lifecycle,
        "pip": {
            "check": pip_check or "No broken requirements found.",
            "freeze_sha256": hashlib.sha256(freeze.encode("utf-8")).hexdigest(),
            "freeze_count": len(freeze.splitlines()),
        },
    }
    if args.write_report:
        atomic_write_json(
            PROJECT_ROOT / "reports" / "ocr_upgrade" / "training_environment.json",
            report,
        )
    print(json.dumps(report, indent=2))
    return 0


def _run_training_lifecycle(
    paddle: Any,
    environment_root: Path,
    device: str,
) -> dict[str, Any]:
    paddle.seed(42)
    model = paddle.nn.Sequential(
        paddle.nn.Conv2D(1, 4, kernel_size=3, padding=1),
        paddle.nn.ReLU(),
        paddle.nn.AdaptiveAvgPool2D((1, 1)),
        paddle.nn.Flatten(),
        paddle.nn.Linear(4, 2),
    )
    optimizer = paddle.optimizer.Adam(
        learning_rate=1e-3,
        parameters=model.parameters(),
    )
    inputs = paddle.randn([2, 1, 16, 16], dtype="float32")
    targets = paddle.to_tensor([0, 1], dtype="int64")
    logits = model(inputs)
    loss = paddle.nn.functional.cross_entropy(logits, targets)
    if not bool(paddle.isfinite(loss).item()):
        raise RuntimeError("training smoke loss is not finite")
    loss.backward()
    gradient_count = sum(
        1 for parameter in model.parameters() if parameter.grad is not None
    )
    if gradient_count == 0:
        raise RuntimeError("training smoke produced no gradients")
    optimizer.step()
    optimizer.clear_grad()

    verification_root = environment_root / "verification"
    verification_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = verification_root / "training_lifecycle.pdparams"
    paddle.save(model.state_dict(), str(checkpoint_path))
    reloaded = paddle.nn.Sequential(
        paddle.nn.Conv2D(1, 4, kernel_size=3, padding=1),
        paddle.nn.ReLU(),
        paddle.nn.AdaptiveAvgPool2D((1, 1)),
        paddle.nn.Flatten(),
        paddle.nn.Linear(4, 2),
    )
    reloaded.set_state_dict(paddle.load(str(checkpoint_path)))
    with paddle.no_grad():
        reloaded_logits = reloaded(inputs)
    if tuple(reloaded_logits.shape) != (2, 2):
        raise RuntimeError("reloaded model output shape mismatch")
    amp_model = paddle.nn.Linear(8, 4)
    amp_optimizer = paddle.optimizer.Adam(
        learning_rate=1e-3,
        parameters=amp_model.parameters(),
    )
    amp_model, amp_optimizer = paddle.amp.decorate(
        models=amp_model,
        optimizers=amp_optimizer,
        level="O2",
        master_weight=True,
    )
    amp_scaler = paddle.amp.GradScaler(
        init_loss_scaling=32768.0,
        use_dynamic_loss_scaling=True,
    )
    amp_inputs = paddle.randn([4, 8], dtype="float32")
    with paddle.amp.auto_cast(level="O2"):
        amp_outputs = amp_model(amp_inputs)
        amp_loss = paddle.mean(amp_outputs * amp_outputs)
    if not bool(paddle.isfinite(amp_loss).item()):
        raise RuntimeError("O2 AMP smoke loss is not finite")
    scaled_loss = amp_scaler.scale(amp_loss)
    scaled_loss.backward()
    amp_scaler.minimize(amp_optimizer, scaled_loss)
    amp_optimizer.clear_grad()
    return {
        "device": device,
        "forward_pass": True,
        "backward_pass": True,
        "optimizer_step": True,
        "gradient_parameter_count": gradient_count,
        "loss": float(loss.item()),
        "checkpoint_path": checkpoint_path.as_posix(),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_reload": True,
        "output_shape": list(reloaded_logits.shape),
        "amp_o2_forward_pass": True,
        "amp_o2_backward_pass": True,
        "amp_o2_optimizer_step": True,
        "amp_o2_loss": float(amp_loss.item()),
    }


def prepare_architecture_for_training(
    config: dict[str, Any],
    build_post_process: Any,
) -> None:
    """Apply the official train.py dictionary-derived output-channel setup."""
    architecture = config["Architecture"]
    if architecture.get("model_type") != "rec":
        return
    post_process = build_post_process(
        config["PostProcess"],
        config.get("Global", {}),
    )
    if not hasattr(post_process, "character"):
        raise RuntimeError("recognition postprocessor has no character inventory")
    character_count = len(post_process.character)
    head = architecture["Head"]
    if head.get("name") != "MultiHead":
        head["out_channels"] = character_count
        return
    out_channels = {"CTCLabelDecode": character_count}
    losses = config.get("Loss", {}).get("loss_config_list", [])
    if any("NRTRLoss" in entry for entry in losses):
        out_channels["NRTRLabelDecode"] = character_count + 3
    elif any("SARLoss" in entry for entry in losses):
        out_channels["SARLabelDecode"] = character_count + 2
    else:
        raise RuntimeError("MultiHead recognizer lacks an auxiliary decoder loss")
    head["out_channels_list"] = out_channels


def resolve_vendor_config_paths(
    config: dict[str, Any],
    vendor_root: Path,
) -> None:
    """Resolve repository-relative dictionary paths without changing the vendor tree."""
    character_dict = config.get("Global", {}).get("character_dict_path")
    if character_dict and not Path(character_dict).is_absolute():
        config["Global"]["character_dict_path"] = str(
            (vendor_root / character_dict).resolve()
        )


def _shape_product(shape: Any) -> int:
    result = 1
    for dimension in shape:
        result *= int(dimension)
    return result


def _require_expected_leaf(path: Path, expected: str) -> None:
    if path.name.casefold() != expected.casefold():
        raise ValueError(f"unexpected path {path}; expected leaf {expected}")
    if not path.is_dir():
        raise FileNotFoundError(path)


def configure_training_runtime(environment_root: Path) -> list[str]:
    """Register the training environment's bundled CUDA DLL directories."""
    directories = configure_windows_nvidia_dlls(environment_root)
    if os.name == "nt" and not directories:
        raise RuntimeError(
            "training environment has no bundled CUDA/cuDNN DLL directories"
        )
    return directories


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
