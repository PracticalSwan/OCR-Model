#!/usr/bin/env python3
"""Build profile-bound public ground-truth, PaddleOCR, and hybrid streams."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config as cfgmod  # noqa: E402
from src.information_extraction.model_dataset import prepare_model_dataset  # noqa: E402
from src.ocr.environment import configure_external_environment, require_storage_gate  # noqa: E402
from src.ocr.model_registry import ModelRegistry  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--profile", choices=("smoke", "development", "final"), default="smoke")
    parser.add_argument("--device", choices=("cpu", "gpu:0"), default="cpu")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--ocr-variant-limit",
        type=int,
        default=0,
        help="Bound only PaddleOCR/hybrid variants; ground-truth pages remain complete.",
    )
    parser.add_argument("--ocr-variant-train-limit", type=int)
    parser.add_argument("--ocr-variant-dev-select-limit", type=int)
    parser.add_argument(
        "--manifest-output",
        help="optional profile-bound manifest path (for example final_model_dataset_manifest_ocr_v2.csv)",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--streams",
        nargs="+",
        choices=("ground_truth", "paddleocr", "hybrid", "ocr_noise"),
        default=("ground_truth",),
    )
    parser.add_argument("--model-setup", default=str(PROJECT_ROOT / "reports" / "ocr" / "model_setup.json"))
    parser.add_argument(
        "--model-registry",
        default=str(PROJECT_ROOT / "reports" / "ocr_upgrade" / "model_registry.json"),
    )
    parser.add_argument(
        "--detector-model",
        choices=("original", "custom", "auto"),
        default="auto",
    )
    parser.add_argument(
        "--general-recognizer",
        choices=("original", "custom", "auto"),
        default="auto",
    )
    parser.add_argument(
        "--thai-recognizer",
        choices=("original", "custom", "auto"),
        default="auto",
    )
    args = parser.parse_args()
    if (
        args.limit < 0
        or args.ocr_variant_limit < 0
        or any(
            value is not None and value < 0
            for value in (
                args.ocr_variant_train_limit,
                args.ocr_variant_dev_select_limit,
            )
        )
    ):
        parser.error("--limit and --ocr-variant-limit must be non-negative")
    if args.ocr_variant_limit and (
        args.ocr_variant_train_limit is not None
        or args.ocr_variant_dev_select_limit is not None
    ):
        parser.error(
            "--ocr-variant-limit cannot be combined with split-specific limits"
        )
    cfg = cfgmod.load_config(args.config)
    asset_root = cfgmod.resolve_path(cfg, "external_assets")
    configure_external_environment(asset_root)
    anticipated_asset_gib = {"smoke": 1.0, "development": 5.0, "final": 15.0}[args.profile]
    require_storage_gate(
        asset_root,
        operation=f"{args.profile} model-dataset preparation",
        anticipated_c_gib=0.25,
        anticipated_asset_gib=anticipated_asset_gib,
    )
    registry = ModelRegistry.from_setup(
        args.model_setup,
        upgrade_registry=args.model_registry,
        detector_choice=args.detector_model,
        general_choice=args.general_recognizer,
        thai_choice=args.thai_recognizer,
    )
    split_limits = None
    if (
        args.ocr_variant_train_limit is not None
        or args.ocr_variant_dev_select_limit is not None
    ):
        split_limits = {
            "train": int(args.ocr_variant_train_limit or 0),
            "dev_select": int(args.ocr_variant_dev_select_limit or 0),
        }
    summary = prepare_model_dataset(
        cfg,
        registry,
        profile=args.profile,
        device=args.device,
        limit=args.limit,
        force=args.force,
        streams=tuple(args.streams),
        ocr_variant_limit=args.ocr_variant_limit,
        ocr_variant_split_limits=split_limits,
        manifest_path_override=args.manifest_output,
    )
    import json

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
