"""One-command local CLI for the finished OCR information-extraction model."""
from __future__ import annotations

import argparse
import json
import sys

from .api import ExtractionError, _display_path, run_extraction
from .results import field_rows
from .runtime import RuntimeSettings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run full local OCR + layout information extraction on one image or PDF."
    )
    parser.add_argument("input", help="image or PDF to extract")
    parser.add_argument("-o", "--output", help="output folder (default: timestamped outputs folder)")
    parser.add_argument(
        "--language",
        choices=("auto", "general", "thai", "en", "tr", "th"),
        default="auto",
    )
    parser.add_argument("--device", choices=("cpu", "gpu:0"))
    parser.add_argument(
        "--ocr-profile",
        choices=("original", "auto"),
        default="auto",
        help=(
            "portable calibrated extraction supports the original profile; "
            "experimental profiles require the lower-level CLI and explicit "
            "generic-layout fallback"
        ),
    )
    parser.add_argument(
        "--detector-model",
        choices=("original", "auto"),
        default="auto",
    )
    parser.add_argument(
        "--general-recognizer",
        choices=("original", "auto"),
        default="auto",
    )
    parser.add_argument(
        "--thai-recognizer",
        choices=("original", "auto"),
        default="auto",
    )
    parser.add_argument("--max-pages", type=int)
    parser.add_argument("--no-visualization", action="store_true")
    parser.add_argument(
        "--private-document",
        action="store_true",
        help="store this document under the protected private output root",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    try:
        runtime = RuntimeSettings.load()
        run = run_extraction(
            args.input,
            settings=runtime,
            output_dir=args.output,
            language=args.language,
            device=args.device,
            max_pages=args.max_pages,
            save_visualization=not args.no_visualization,
            private_document=args.private_document,
            on_log=None if args.quiet else lambda line: print(line, file=sys.stderr),
            ocr_profile=args.ocr_profile,
            detector_model=args.detector_model,
            general_recognizer=args.general_recognizer,
            thai_recognizer=args.thai_recognizer,
        )
    except (ExtractionError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "status": "complete",
                "result": _display_path(
                    run.result_path, private_document=run.private_document
                ),
                "output_directory": _display_path(
                    run.output_dir, private_document=run.private_document
                ),
                "fields": [
                    {
                        "name": row[0],
                        "value": row[1],
                        "confidence": row[2],
                    }
                    for row in field_rows(run.payload)
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0
