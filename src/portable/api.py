"""Stable subprocess API around the verified OCR/layout extraction CLI."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .results import load_result
from .runtime import RuntimeSettings


SUPPORTED_SUFFIXES = {
    ".bmp", ".jpeg", ".jpg", ".pdf", ".png", ".tif", ".tiff", ".webp"
}


class ExtractionError(RuntimeError):
    """Raised when the existing model worker cannot finish an extraction."""


@dataclass(frozen=True)
class ExtractionRun:
    input_path: Path
    output_dir: Path
    result_path: Path
    payload: dict
    command: tuple[str, ...]
    private_document: bool = False


def _safe_stem(path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._")
    return (stem or "document")[:60]


def default_output_dir(
    settings: RuntimeSettings,
    input_path: Path,
    *,
    private_document: bool = False,
) -> Path:
    if private_document:
        return private_output_dir(settings)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = settings.output_root / f"{timestamp}_{_safe_stem(input_path)}"
    candidate = base
    suffix = 2
    while candidate.exists():
        candidate = Path(f"{base}_{suffix}")
        suffix += 1
    return candidate


def private_output_dir(settings: RuntimeSettings) -> Path:
    """Return a fresh opaque destination below the configured private root."""
    root = _private_root(settings)
    root.mkdir(parents=True, exist_ok=True)
    # UUIDs avoid source-derived names and are safe to display as opaque handles.
    return root / f"run_{uuid.uuid4()}"


def _private_root(settings: RuntimeSettings) -> Path:
    configured = settings.private_output_root or (settings.output_root / "private")
    return Path(configured).expanduser().resolve()


def _redaction_replacements(source: Path, private_root: Path) -> tuple[str, ...]:
    values = {
        str(source),
        source.as_posix(),
        source.name,
        source.stem,
        str(private_root),
        private_root.as_posix(),
    }
    return tuple(sorted((value for value in values if value), key=len, reverse=True))


def _redact_text(text: str, source: Path, private_root: Path) -> str:
    redacted = text
    stem = source.stem
    for value in _redaction_replacements(source, private_root):
        if value == stem:
            continue
        redacted = redacted.replace(value, "<private>")
    if stem:
        # Keep structured keys such as ``private_output`` intact when a user
        # happens to upload ``private.pdf`` while still removing standalone
        # document-id/log occurrences of the exact source stem.
        redacted = re.sub(
            rf"(?<![A-Za-z0-9]){re.escape(stem)}(?![A-Za-z0-9])",
            "<private>",
            redacted,
        )
    return redacted


def redact_private_payload(value, source: Path, private_root: Path):
    """Recursively redact source/root identifiers from a displayed payload."""
    if isinstance(value, dict):
        return {
            key: redact_private_payload(item, source, private_root)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_private_payload(item, source, private_root) for item in value]
    if isinstance(value, str):
        return _redact_text(value, source, private_root)
    return value


def _redact_private_text_outputs(
    destination: Path,
    source: Path,
    private_root: Path,
) -> None:
    """Redact the fixed set of text outputs written by the extraction worker."""
    paths = [
        destination / "document_result.json",
        destination / "portable_run.log",
        destination / "logs" / "inference.log",
    ]
    pages = destination / "pages"
    if pages.is_dir():
        paths.extend(sorted(pages.glob("page_*.json")))
    for path in paths:
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        redacted = _redact_text(text, source, private_root)
        if redacted != text:
            path.write_text(redacted, encoding="utf-8")


def _display_path(path: Path, *, private_document: bool) -> str:
    if not private_document:
        return str(path)
    # Never return the configured private root (or an absolute path) to a caller.
    if path.parent.name.startswith("run_"):
        return f"{path.parent.name}/{path.name}"
    return path.name if path.name else "private-run"


def _stage_private_input(source: Path, destination: Path) -> tuple[Path, Path]:
    """Copy a private source to an opaque, short-lived worker input path."""
    stage = destination / ".private_input"
    stage.mkdir(parents=True, exist_ok=False)
    staged_source = stage / f"document{source.suffix.casefold()}"
    try:
        shutil.copyfile(source, staged_source)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return staged_source, stage


def _remove_private_input_stage(stage: Path, destination: Path) -> None:
    """Delete only the exact private-input stage created for this run."""
    stage_absolute = Path(stage).absolute()
    destination_absolute = Path(destination).absolute()
    if (
        stage_absolute.name != ".private_input"
        or stage_absolute.parent != destination_absolute
    ):
        raise ExtractionError("refusing to remove an unexpected private input path")
    if stage_absolute.exists():
        shutil.rmtree(stage_absolute)


def build_command(
    settings: RuntimeSettings,
    input_path: Path,
    output_dir: Path,
    *,
    language: str = "auto",
    device: str | None = None,
    max_pages: int | None = None,
    save_visualization: bool = True,
    ocr_profile: str = "auto",
    detector_model: str = "auto",
    general_recognizer: str = "auto",
    thai_recognizer: str = "auto",
    private_document: bool = False,
) -> list[str]:
    command = [
        str(settings.ocr_python),
        str(settings.home / "scripts" / "extract_document.py"),
        "--input",
        str(input_path),
        "--output",
        str(output_dir),
        "--config",
        str(settings.config),
        "--model-setup",
        str(settings.model_setup),
        "--model-checkpoint",
        str(settings.layout_checkpoint),
        "--language",
        language,
        "--device",
        device or settings.device,
        "--force",
        "--ocr-profile",
        ocr_profile,
        "--detector-model",
        detector_model,
        "--general-recognizer",
        general_recognizer,
        "--thai-recognizer",
        thai_recognizer,
    ]
    if private_document:
        # Private runs are always private-output and never materialize display
        # artifacts, including the display-only K-Means branch.
        command.append("--private-output")
        command.append("--disable-kmeans-display")
    elif save_visualization:
        command.append("--save-visualization")
    if max_pages is not None:
        command.extend(["--max-pages", str(max_pages)])
    return command


def run_extraction(
    input_path: str | Path,
    *,
    settings: RuntimeSettings | None = None,
    output_dir: str | Path | None = None,
    language: str = "auto",
    device: str | None = None,
    max_pages: int | None = None,
    save_visualization: bool = True,
    on_log: Callable[[str], None] | None = None,
    ocr_profile: str = "auto",
    detector_model: str = "auto",
    general_recognizer: str = "auto",
    thai_recognizer: str = "auto",
    private_document: bool = False,
) -> ExtractionRun:
    runtime = settings or RuntimeSettings.load()
    source = Path(input_path).expanduser().resolve()
    if not source.is_file():
        message = f"input file not found: {source}"
        if private_document:
            message = _redact_text(message, source, _private_root(runtime))
        raise ExtractionError(message)
    if source.suffix.casefold() not in SUPPORTED_SUFFIXES:
        allowed = ", ".join(sorted(SUPPORTED_SUFFIXES))
        message = f"unsupported input type {source.suffix!r}; expected one of: {allowed}"
        if private_document:
            message = _redact_text(message, source, _private_root(runtime))
        raise ExtractionError(message)
    if not runtime.ready:
        missing = [item["name"] for item in runtime.checks() if not item["ok"]]
        raise ExtractionError(
            "runtime is not ready; run the setup/doctor first. Missing: "
            + ", ".join(missing)
        )

    try:
        destination = (
            default_output_dir(runtime, source, private_document=True)
            if private_document
            else (
                Path(output_dir).expanduser().resolve()
                if output_dir
                else default_output_dir(runtime, source)
            )
        )
    except OSError as exc:
        detail = str(exc)
        if private_document:
            detail = _redact_text(detail, source, _private_root(runtime))
        raise ExtractionError(f"could not prepare private output: {detail}") from exc
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        detail = str(exc)
        if private_document:
            detail = _redact_text(detail, source, _private_root(runtime))
        raise ExtractionError(f"could not prepare extraction output: {detail}") from exc
    execution_source = source
    private_input_stage: Path | None = None
    if private_document:
        try:
            execution_source, private_input_stage = _stage_private_input(
                source,
                destination,
            )
        except OSError as exc:
            detail = _redact_text(str(exc), source, _private_root(runtime))
            raise ExtractionError(
                f"could not stage private document: {detail}"
            ) from exc
    command = build_command(
        runtime,
        execution_source,
        destination,
        language=language,
        device=device,
        max_pages=max_pages,
        save_visualization=False if private_document else save_visualization,
        ocr_profile=ocr_profile,
        detector_model=detector_model,
        general_recognizer=general_recognizer,
        thai_recognizer=thai_recognizer,
        private_document=private_document,
    )
    lines: list[str] = []
    try:
        try:
            process = subprocess.Popen(
                command,
                cwd=runtime.home,
                env=runtime.environment(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except OSError as exc:
            detail = str(exc)
            if private_document:
                detail = _redact_text(detail, source, _private_root(runtime))
            raise ExtractionError(f"could not start model worker: {detail}") from exc
        assert process.stdout is not None
        for line in process.stdout:
            clean = line.rstrip()
            if private_document:
                clean = _redact_text(clean, source, _private_root(runtime))
            lines.append(clean)
            if on_log:
                on_log(clean)
        return_code = process.wait()
        if destination.exists():
            (destination / "portable_run.log").write_text(
                "\n".join(lines) + ("\n" if lines else ""),
                encoding="utf-8",
            )
            if private_document:
                _redact_private_text_outputs(
                    destination,
                    source,
                    _private_root(runtime),
                )
        if return_code != 0:
            detail = "\n".join(lines[-20:]) or "worker exited without output"
            if private_document:
                detail = _redact_text(detail, source, _private_root(runtime))
            raise ExtractionError(
                f"model extraction failed with exit code {return_code}:\n{detail}"
            )
        result_path = destination / "document_result.json"
        if not result_path.is_file():
            raise ExtractionError(
                "model worker reported success but document_result.json was not created"
            )
        try:
            payload = load_result(result_path)
        except Exception as exc:
            detail = str(exc)
            if private_document:
                detail = _redact_text(detail, source, _private_root(runtime))
            raise ExtractionError(
                f"could not read extraction result: {detail}"
            ) from exc
    finally:
        if private_input_stage is not None:
            _remove_private_input_stage(private_input_stage, destination)
    return ExtractionRun(
        input_path=Path("<private-input>") if private_document else source,
        output_dir=destination,
        result_path=result_path,
        payload=payload,
        command=tuple(
            _redact_text(item, source, _private_root(runtime))
            for item in command
        )
        if private_document
        else tuple(command),
        private_document=private_document,
    )


def run_to_json(*args, **kwargs) -> str:
    run = run_extraction(*args, **kwargs)
    return json.dumps(
        {
            "status": "complete",
            "result": _display_path(run.result_path, private_document=run.private_document),
            "output_directory": _display_path(run.output_dir, private_document=run.private_document),
        },
        indent=2,
    )
