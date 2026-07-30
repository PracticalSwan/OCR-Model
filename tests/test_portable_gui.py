from __future__ import annotations

import threading
from pathlib import Path

import pytest
from PIL import Image

from src.portable.gui import (
    APP_CSS,
    FOOTER_LINKS,
    _clean_log_line,
    _gradio_blocked_paths,
    _on_document_change,
    _on_private_mode_change,
    _preview_document,
    _remove_gradio_temp_root,
    build_app,
)
from src.portable.api import ExtractionRun
from src.portable.runtime import RuntimeSettings


def test_preview_document_renders_uploaded_image(tmp_path: Path) -> None:
    source = tmp_path / "sample.png"
    Image.new("RGB", (48, 32), "navy").save(source)

    preview, note = _preview_document(str(source))

    assert isinstance(preview, Image.Image)
    assert preview.size == (48, 32)
    assert "sample.png" in note
    assert "Previewing the image locally" in note


def test_preview_document_waits_for_gradio_upload_copy(tmp_path: Path) -> None:
    source = tmp_path / "settling.png"

    def materialize_upload() -> None:
        Image.new("RGB", (20, 16), "navy").save(source)

    timer = threading.Timer(0.05, materialize_upload)
    timer.start()
    try:
        preview, note = _preview_document(str(source))
    finally:
        timer.join()

    assert isinstance(preview, Image.Image)
    assert preview.size == (20, 16)
    assert "settling.png" in note


def test_preview_document_renders_only_first_pdf_page(tmp_path: Path) -> None:
    fitz = pytest.importorskip("fitz")
    source = tmp_path / "sample.pdf"
    document = fitz.open()
    document.new_page(width=144, height=72)
    document.new_page(width=72, height=144)
    document.save(source)
    document.close()

    preview, note = _preview_document(str(source))

    assert isinstance(preview, Image.Image)
    assert preview.width > preview.height
    assert "first PDF page" in note


def test_preview_document_clears_when_no_file_is_selected() -> None:
    preview, note = _preview_document(None)

    assert preview is None
    assert note == "Upload an image or PDF to preview it here."


def test_document_change_clears_stale_results(tmp_path: Path) -> None:
    source = tmp_path / "replacement.png"
    Image.new("RGB", (12, 12), "white").save(source)

    outputs = _on_document_change(str(source))

    assert len(outputs) == 9
    assert "Ready to extract **replacement.png**" in outputs[2]
    assert outputs[3:] == ([], "", None, [], None, "")


def test_result_panes_have_bounded_independent_scroll_contract() -> None:
    assert "#ocr-text textarea" in APP_CSS
    assert "#run-log textarea" in APP_CSS
    assert "overflow-y: scroll !important" in APP_CSS
    assert "scrollbar-gutter: stable" in APP_CSS


def test_run_log_strips_terminal_color_sequences() -> None:
    assert _clean_log_line("\x1b[32mCreating model\x1b[0m") == "Creating model"


def test_private_document_change_hides_filename(tmp_path: Path) -> None:
    source = tmp_path / "Sensitive Private Invoice.png"
    Image.new("RGB", (12, 12), "white").save(source)

    preview, note, status, *_ = _on_document_change(str(source), True)

    assert preview is None
    assert "preview is disabled" in note.casefold()
    assert source.name not in note
    assert source.stem not in note
    assert source.name not in status
    assert source.stem not in status


def test_private_mode_change_only_updates_preview(tmp_path: Path) -> None:
    source = tmp_path / "private.png"
    Image.new("RGB", (12, 12), "white").save(source)

    outputs = _on_private_mode_change(str(source), True)

    assert len(outputs) == 2
    assert outputs[0] is None
    assert "preview is disabled" in outputs[1].casefold()


def test_gui_events_preserve_completed_results_and_upload(
    tmp_path: Path,
) -> None:
    pytest.importorskip("gradio")
    settings = object.__new__(RuntimeSettings)
    object.__setattr__(settings, "home", tmp_path)
    object.__setattr__(settings, "device", "cpu")
    object.__setattr__(settings, "output_root", tmp_path / "outputs")
    object.__setattr__(
        settings,
        "private_output_root",
        tmp_path / "outputs" / "private",
    )
    demo = build_app(settings, gradio_temp_root=tmp_path / "gradio")
    config = demo.get_config_file()
    component_ids = {
        component.get("props", {}).get("elem_id"): component["id"]
        for component in config["components"]
        if component.get("props", {}).get("elem_id")
    }

    extract_dependency = next(
        dependency
        for dependency in config["dependencies"]
        if any(
            target[0] == component_ids["extract-button"] and target[1] == "click"
            for target in dependency["targets"]
        )
    )
    private_dependency = next(
        dependency
        for dependency in config["dependencies"]
        if any(
            target[0] == component_ids["private-document"]
            and target[1] == "change"
            for target in dependency["targets"]
        )
    )

    assert component_ids["document-upload"] not in extract_dependency["outputs"]
    assert len(extract_dependency["outputs"]) == 7
    assert private_dependency["outputs"] == [
        component_ids["document-preview"],
        component_ids["document-preview-note"],
    ]


def test_gui_footer_does_not_expose_broken_settings_control() -> None:
    assert "settings" not in FOOTER_LINKS


def test_private_gui_returns_no_gallery_or_archive_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.portable import gui

    settings = object.__new__(RuntimeSettings)
    object.__setattr__(settings, "output_root", tmp_path / "outputs")
    object.__setattr__(settings, "private_output_root", tmp_path / "outputs" / "private")
    source = tmp_path / "Sensitive Private Invoice.png"
    Image.new("RGB", (12, 12), "white").save(source)
    output = settings.private_output_root / "run_opaque"
    output.mkdir(parents=True)
    run = ExtractionRun(
        input_path=source,
        output_dir=output,
        result_path=output / "document_result.json",
        payload={
            "document_id": source.stem,
            "fields": {},
            "pages": [],
            "processing": {"private_output": True},
        },
        command=(),
        private_document=True,
    )
    monkeypatch.setattr(gui, "run_extraction", lambda *args, **kwargs: run)

    first_result = gui._run_gui(
        str(source), "auto", "cpu", None, settings=settings, private_document=True
    )
    second_result = gui._run_gui(
        str(source), "auto", "cpu", None, settings=settings, private_document=True
    )

    assert first_result[4] == []
    assert first_result[5] is None
    assert second_result[4] == []
    assert second_result[5] is None
    assert source.is_file()
    assert source.name not in second_result[0]
    assert source.stem not in str(second_result[3])


def test_gradio_session_cache_is_removed_on_shutdown(tmp_path: Path) -> None:
    settings = object.__new__(RuntimeSettings)
    object.__setattr__(settings, "home", tmp_path)
    session_root = tmp_path / ".runtime" / "gradio" / "session_opaque"
    cached = session_root / "nested" / "upload.png"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"temporary")

    _remove_gradio_temp_root(session_root, settings)

    assert not session_root.exists()


def test_gradio_upload_cache_is_not_blocked(tmp_path: Path) -> None:
    settings = object.__new__(RuntimeSettings)
    object.__setattr__(settings, "home", tmp_path)
    object.__setattr__(settings, "output_root", tmp_path / "outputs")
    object.__setattr__(
        settings,
        "private_output_root",
        tmp_path / "outputs" / "private",
    )
    upload_cache = tmp_path / ".runtime" / "gradio" / "session_opaque"

    blocked = _gradio_blocked_paths(settings)

    assert str(settings.private_output_root) in blocked
    assert str(upload_cache) not in blocked
