# OCR Model portable package

This package runs the finished PaddleOCR + LayoutXLM information-extraction
pipeline on local images and PDFs. It includes the trained LayoutXLM checkpoint,
the three pinned PaddleOCR model directories, calibration, rotation-display
artifacts, schemas, launchers, and a safe synthetic sample. It does not include
raw training data, private Gmail documents, private outputs, credentials, or an
OpenAI API integration.

Extraction is local and does not require an OpenAI API key. The optional Codex
review workflow uses the user's signed-in Codex session and is described in
`docs/CODEX_INTEGRATION.md`.

## Windows quick start

Requirements:

- Windows 10 or 11, 64-bit
- Python 3.10 from python.org, with the `py` launcher enabled
- Internet access for the one-time dependency installation
- At least 20 GB of free disk space for Python environments and Docker caches

Run:

1. Extract `OCR_Model.zip` to a normal writable folder.
2. Double-click `setup_windows.bat`. The default setup is CPU-only and works
   without NVIDIA hardware.
3. Double-click `launch_windows.bat`.
4. Select an image/PDF and click **Extract document**.

The GUI starts in private mode. Private uploads are not previewed. After
deliberately clearing **Private document** for public material, the upload card
previews the complete selected image or the first page of a PDF. Selecting a
different document clears the previous status and results. A run uses one
compact loading indicator; it does not cover each output tab with separate
spinners.

After setup, the model can also be run in one command:

```powershell
.\run_cli.bat "C:\path\to\document.pdf"
```

For a sensitive local document, use the explicit private mode:

```powershell
.\.runtime\app\Scripts\python.exe .\extract_document.py `
  "C:\private\document.pdf" --private-document
```

The GUI exposes the same choice as **Private document** and selects it by
default. This mode ignores any public output override, creates an opaque
`outputs/private/run_<uuid>` folder, forces the lower-level private-output
guard, hides the source filename and filesystem paths, disables preview,
K-Means display, and visualizations, and offers no downloadable result
archive. The original source path is not passed to the model child process:
the input is copied to an opaque short-lived path and removed after the run.
The session upload cache is also removed after a private run. The private
output root is blocked from Gradio file serving, and the GUI remains
loopback-only outside its container.
Public previews, galleries, and result downloads use a separate per-session
cache below the allowed public output root. Both session caches are removed
when the GUI shuts down.

Or with the lightweight app Python:

```powershell
.\.runtime\app\Scripts\python.exe .\extract_document.py "C:\path\to\image.png"
```

The owner of the original development machine can reuse the already verified
OCR/layout environments without altering them:

```powershell
.\setup_windows.ps1 -Device gpu -ReuseExisting
```

This writes only `runtime.local.json` and the lightweight app environment inside
the package. It does not modify the trained checkpoint or existing OCR/layout
environments.

## macOS quick start

Native macOS Paddle/LayoutXLM environments are not claimed or tested. The
supported macOS path uses Docker Desktop:

1. Install and start Docker Desktop.
2. Extract the package.
3. In Terminal, run:

   ```bash
   cd /path/to/OCR_Model
   bash launch_macos.command
   ```

4. Open <http://127.0.0.1:7860>.

The Compose service is pinned to `linux/amd64` so it can run on Intel Macs and,
through Docker Desktop emulation, Apple Silicon Macs. It is CPU-only and can be
slow. The container and host port are local; the GUI is published only at
`127.0.0.1`.

The exact `linux/amd64` image was built and exercised on Windows Docker
Desktop: all readiness probes passed and the full bundled sample produced the
same field values, OCR text, entity triplets, and relation triplets as the
native GPU run. No physical Intel or Apple Silicon Mac was available, so that
host-specific launch remains explicitly untested.

## Outputs

Every run creates a timestamped folder under `outputs/` containing:

- `document_result.json`: schema-validated full result
- `pages/page_###_ocr.json`: OCR words, lines, confidence, and provenance
- `pages/page_###_entities.json`: learned and generic entities
- `pages/page_###_relations.json`: key/value relations
- `pages/page_###_visualization.png`: visual overlay
- `portable_run.log`: local diagnostic log

The GUI shows a field table, combined OCR text, full JSON, page visualizations,
and a downloadable ZIP of that run. The OCR text and run-log panes have fixed
heights with independent vertical scrolling, so long output remains usable.
Terminal color sequences are removed from the displayed log.

In the GUI, **Maximum PDF pages = 0** means process every page.

Private-document runs are the exception to the public output list above: they
remain below `outputs/private/`, use opaque run IDs, and do not create or show
visualizations or downloadable archives.

## Readiness and troubleshooting

Run the fast check:

```powershell
.\.runtime\app\Scripts\python.exe .\doctor.py
```

Run framework imports too:

```powershell
.\.runtime\app\Scripts\python.exe .\doctor.py --probe
```

`MODEL_MANIFEST.json` records the included model filenames, sizes, and SHA-256
hashes. Every published `OCR_Model.zip` is accompanied by
`OCR_Model.zip.sha256`. On Windows, recompute and compare the digest before
extracting:

```powershell
Get-FileHash .\OCR_Model.zip -Algorithm SHA256
Get-Content .\OCR_Model.zip.sha256
```

Users and judges can download the archive and checksum from the public
[`v1.1.0-ocr-upgrade` Release](https://github.com/PracticalSwan/csx4201-vision-info-extraction/releases/tag/v1.1.0-ocr-upgrade).
The earlier
[`v1.0.0-build-week` Release](https://github.com/PracticalSwan/csx4201-vision-info-extraction/releases/tag/v1.0.0-build-week)
remains the historical July 21 package.

Use the live Release asset and `OCR_Model.zip.sha256` sidecar for the current
archive size and digest. `BUILD_INFO.json` records the clean source commit,
exact Git candidate-tree SHA-256, candidate count, and clean-state flag. The
release build runs only from isolated staging below
`D:\CSX4201\vision-info-extraction-assets`; it rejects `D:\OCR_Model` as a
build target unconditionally. Rebuilding an existing staging target also
requires the exact builder-owned sentinel in that target's parent.

Before publication, the completed payload is scanned for prohibited data
paths, reparse points, secret patterns, and live private-filename inventory
matches without skipping large or binary files. Source copies are verified
before and after copying, reparse points are never followed, and the synthetic
sample must match its executed integration-evidence size and SHA-256. A
`PAYLOAD_MANIFEST.json` records every other package file, size, and SHA-256.
The builder archives only that frozen list, then stream-hashes every ZIP entry
and requires exact path/size/SHA-256 agreement with the embedded manifest
before writing the sidecar. The ZIP also passes CRC, single-root,
duplicate-name, absolute-path, and traversal checks. Generation-specific
executed evidence is `reports/ocr_upgrade/portable_verification.json` in the
source repository; it is authoritative only when its provenance,
`BUILD_INFO.json`, tag, sidecar, and live Release asset all agree.
Machine-local `.runtime`, `runtime.local.json`, and outputs are never part of
the clean ZIP.

The package includes the project's MIT `LICENSE` and `CONTRIBUTING.md`.
LayoutXLM-derived weights and other third-party components retain the upstream
licenses documented in `docs/THIRD_PARTY_NOTICES.md`.

The final layout checkpoint must have this SHA-256:

```text
f257538849bd2067a9df9df83385aa10ae468d0499510fb0621a03a5f0155180
```

The current source checkpoint is:

```text
D:\CSX4201\vision-info-extraction-assets\checkpoints\layoutxlm_multitask\ocr_upgrade_fresh_b_noise
```

The portable registry keeps the selected original detector and general
recognizer, the accepted synthetic-only custom Thai recognizer, and explicit
original/custom/adaptive profile metadata. The original detector and
recognizers remain available as fallbacks. Every result records the resolved
profile and model identities.

The calibrated layout stage is stricter than an OCR-model fallback. Its
checkpoint, calibration, and OCR-stack binding must agree, and a required
layout worker error fails the run. Only the lower-level development CLI can
explicitly opt into generic/rule-only layout fallback; such output is not
calibrated LayoutXLM inference. Because the shipped calibration is bound to the
original OCR stack, the portable one-command CLI offers only
`--ocr-profile original`/`auto` and original/auto component selectors.
Custom/adaptive OCR experiments remain lower-level, explicit degraded-mode
workflows rather than portable calibrated extraction.

The package keeps the original scikit-learn joblib files for provenance, but
the display-only rotation branch loads `models/kmeans_rotation/inference_params.npz`.
That hash-bound numeric export avoids unsupported cross-version pickle loading
and matched the original scaler/PCA/K-Means output on all 7,520 public
train/validation/test feature rows with zero cluster-label differences.

If Windows blocks a downloaded archive, right-click the ZIP, choose
**Properties**, select **Unblock**, then extract it again. If port 7860 is in
use, launch `app.py --port 7861`.

For CPU portability, the adapter disables Paddle's oneDNN/MKLDNN optimization;
the pinned PP-OCRv6 models otherwise trigger a PaddlePaddle 3.3 Linux executor
error. This changes the execution backend, not the bundled weights or output
schema.

## Privacy and limitations

- Do not put private documents in a folder that will be re-zipped or shared.
- Outputs remain local unless the user deliberately approves a bounded Codex
  review payload.
- The K-Means rotation zone is display-only and never controls OCR/extraction.
- The failed exact-angle estimator remains disabled.
- Locked reference-token calibrated entity F1 is 0.9835, while the full
  1,760-page locked image-to-JSON entity F1 is 0.0944 because OCR remains the
  main bottleneck. Relation learning is limited by FUNSD-only supervision.
- This is an academic/noncommercial model package, not a production system.
  Human review is required for financial, legal, or other consequential use.

See `docs/THIRD_PARTY_NOTICES.md` before redistributing the package.
