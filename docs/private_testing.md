# Local private-document testing

Private Gmail documents are an operational test set only. This command runs
the fixed final checkpoint locally; it cannot train, calibrate, select a
checkpoint, or change thresholds.

Run the final aggregate-only evaluator from the OCR environment after training
and calibration:

```powershell
$ocr = 'D:\CSX4201\vision-info-extraction-assets\environments\ie-ocr\Scripts\python.exe'
$checkpoint = 'D:\CSX4201\vision-info-extraction-assets\checkpoints\layoutxlm_multitask\ocr_upgrade_fresh_b_noise'
& $ocr scripts/evaluate_private_gmail.py `
  --layout-checkpoint $checkpoint --device gpu:0 --limit 2
```

This command writes the committable aggregate to
`reports/ocr_upgrade/private_aggregate.json` and keeps its anonymous
document-status evidence under the ignored D: private root. `--limit` counts
source documents, not page rows. It exposes no public filename, path, OCR text,
image, or per-document prediction.

For a larger owner-only manual-review run, use the separate private runner:

```powershell
$checkpoint = 'D:\CSX4201\vision-info-extraction-assets\checkpoints\layoutxlm_multitask\ocr_upgrade_fresh_b_noise'
& $ocr scripts/run_private_test.py `
  --input-root 'data\raw\private\gmail' `
  --output-root 'D:\CSX4201\vision-info-extraction-assets\private-evaluation\ocr-upgrade' `
  --language auto --device gpu:0 --private-output `
  --checkpoint $checkpoint --recursive --continue-on-error `
  --no-private-visualizations --aggregate-report `
  --manual-review-csv --force
```

Use `--limit N` for a bounded run, or repeat `--file <relative-path>` to select
specific files under `--input-root`. A selected file cannot escape that root.
`--max-pages N` bounds multipage documents. The command requires the output
root to remain below the configured ignored private root on D:.
The general `extract_document.py` command enforces the same input boundary: a
path under any configured Gmail root cannot run without `--private-output`.

Detailed owner-only output uses anonymous IDs:

```text
private-evaluation/ocr-upgrade/
  aggregate_report.json
  aggregate_report.md
  manual_review.csv
  documents/
    private_000001/
      document_result.json
      pages/
```

`manual_review.csv` is private because predicted values and evidence page
numbers are included for correction. Keep it on D: and never copy it into a
tracked report directory. The optional public aggregate report contains only
counts, timings, route/document-type totals, the checkpoint hash, and explicit
no-content declarations. It contains no source filename, OCR text, image, or
per-document prediction.

A nonzero exit means at least one selected document failed or no document
completed. With `--continue-on-error`, anonymous error records remain local so
the other documents can finish. Never use the aggregate result to tune the
model; there is no private ground truth, so it is not an accuracy estimate.

The OCR-upgrade operational check ran only after every model, calibration, and
threshold choice was fixed. It was intentionally bounded to two anonymous
documents and two pages. Both completed with nonempty output and zero failures
in 126.586 wall seconds. The public aggregate records the selected detector,
recognizer, checkpoint, calibration, zero Gmail fit rows, and explicit
no-content declarations. These counts are operational only and were not used
to change the model.

The older July 17 lifecycle completed 26 documents and 203 pages with the
historical checkpoint. That remains historical evidence and must not be
reported as the current OCR-upgrade private run.

## Portable owner-only private mode

The portable package has a separate convenience boundary for individual
sensitive documents:

```powershell
.\.runtime\app\Scripts\python.exe .\extract_document.py `
  "C:\private\document.pdf" --private-document
```

The GUI exposes the same option as **Private processing**. It creates an opaque
`outputs/private/run_<uuid>` directory regardless of a public output
override, forces `--private-output`, disables K-Means display and
visualizations, redacts source filenames and paths from command/log/error
surfaces, and does not create a downloadable archive or gallery. Gradio is
also configured to block the private root. The browser-local file selector
still displays the selected filename; the upload remains in the single session
cache for repeat runs and is removed when the GUI shuts down. This mode is a
local containment control; it does not authorize publishing private output or
using it for training, calibration, selection, or accuracy claims.
