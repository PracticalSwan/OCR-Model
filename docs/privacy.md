# Privacy and Publication Rules

`data/raw/private/gmail/` contains real personal financial and legal material.
It is local private-test data, not training data or publication material.

## Prohibited

- committing or uploading private source files, rendered pages, OCR caches,
  text, filenames, paths, identifiers, previews, or per-document predictions;
- using Gmail to fit models, routing thresholds, rules, checkpoint selection,
  hyperparameters, or examples;
- sending private files to an external service;
- moving a private output into a public report root.

## Allowed local operation

Private inference may run locally with `--private-output`. Detailed results
must remain below the ignored D: private root. A committed report may contain
aggregate counts and timings only and must explicitly declare that it contains
no filenames, OCR text, images, or per-document predictions.

The current OCR-upgrade aggregate is
`reports/ocr_upgrade/private_aggregate.json`. It covers two anonymous
documents/pages, records zero failures and zero Gmail fit rows, and contains no
filename, path, OCR text, image, or per-document prediction. Detailed local
status remains under the configured ignored D: private root.

The general extraction CLI resolves every input against the four configured
Gmail roots before opening it. A matching input is rejected unless
`--private-output` is present; that mode then requires the destination to stay
below the ignored D: private root. The caller cannot opt out of either guard.

The portable CLI/GUI adds an explicit `--private-document` / **Private
processing** mode for owner-only local use. It forces an opaque
`outputs/private/run_<uuid>` destination, hides source filenames and paths,
redacts command/log/error surfaces, disables preview, visualizations, and
downloadable archives, and blocks the private output root from file serving.
Gradio's own per-session upload cache remains available for required input
preprocessing on the loopback-only GUI. Before the child process starts, the
source is copied to an opaque short-lived input path so the original path is
not present in child-process arguments. That worker copy is removed after the
run. The local file selector still shows the selected filename; result
surfaces do not. Private mode is selected by default in the GUI. Public and
private selections share one Gradio session cache so repeat runs and settings
changes remain usable; the session cache is removed on GUI shutdown.
These containment controls do not make the result publishable.

## Before staging or pushing

1. Verify repository visibility and remote ownership.
2. Run both data and information-extraction verifiers.
3. Inspect `git status --ignored` and every staged path.
4. Inspect the staged diff for private filename/text fragments, secrets,
   `.env` content, credentials, source paths, and generated previews.
5. Reject unexpectedly large files and model/checkpoint/cache artifacts.
6. Confirm the training and checkpoint reports still show Gmail fit rows 0.
7. Confirm all 30 entries in
   `reports/ocr_upgrade/verification_executions.json` are current and that the
   staged-file and portable-package privacy checks point to executed evidence.
8. Build only from a clean Git candidate tree in marked, isolated D: staging.
   Never target the installed `D:\OCR_Model` working copy. Copy without
   following reparse points, bind the synthetic sample to executed evidence,
   scan every completed-payload byte for prohibited paths, secrets, and every
   live private-inventory filename, then write the full payload manifest.
9. Validate ZIP CRC, single-root layout, duplicate names, absolute/traversal
   paths, sidecar, `BUILD_INFO.json`, tag target, and live Release asset as one
   generation.

Ignore rules are a safeguard, not authorization to publish. A successful scan
means no known match was found in the checked surface; it does not prove that
an image/PDF is safe without human review.
