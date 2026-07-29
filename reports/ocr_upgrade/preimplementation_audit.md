# OCR Upgrade Preimplementation Audit

**Audit timestamp:** 2026-07-24T06:24:40Z

**Repository:** `https://github.com/PracticalSwan/csx4201-vision-info-extraction.git`

**Visibility:** public, verified live with GitHub CLI

**Baseline commit:** `af0816b83fde1b3fb8a25802a1967350de654812`

**Recovery tag:** `pre-ocr-upgrade-baseline` (resolves to the baseline commit)

**Implementation branch:** `feat/domain-adapted-ocr`

## Decision

The OCR upgrade can proceed. The current extraction lifecycle, exact original
OCR models, final LayoutXLM checkpoint, calibration, integration paths,
rotation branch, public/private split discipline, and portable archive are
present and verifiable.

The current public OCR accuracy evidence remains too small for model selection:
the upright OCR report contains only three pages. Its recognized-text coverage
is `0.4027889972`, WER is `0.6998877146`, and polygon F1 is
`0.3395106951`. On those same representative pages, upright end-to-end entity
F1 is `0.1772004268`, relation F1 is `0.0074906367`, and canonical-field
accuracy is `0.4444444444`. This supports the project’s diagnosis that OCR,
especially missed text regions, remains the main end-to-end bottleneck. It is
not sufficient evidence for choosing a new detector, recognizer, preprocessing
profile, or adaptive inference rule.

No additional dataset is required before work begins. The existing public
DEV_SELECT population already contains 20 FUNSD, 97 SROIE, and 996 FATURA
pages. After group-aware duplicate handling, it can satisfy the required
400-page benchmark. A Thai dataset decision still requires a current
authoritative license review; unclear-license data will not be downloaded.

## Baseline verification executed

| Gate | Result |
|---|---|
| Host tests | `244 passed, 2 skipped` in 32.11 seconds |
| Python compilation | `python -m compileall -q src scripts tests` exited 0 |
| OCR environment partition | `123 passed` in 2.86 seconds |
| CUDA layout partition | `2 passed` in 34.97 seconds |
| Raw/data verifier before repair | `19/20`; only Codex internal `.git/refs/codex/turn-diffs/checkpoints` was misclassified |
| Verifier regression repair | New focused test failed with the observed path, then passed after pruning `.git` metadata |
| Raw/data verifier after repair | `20/20`; 128,767 inventory paths resolved, 100 move hashes stable, 3,302 duplicate groups valid |
| Host tests after verifier repair | `245 passed, 2 skipped` in 15.81 seconds |
| Rotation verifier | `20/20` |
| Annotation verifier | Passed; 12,433 normalized public annotations, zero validation errors, Gmail fit rows zero |
| Exact OCR model verifier | Passed on `gpu:0`; general, Thai, rotated, and arbitrary phrase checks passed |
| Complete IE verifier | `46/46` |
| Fresh integration smoke | Passed on `gpu:0`; image, arbitrary rotation, Thai routing, and mixed-language multipage PDF |

The data-verifier defect was in repository traversal, not in project data. The
check scanned Git metadata and interpreted a Codex-internal directory named
`checkpoints` as a committable model output. The minimal fix ignores `.git`
metadata while the existing regression continues to reject real workspace
`checkpoints/` and OCR-output directories.

## Frozen artifacts

| Artifact | SHA-256 |
|---|---|
| Original detector `PP-OCRv6_medium_det` | `393f629d341e6388ca72d19b25983b96cae36dfdf1f7146adf42d6ab68789388` |
| Original general recognizer `PP-OCRv6_medium_rec` | `666f7c4d6d5c846c7e202f63356b3fc4a7a3d4ff7040e763b93c5771608c7ae0` |
| Original Thai recognizer `th_PP-OCRv5_mobile_rec` | `507c659e3abb6c2c7262a07965ec880119751a29e3c8ed7437b337f74b36608d` |
| Final LayoutXLM weights | `34c7a26e78d6285a2739e1b61839eadfd0e686ccbcf57f9cb47997c12cef2189` |
| Final LayoutXLM config | `68046da6e64032264df40c8110b3b6eeb06fb769f761ad8e02c2bc2627841766` |
| Calibration | `b1c683d3f1e1cfc4b9515bb83cf64ce455343445a553ab7525f845272b902087` |
| Project config | `b03d0ae5c8ef56ccf01051f9b0e8dd8a59fd3391235db9b9de16126d61312acd` |
| Existing portable ZIP | `c6c874f5b0879478497c9a33529f6416d48be60d586197fb625540d795f9ec6b` |

The original OCR artifact directories will remain unchanged and selectable.
New training, exported models, generated datasets, caches, logs, and temporary
files will use versioned paths under
`D:\CSX4201\vision-info-extraction-assets`.

## Runtime and storage

- C: free space at the freeze point: `41.083 GiB`.
- D: free space at the freeze point: `367.210 GiB`.
- Required C: reserve: `15 GiB`.
- Required D: planning margin: estimated requirement plus `35%`.
- GPU: NVIDIA GeForce RTX 5050 Laptop GPU, `8,151 MiB` total memory.
- Host: Python `3.14.2`.
- OCR inference environment: Python `3.10.11`, PaddlePaddle GPU `3.3.0`,
  PaddleOCR `3.7.0`, PaddleX `3.7.2`.
- Layout environment: Python `3.10.11`, PyTorch `2.8.0+cu128`,
  Transformers `4.57.6`.

A direct, unconfigured `import paddleocr` in the OCR environment can traverse
PaddleX/ModelScope into PyTorch and fail to load `torch\lib\shm.dll` with
Windows error 127. The project’s configured exact-model verifier and full
inference path pass. The separate training environment must therefore prove
its precise import order and one-step train/save/resume/evaluate/export
lifecycle before any counted trial. Paddle GPU and CUDA PyTorch will continue
to run in process-isolated environments.

## Data and leakage baseline

- Normalized public split manifest: 12,433 rows.
- TRAIN: 7,646 rows.
- DEV_SELECT: 1,113 rows.
- DEV_CALIBRATION: 653 rows.
- TEST_IN_DOMAIN: 1,760 rows.
- CORU unseen: 1,261 rows.
- DEV_SELECT source counts: FATURA 996, SROIE 97, FUNSD 20.
- Four exact-SHA duplicate groups occur within DEV_SELECT and must be kept
  group-aware during benchmark selection.
- Current final model dataset: 11,684 examples, including 7,782 TRAIN,
  1,243 DEV_SELECT, 763 DEV_CALIBRATION, and 1,896 TEST_IN_DOMAIN examples.
- Current streams: 11,172 ground-truth, 256 PaddleOCR, and 256 hybrid examples.
- Gmail/private fit rows: zero in dataset, training, calibration, and reports.

Every new raw or generated item must inherit a stable document-family lineage.
TRAIN is the only weight-fitting role; DEV_SELECT is the only selection role;
DEV_CALIBRATION is calibration-only; TEST_IN_DOMAIN and CORU remain locked
until the final candidate; Gmail remains aggregate-only private operation.

## Current final-model reference

The current four-epoch public LayoutXLM run used build
`final-6be3e0b46b0a4e4c`, completed in `3198.296` seconds, and selected epoch 4.
The locked reference-token TEST_IN_DOMAIN report records:

- calibrated entity F1: `0.9812706023`;
- calibrated canonical-evidence F1: `0.9814241916`;
- calibrated relation F1: `0.4632400781`;
- document accuracy and macro-F1: `1.0`.

These reference-token metrics are protected regression gates. They do not
describe deployable end-to-end OCR quality.

The previous 18-angle end-to-end grid used three public pages and completed in
`335.749` seconds. The previous fixed unseen CORU run processed 100/100 pages
with nonempty rate `1.0`, answer-string recall `0.7853036741`, canonical exact
match `0.1567877629`, and mean processing time `31.0417` seconds per page. The
previous private operation processed 26/26 anonymous documents and 203 pages,
with zero Gmail fit rows and no public filenames, OCR text, images, or row-level
predictions.

## Implementation gates

1. Freeze a group-aware DEV_SELECT benchmark and verify geometry/transcripts
   before inference selection or training.
2. Improve detection and reference-text coverage before spending recognizer
   budget when ground-truth crop recognition is already adequate.
3. Use at most four detector and four general recognizer trials; record
   meaningful failed or aborted runs.
4. Keep original OCR behavior independently runnable.
5. Keep K-Means display-only and absent from every OCR decision/evaluation.
6. Lock OCR hashes/configuration before rebuilding deployable LayoutXLM streams.
7. Select downstream adaptation on DEV_SELECT, then recalibrate only on
   DEV_CALIBRATION.
8. Run prediction-affecting review before the first final TEST execution.
9. Use the single permitted specialized review agent only after implementation,
   evaluation, documentation, and passing tests.
10. Treat any post-TEST prediction change as invalidating the tested artifact
    and rerun transparently.

The detailed task-by-task contract is
`docs/superpowers/plans/2026-07-24-domain-adapted-ocr.md`.
