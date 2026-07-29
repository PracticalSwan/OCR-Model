# Summary — Domain-Adapted OCR and Information-Extraction Pre-Model

**Project:** CSX4201 vision-info-extraction
**Verified through:** 2026-07-30

## Outcome

The workspace contains a public-trained academic pre-model with original,
custom, and adaptive OCR experiment profiles. Only the original OCR stack is
bound to the shipped LayoutXLM calibration; custom/adaptive runs require the
lower-level CLI's explicit generic-layout fallback and are not calibrated
LayoutXLM extraction. A deterministic
400-page `DEV_SELECT` benchmark replaced the prior three-page selection
evidence. Public-only detector/recognizer trials were executed, OCR-realistic
LayoutXLM streams were rebuilt, a fresh checkpoint was selected and
recalibrated, and the frozen stack completed one locked 1,760-page
image-to-JSON evaluation.

The selected global OCR profile remains `original`: no custom detector
produced an eligible checkpoint on the verified laptop GPU, and the custom
general recognizer failed the WER and Turkish non-regression gates. The custom
Thai recognizer is available only in explicit custom/adaptive profiles and has
synthetic-only selection evidence.

The preserved K-Means quadrant experiment remains an auxiliary display branch.
It never controls OCR or extraction. Its weak mapped accuracy and failed
exact-angle estimator are reported, not hidden.

## Final evidence snapshot

| Surface | Result |
|---|---|
| Raw integrity | 128,793 files; 35,459,126,772 bytes; public/private separation retained |
| Normalized public population | 12,433 pages; zero Gmail fit rows; zero leakage across 29,886 identities |
| OCR benchmark | 400 public DEV_SELECT pages: 283 FATURA, 20 FUNSD, 97 SROIE |
| OCR training data | 8,755 detector pages/163,679 regions; 137,886 real recognition crops; 48,471 synthetic crops |
| OCR selection | Original detector and general recognizer; custom Thai only for explicit custom/adaptive use |
| Adapted model data | 16,781 examples; 12,455 train; 1,913 dev-select; 653 calibration; 1,760 locked test |
| Adapted training | Four epochs; 12,556 optimizer steps; epoch 4 score 0.843343; reload max difference 0 |
| Locked layout test | 1,760 examples; calibrated entity/canonical/relation F1 0.9835/0.9860/0.5603 |
| Locked image-to-JSON test | 1,760 pages; polygon F1 0.3815; coverage 0.1663; WER 0.9692; entity/relation F1 0.0944/0.0111 |
| Layout rotation | 540/540; minimum calibrated entity/canonical/relation F1 0.7683/0.9640/0.3358 |
| End-to-end rotation | 72/72 nonempty; public coverage 0.3068–0.3839; entity F1 0.1326–0.1807; synthetic Thai 18/18 |
| Unseen CORU | 100/100 pages; 78.53% QA-answer text recall; 15.68% canonical exact match; 25.96 seconds/page |
| Private operation | 2/2 anonymous documents and pages; aggregate only; no filename/text/image/per-document output |
| Integration | Strict original calibrated image/PDF paths plus explicit degraded custom/adaptive OCR experiments are schema-valid; historical custom/adaptive outputs did not prove calibrated LayoutXLM |

## Portable product and publication

The model is available through a one-command CLI and repaired local GUI.
For public mode, the GUI previews images and first-page PDF renders. Private
mode is selected by default and disables preview. The GUI uses one progress
surface and keeps long OCR and run-log output independently scrollable.
Extraction is local and requires no OpenAI API key.

The public `v1.0.0-build-week` Release remains the historical July 21
privacy-audited package for Windows and a Docker-backed macOS route. The
1,152,835,265-byte ZIP has SHA-256
`c6c874f5b0879478497c9a33529f6416d48be60d586197fb625540d795f9ec6b`
and was built from clean commit
`e47023de2a201092df6fd3393ec297b2835e0a50`. ZIP integrity, model manifests,
portable doctor probes, and a full CPU sample extraction pass. Windows native
GPU and Docker Linux/AMD64 CPU output parity was verified earlier; physical
Apple hardware remains untested.

OpenAI Build Week submission `1102544` is `Submitted` in the Work &
Productivity track. It uses the public 2:54 demo, public repository, Thailand
as the owner-confirmed country, and `/feedback` Session ID
`019f7669-11fd-7923-ad68-ea1a09bd7d74`.

Post-publication cleanup removed 2,443,608,061 bytes of verified generated
caches, logs, screenshots, local test outputs, obsolete package-only
submission docs, the empty legacy directory tree, and staging copies. Raw
data, final model assets, the local runtime, runtime configuration, canonical
ZIP, and demo MP4 were preserved.

The OCR upgrade merged into `main` through PR #2 at
`c6303f6843de9af1c7c97fde1ef6ff43e01de553`. The current portable archive is
published under `v1.1.0-ocr-upgrade`; its live Release asset and
`OCR_Model.zip.sha256` sidecar are the authoritative current size and digest.
Each corrected build is produced from isolated staging, refuses a dirty Git
candidate tree, records the exact source commit/tree hash in `BUILD_INFO.json`,
refuses every existing target and all installed-working-copy targets, rejects
source reparse points, binds the safe sample to integration evidence, scans
every payload byte, emits a complete SHA-256 payload manifest, and validates
ZIP integrity before publication.
The historical `v1.0.0-build-week` Release remains available.

Calibrated layout inference now validates the calibration's OCR-stack binding
before starting its worker and fails closed on required worker errors. An
explicit developer-only generic/rule fallback remains available but is never
reported as calibrated LayoutXLM output. The portable CLI and GUI also expose
an explicit private-document mode with opaque run IDs, filename/path
redaction, no private preview, opaque short-lived worker input, session-cache
cleanup, private-root isolation, and no visualization or downloadable archive.

## Current model and runtime

The final checkpoint uses `microsoft/layoutxlm-base` multilingual text and
normalized 2D-layout embeddings with entity, document, canonical-evidence, and
real relation heads. It omits the Detectron2 visual backbone. Dynamic training
uses 60% upright and 40% arbitrary-angle examples; checkpoint selection also
includes a fixed 37° slice.

Checkpoint:

```text
D:\CSX4201\vision-info-extraction-assets\checkpoints\layoutxlm_multitask\ocr_upgrade_fresh_b_noise
```

Model SHA-256:

```text
f257538849bd2067a9df9df83385aa10ae468d0499510fb0621a03a5f0155180
```

Paddle GPU and CUDA PyTorch run in separate Python 3.10 processes to avoid a
Windows cuDNN DLL collision. All large environments, caches, datasets,
checkpoints, generated outputs, and private operational results remain on D:.
The source and derived checkpoint license is CC-BY-NC-SA-4.0.

## What the scores mean

The layout heads are strong when evaluated on reference tokens and boxes, but
the real OCR pipeline substantially limits end-to-end extraction. The full
locked image-to-JSON run measured only 0.1663 recognized-text coverage and
0.9692 WER. Sparse FUNSD-only relation labels further constrain learned
relations. These are measured model limitations, not verifier failures.

CORU contributes no fit or selection row. Its 100-page result measures whether
known answer strings appear in OCR and whether canonical values match exactly;
it does not invent token-level entity/relation ground truth. Private Gmail
results prove local operation only and are never accuracy evidence.

## Complete versus open

Complete:

- public normalization, leakage-safe splits, 400-page OCR benchmark, detector
  data, recognition crops, and licensed synthetic data;
- bounded detector/general/Thai trials with explicit acceptance decisions;
- rebuilt ground-truth, PaddleOCR, hybrid, and train-only noise streams;
- fresh four-epoch public-only training, reload check, and calibration;
- exact original/custom OCR verification and automatic arbitrary-angle deskew;
- image, PDF, multipage, rotated, unknown-type, and Thai inference;
- locked in-domain, 18-angle layout, 18-angle end-to-end, and 100-page unseen
  evaluation;
- schema validation, private path/cache gates, aggregate-only private testing;
- 14 model-supervised canonical-evidence fields and 27 schema-supported output
  fields with their different scopes stated explicitly;
- fail-closed calibration/OCR-stack binding plus explicit generic-only
  fallback, and an opaque portable private-document workflow;
- cryptographically bound integration evidence and final report bundle;
- preserved, failure-isolated K-Means display baseline.

Still open research/product decisions:

- professor-approved canonical fields, document types, and official quality
  thresholds;
- a compatible labeled public Thai benchmark;
- broader OCR/domain adaptation and stronger relation supervision;
- an approved orientation/zone method if the professor requires more than the
  preserved K-Means diagnostic plus independent OCR correction;
- any future commercial redistribution path, because the inherited
  LayoutXLM-derived checkpoint is CC BY-NC-SA 4.0.

The result is a working academic pre-model with measured limitations. The
locked end-to-end quality targets were not reached, so it is not a claim of
production readiness or accurate operation on every document. Full upgrade
evidence and commands are in
[`docs/OCR_UPGRADE_RELEASE_NOTES.md`](docs/OCR_UPGRADE_RELEASE_NOTES.md).
