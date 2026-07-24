# Domain-Adapted OCR Upgrade Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task by task. Keep prediction-affecting review before the locked evaluation and use the one permitted specialized review agent only after implementation, evaluation, documentation, and tests are complete.

**Goal:** Reduce OCR-driven end-to-end extraction loss with a public-only, leakage-safe detector/recognizer upgrade while preserving the original OCR fallback, LayoutXLM contract, private-data boundary, and display-only K-Means behavior.

**Architecture:** Add a reproducible DEV_SELECT OCR benchmark and public TRAIN conversion layer, use official pinned PaddleOCR training artifacts in a separate D:-backed environment, select bounded detector and recognizer trials only on DEV_SELECT, then integrate the accepted model stack through versioned registry entries and an expanded cache contract. Adaptive rendering, tiling, crop rectification, recognition retries, and orientation scoring remain bounded inference components. Only after the OCR stack is locked are deployable OCR/hybrid LayoutXLM streams rebuilt, downstream adaptation selected on DEV_SELECT, confidence recalibrated on DEV_CALIBRATION, and TEST_IN_DOMAIN/CORU/private operation executed.

**Technology:** Python 3.10, PaddlePaddle/PaddleOCR/PaddleX, OpenCV, Pillow, PyMuPDF, PyTorch/Transformers/LayoutXLM, pytest, PowerShell, JSON/CSV/SHA-256 provenance, Windows CUDA and Docker Linux/AMD64 verification.

**Evaluation roles:** `TRAIN` fits weights; `DEV_SELECT` selects every model, preprocessing, rendering, tiling, crop, retry, routing, stream, and adaptation choice; `DEV_CALIBRATION` fits only calibration and abstention artifacts; `TEST_IN_DOMAIN` is run only after all prediction-affecting choices are frozen; CORU is unseen final evaluation only; Gmail is aggregate-only private operation after public evaluation.

---

## Task 1: Freeze the baseline and repair the verifier false positive

**Files:**

- Create: `reports/ocr_upgrade/preimplementation_audit.md`
- Create: `reports/ocr_upgrade/baseline_snapshot.json`
- Modify: `scripts/verify_data.py` or its owning verification module
- Modify: `tests/test_verify_data_stage.py`
- Regenerate: `data/metadata/organization_report.md`
- Regenerate: `data/metadata/verification_result.json`
- Regenerate: `reports/ocr/model_verification.json`
- Regenerate: `reports/verification/rotation_verification.json`
- Regenerate: `reports/verification/information_extraction_verification.json`
- Regenerate: `reports/information_extraction/integration_smoke.json`

**Steps:**

1. Write a failing regression test proving repository-internal `.git/refs/codex/turn-diffs/checkpoints` paths are ignored while genuine unignored checkpoint/output directories remain rejected.
2. Run only `tests/test_verify_data_stage.py` and confirm the new test fails for the observed reason.
3. Apply the smallest ownership-level fix and rerun the targeted test.
4. Record the exact baseline commit/tag, environment versions, storage, GPU, original OCR aggregate hashes, LayoutXLM/config/calibration/package hashes, commands, durations, and current metrics.
5. Regenerate the data verifier and require 20/20 without weakening its check of real workspace outputs.
6. Rerun the host suite, compilation, OCR and layout environment partitions, exact-model verifier, rotation verifier, annotation verifier, full IE verifier, and integration smoke.
7. Commit this recovery/audit stage before dataset generation.

## Task 2: Make split roles, lineage, and metric-report contracts executable

**Files:**

- Create: `src/ocr/lineage.py`
- Create: `src/evaluation/ocr_benchmark.py`
- Create: `tests/test_ocr_lineage.py`
- Create: `tests/test_ocr_benchmark_metrics.py`
- Modify: `src/information_extraction/manifest.py`
- Modify: `src/ocr/cache.py`
- Modify: `config.yaml`

**Steps:**

1. Add failing tests for allowed split roles, inherited `document_family_id`, parent/transform lineage, mixed-build rejection, private-row rejection, and required report provenance.
2. Implement immutable lineage records containing sample/document/family IDs, split role, source dataset/hash, parent sample, transformation chain, license ID, and annotation version.
3. Add a common metric-report envelope containing build ID, split, manifest/model/checkpoint/calibration/config hashes, source commit, device, counts, failures, duration, and private-row count.
4. Expand `OCRCacheKey` to cover rendered-page settings, crop padding, tiling, preprocessing/orientation configuration, language route, package version, and a code/configuration hash.
5. Prove old cache keys miss and unchanged complete keys hit deterministically.

## Task 3: Build and verify the 400+ page DEV_SELECT OCR benchmark

**Files:**

- Create: `src/ocr/benchmark.py`
- Create: `scripts/build_ocr_benchmark.py`
- Create: `scripts/verify_ocr_benchmark.py`
- Create: `tests/test_ocr_benchmark.py`
- Create: `data/metadata/ocr_benchmark_manifest.csv`
- Create: `reports/ocr_upgrade/benchmark_summary.json`
- Create: `reports/ocr_upgrade/benchmark_selection.md`

**Steps:**

1. Add failing converter/selection tests using compact FUNSD, SROIE, and FATURA fixtures.
2. Implement deterministic group-aware sampling of every FUNSD and SROIE DEV_SELECT page plus at least 283 FATURA DEV_SELECT pages, deduplicated by family/hash and stratified by the required density, size, quality, table, field, and diagnostic-rotation attributes.
3. Emit only public rows and keep all generated artifacts outside raw directories.
4. Add geometry, transcript normalization, SHA-256, duplicate/leakage, split-role, and transform round-trip verification.
5. Build the live manifest; require at least 400 pages and evaluate whether 500–800 is justified by measured stratum coverage and runtime.
6. Freeze the benchmark manifest hash before inference ablations or training.

## Task 4: Build public detector and recognition datasets

**Files:**

- Create: `src/ocr/training_data.py`
- Create: `src/ocr/crops.py`
- Create: `scripts/build_ocr_detection_dataset.py`
- Create: `scripts/build_ocr_recognition_dataset.py`
- Create: `tests/test_ocr_detection_dataset.py`
- Create: `tests/test_ocr_recognition_dataset.py`

**Steps:**

1. Write failing FATURA, SROIE, and FUNSD conversion tests for polygon preservation, box-to-polygon conversion, bounds/area/self-intersection validation, source image dimensions, and source provenance.
2. Implement detector outputs under `D:\CSX4201\vision-info-extraction-assets\data\detector_training` with TRAIN images/labels, DEV_SELECT validation images/labels, manifest, errors, statistics, and checksums.
3. Enforce no DEV_CALIBRATION, TEST_IN_DOMAIN, CORU, Gmail, mixed build, private row, or cross-split duplicate.
4. Write failing line-grouping and crop tests for baseline/vertical overlap, reading order, perspective rectification, proportional padding, aspect ratio, critical word crops, UTF-8 lists, and invalid-crop rejection.
5. Implement recognition outputs under `D:\CSX4201\vision-info-extraction-assets\data\recognition_training`.
6. Verify the general dictionary and required English, numeric, financial, currency, email, punctuation, and Turkish character inventory without silently changing pretrained indices.
7. Record exact page/crop counts, exclusions, manifests, checksums, and storage.

## Task 5: Generate bounded synthetic financial and Thai data

**Files:**

- Create: `src/ocr/synthetic_data.py`
- Create: `scripts/generate_synthetic_recognition_data.py`
- Create: `tests/test_ocr_synthetic_data.py`
- Create: `reports/ocr_upgrade/synthetic_data_manifest.json`
- Create: `docs/ocr_synthetic_data.md`
- Create or update: `reports/ocr_upgrade/thai_trials.csv`
- Modify: `docs/THIRD_PARTY_NOTICES.md`

**Steps:**

1. Use current authoritative sources to decide whether a suitable Thai document/text-line dataset has explicit compatible licensing and redistribution terms.
2. Do not download or train on unclear-license data; otherwise record source, license, permitted use, redistribution, and derived-weight terms.
3. Add deterministic synthetic-corpus tests for seed stability, TRAIN/DEV_SELECT separation, font inventory/license provenance, UTF-8, Thai/English/financial coverage, readable bounds, and private-source exclusion.
4. Generate financial/identity-like lines with bounded blur, compression, perspective, brightness, contrast, noise, rotation, texture, spacing, and size variation.
5. Keep synthetic general crops between 15% and 30% of the recognition mix.
6. Build separate Thai TRAIN and DEV_SELECT-equivalent synthetic/public splits covering Thai-English text, dates, amounts, currency, phone, tax IDs, and supported numeral forms.

## Task 6: Create and prove the D:-backed OCR training environment

**Files:**

- Create: `scripts/setup_ocr_training_environment.ps1`
- Create: `scripts/verify_ocr_training_environment.py`
- Create: `scripts/download_ocr_training_checkpoints.py`
- Create: `tests/test_ocr_training_artifacts.py`
- Create: `reports/ocr_upgrade/training_environment.json`
- Create: `reports/ocr_upgrade/pretrained_training_models.json`
- Create: `configs/ocr_upgrade/README.md`

**Steps:**

1. Verify current official PaddleOCR/PaddlePaddle training documentation and repository sources; record exact URLs, tag/commit, Python/CUDA compatibility, and PP-OCRv6/Thai configuration identities.
2. Clone the official repository only under `D:\CSX4201\vision-info-extraction-assets\vendor\PaddleOCR` and pin the selected compatible commit.
3. Create `D:\CSX4201\vision-info-extraction-assets\environments\ie-ocr-train` without modifying the existing inference environments.
4. Route Paddle, Hugging Face, pip, Torch, temporary, dataset, checkpoint, and log paths to D: and enforce the 15 GiB C: reserve plus 35% D: safety margin.
5. Download only official trainable PP-OCRv6 medium detector/recognizer and Thai PP-OCRv5 mobile checkpoints with bounded retries, partial-download detection, type/config compatibility checks, and SHA-256.
6. Prove imports, GPU visibility, one forward/backward/optimizer step, checkpoint save/reload/resume, evaluation, and inference export before counting a trial.

## Task 7: Run bounded detector trials and lock the detector decision

**Files:**

- Create: `configs/ocr_upgrade/detector/*.yml`
- Create: `scripts/train_ocr_detector.ps1`
- Create: `scripts/evaluate_ocr_detector.py`
- Create: `tests/test_ocr_detector_evaluation.py`
- Create: `reports/ocr_upgrade/detector_trials.csv`
- Create: `reports/ocr_upgrade/error_analysis.md`

**Steps:**

1. Freeze a maximum-four trial ledger before launching each meaningful compute run.
2. Evaluate the original detector on the benchmark for polygon precision/recall/F1, size recall, critical-region recall, failures, latency, and GPU memory.
3. Run the official fine-tuning configuration, then only evidence-justified document, small-text, or tiling-aware variants.
4. Select on DEV_SELECT only and stop when further trials are unlikely to change the decision.
5. Accept a custom detector only if it reaches F1 ≥ 0.60 or improves by ≥0.10 absolute, preserves critical recall, reloads after export, keeps contract compatibility, and passes rotated/blank/low-text tests.
6. Preserve every config, log, model hash, prediction manifest, metric, failure, and rejection reason.

## Task 8: Run bounded general and Thai recognizer trials

**Files:**

- Create: `configs/ocr_upgrade/recognizer/*.yml`
- Create: `scripts/train_ocr_recognizer.ps1`
- Create: `scripts/evaluate_ocr_recognizer.py`
- Create: `tests/test_ocr_recognizer_evaluation.py`
- Create: `reports/ocr_upgrade/recognizer_trials.csv`
- Update: `reports/ocr_upgrade/thai_trials.csv`

**Steps:**

1. Evaluate originals on ground-truth DEV_SELECT crops before training to separate recognition error from detection/crop error.
2. Run at most four general/Thai recognition trials in total per the task’s bounded ledgers, using low learning rates and only public TRAIN plus bounded synthetic data.
3. Measure CER, WER, line/numeric/amount/date/identifier/currency/email exact match, Turkish coverage, Thai synthetic/public holdout, confidence, latency, failures, and export/reload.
4. Accept the general custom recognizer only if WER improves by ≥15% relative or reaches ≤0.45, critical exact match improves ≥10 points, and protected numeric/English/Turkish behavior does not materially regress.
5. Accept custom Thai only if it beats the original on the licensed public/synthetic development benchmark; otherwise keep the original and preserve the failed experiment.

## Task 9: Implement and select bounded inference improvements

**Files:**

- Create: `src/ocr/adaptive_rendering.py`
- Create: `src/ocr/tiling.py`
- Create: `src/ocr/recognition_retry.py`
- Create: `tests/test_ocr_adaptive_rendering.py`
- Create: `tests/test_ocr_tiling.py`
- Create: `tests/test_ocr_crops.py`
- Create: `tests/test_ocr_recognition_retry.py`
- Modify: `src/inference/document_io.py`
- Modify: `src/ocr/preprocessing.py`
- Modify: `src/ocr/fine_deskew.py`
- Modify: `src/ocr/scoring.py`
- Modify: `src/ocr/pipeline.py`
- Create: `scripts/run_large_ocr_ablation.py`
- Create: `reports/ocr_upgrade/preprocessing_ablation.csv`
- Create: `reports/ocr_upgrade/preprocessing_selection.json`
- Create: `reports/ocr_upgrade/adaptive_rendering_metrics.json`
- Create: `reports/ocr_upgrade/tiling_metrics.json`
- Create: `reports/ocr_upgrade/crop_padding_metrics.json`
- Create: `reports/ocr_upgrade/orientation_metrics.json`

**Steps:**

1. Add failing tests for 200-DPI first pass, fixed weak-page signals, bounded 300-DPI rerender, same-score comparison, page numbering, and no global DPI change.
2. Add failing tests for 2×2/3×3 overlap, tile/source transform round trips, clipped-edge handling, polygon duplicate suppression, confidence preservation, reading order, maximum tile count, and weak-page-only activation.
3. Add failing tests for perspective crop rectification, padding profiles A/B/C, small-character upscaling, background fill, invalid crops, one-step larger-padding retry, and a maximum of five candidates.
4. Add failing tests for confidence/valid-character/script/agreement reranking that never invents text and records every candidate/evidence/selection reason.
5. Improve robust line-angle estimation and orientation scoring with fragmented/duplicate/empty penalties while keeping K-Means absent from all decisions.
6. Run benchmark-scale original/grayscale/CLAHE/background-normalization/sharpen/denoise/adaptive ablations and retain only gains outside bootstrap uncertainty without protected-stratum regressions.
7. Measure quality and latency of adaptive rendering, tiling, crop padding, retries, and orientation separately before combining them.

## Task 10: Add critical-field metrics, registry modes, and adapter compatibility

**Files:**

- Create: `src/evaluation/critical_field_ocr.py`
- Create: `tests/test_critical_field_ocr.py`
- Modify: `src/ocr/model_registry.py`
- Modify: `src/ocr/paddleocr_adapter.py`
- Modify: `src/ocr/result_normalizer.py`
- Modify: `src/ocr/pipeline.py`
- Modify: `src/inference/document_pipeline.py`
- Modify: `src/portable/cli.py`
- Modify: `scripts/extract_document.py`
- Modify or create: `run_ocr.py`
- Create: `tests/test_ocr_model_profiles.py`
- Create: `reports/ocr_upgrade/critical_field_metrics.json`
- Create: `reports/ocr_upgrade/ocr_model_comparison.csv`

**Steps:**

1. Add field normalization/matching tests for dates, amounts/tolerance/decimal separators, identifiers, currency, email, phone, organization, and address.
2. Register immutable original/custom detector, general recognizer, and Thai recognizer entries with upstream base, local/export paths, hashes, config/manifest hashes, license, domain, DEV metrics, and selected/default state.
3. Support `original|custom|auto` model flags and `original|custom|adaptive` OCR profiles with explicit missing-custom fallback warnings.
4. Preserve the normalized page output and source-coordinate polygon/box/reading-order/page-number contract.
5. Compare original/original, custom/original, original/custom, custom/custom, adaptive model selection, and the full adaptive inference pipeline on DEV_SELECT.
6. Choose the simplest accepted default by the documented composite gate; keep original behavior separately runnable and regression-tested.

## Task 11: Rebuild OCR-realistic streams and compare LayoutXLM adaptation

**Files:**

- Modify: `src/information_extraction/model_dataset.py`
- Modify: `src/information_extraction/multitask_data.py`
- Modify: `scripts/prepare_model_dataset.py`
- Modify: `scripts/train_multitask_model.py`
- Create: `tests/test_ocr_noise_augmentation.py`
- Modify: `tests/test_model_dataset.py`
- Modify: `tests/test_multitask_data.py`
- Create: `data/metadata/final_model_dataset_manifest_ocr_v2.csv`
- Create: `reports/ocr_upgrade/layout_stream_trials.csv`
- Create: `reports/ocr_upgrade/layout_adaptation_trials.csv`

**Steps:**

1. Increment preprocessing/cache versions and reject all stale OCR-derived examples.
2. Rebuild bounded public TRAIN PaddleOCR and hybrid examples with locked detector/recognizer hashes and deployable inference-only tokens/geometry.
3. Compare 80/10/10, 70/15/15, and existing stream ratios on DEV_SELECT; record alignment, entity, relation, and canonical retention.
4. Add deterministic low-rate OCR noise while preserving unmodified examples, masking unsafe labels, and retaining relations only when entities survive.
5. Compare continued adaptation from the current final checkpoint for 1–2 epochs against fresh task training from the approved `microsoft/layoutxlm-base`, with identical public data, budgets, seeds, selection rules, and DEV_SELECT procedures.
6. Select using reference-token, real-OCR, fixed-37°, critical-field, relation, canonical-evidence, and document metrics; enforce ≤0.02 reference entity/canonical regression.
7. Reload and hash the selected checkpoint in a clean process.

## Task 12: Recalibrate and freeze the release candidate

**Files:**

- Modify: `src/information_extraction/multitask_calibration.py`
- Modify: `scripts/calibrate_multitask_model.py`
- Modify: `tests/test_multitask_calibration.py`
- Create: `reports/ocr_upgrade/calibration_metrics.json`
- Create: `reports/ocr_upgrade/candidate_manifest.json`

**Steps:**

1. Freeze code, OCR model/dictionary/config hashes, LayoutXLM checkpoint, stream manifest, and candidate configuration.
2. Fit temperature, global/canonical/relation/abstention thresholds on DEV_CALIBRATION only.
3. Bind calibration to build/checkpoint/detector/recognizer/preprocessing hashes and require private/Gmail rows zero.
4. Measure NLL, ECE, coverage, selective accuracy, and F1 before/after abstention.
5. Add a locked evaluator guard that refuses any candidate/hash/config/calibration drift.
6. Perform the final prediction-affecting code review locally before first TEST_IN_DOMAIN execution; any subsequent prediction change invalidates and must explicitly rerun final evaluation.

## Task 13: Execute final public, unseen, angle, and private evaluation

**Files:**

- Modify: `scripts/evaluate_information_extraction.py`
- Modify: `scripts/evaluate_end_to_end_angles.py`
- Modify: `scripts/evaluate_unseen_coru.py`
- Modify: `scripts/evaluate_private_gmail.py`
- Create: `reports/ocr_upgrade/locked_test_ocr_metrics.json`
- Create: `reports/ocr_upgrade/locked_test_end_to_end_metrics.json`
- Create: `reports/ocr_upgrade/angle_metrics.json`
- Create: `reports/ocr_upgrade/unseen_coru_metrics.json`
- Create: `reports/ocr_upgrade/private_aggregate.json`

**Steps:**

1. Run locked TEST_IN_DOMAIN only after the candidate guard passes and record OCR/layout/end-to-end metrics by the required strata.
2. Run the 18-angle grid with K-Means disabled and calculate coverage/entity/canonical/critical-field retention.
3. Run deterministic CORU unseen evaluation without using it for any subsequent selection.
4. Run fixed private Gmail operation only after public evaluation, store details solely under `D:\CSX4201\vision-info-extraction-assets\private-evaluation\ocr-upgrade-final`, publish aggregates only, and make no configuration change from private output.
5. Compare quality, failure rate, and time against the frozen baseline without fabricating unsupported F1.

## Task 14: Update portable release, reports, documentation, and verification

**Files:**

- Create: `reports/ocr_upgrade/final_ocr_model_card.md`
- Create: `reports/ocr_upgrade/final_upgrade_summary.md`
- Create: `reports/ocr_upgrade/verification.json`
- Modify: `scripts/build_portable_release.py`
- Modify: `scripts/setup_portable_windows.ps1`
- Modify: `scripts/verify_information_extraction.py`
- Modify: `README.md`
- Modify: `SUMMARY.md`
- Modify: `config.yaml`
- Modify: requirements/lock files as actually needed
- Modify: `.gitignore`
- Modify: `docs/ocr_setup.md`
- Modify: `docs/information_extraction.md`
- Modify: `docs/evaluation.md`
- Modify: `docs/privacy.md`
- Modify: `docs/private_testing.md`
- Modify: `docs/PORTABLE_USAGE.md`
- Modify: `docs/THIRD_PARTY_NOTICES.md`
- Modify: `AGENTS.md`
- Modify: `AGENT_MEMORY.md`
- Modify: `LESSONS.md`
- Modify: release notes

**Steps:**

1. Update only documentation that describes changed behavior, exact commands, artifacts, storage, data/license policy, metrics, limitations, or release operation.
2. Build the portable package with selected inference models plus the original fallback when size allows, excluding all training data, crops, caches, logs, private paths/content, and virtual environments.
3. Verify clean Windows CPU and GPU behavior and Docker Linux/AMD64 CPU behavior for setup, launch, image, PDF, multipage, rotation, Thai, visualization, profiles/fallbacks, JSON schema, and relocatable paths.
4. Emit a verification record for every required check with command, status, evidence path, timestamp, and hashes.
5. Run the complete original/new tests, compilation, environment/model/download/export/reload/benchmark/split/storage/cache/profile/integration/layout/calibration/evaluation/privacy/package checks.

## Task 15: Run the one permitted specialized review, fix confirmed defects, and publish

**Files:**

- Modify only files implicated by reproducible review findings
- Finalize: `reports/ocr_upgrade/verification.json`
- Finalize: `reports/ocr_upgrade/final_upgrade_summary.md`

**Steps:**

1. After training, selection, integration, adaptation, calibration, locked evaluation, docs, and passing tests, call one specialized `code-reviewer` with the exact evidence checklist from the task.
2. Reproduce every finding; reject unsupported findings; classify blockers, non-blocking defects, test gaps, documentation defects, optional improvements, and out-of-scope suggestions.
3. Fix confirmed in-scope defects with regression tests. If a fix changes predictions, mark prior final evaluation non-representative and rerun the affected locked evaluations transparently.
4. Use at most one follow-up review pass, and only when fixes were required.
5. Rerun full verification and privacy/package scans.
6. Inspect git status, ignored files, staged paths/diffs/sizes, secrets, credentials, private filenames/text/images/predictions, datasets, checkpoints, environments, caches, and temporary files.
7. Stage only public-safe source, tests, small configs, manifests, reports, documentation, registry metadata, checksums, and release tooling.
8. Commit clear stage-scoped changes and push `feat/domain-adapted-ocr` without force.

## Final handoff evidence

The completion response must report all 59 requested items: baseline identity and metrics; compatibility/environment/storage; benchmark/training/synthetic/Thai counts and licensing; every detector/recognizer trial and selected hashes/metrics; field/adaptive/tiling/crop/retry/orientation/ablation/comparison evidence; selected OCR profile and cache rebuild; LayoutXLM streams/trials/checkpoint/hash/calibration; locked TEST/angle/CORU/private/time/edge-case evidence; commands and verification; review findings/fixes/result; documentation/portable release/limitations; files; branch/commit/push; and exact setup/training/evaluation/inference commands.
