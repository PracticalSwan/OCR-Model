# Final Evaluation Protocol and Results

> **Historical evaluation record.** The local executable stack was archived and
> removed on 2026-09-06. Results below were not rerun after cleanup and do not
> describe a currently installed model. See [the archive record](../ARCHIVED.md).

## Split discipline

| Split | Allowed use |
|---|---|
| train | Fit model heads and encoder layers. |
| dev_select | Hyperparameters, checkpoint selection, and bounded OCR-profile selection. |
| dev_calibration | Temperature scaling and abstention thresholds only. |
| test_in_domain | One locked reference-token test plus predetermined robustness grids; never tune from it. |
| unseen_domain_test | CORU QA/OCR coverage after all choices are fixed. |
| private Gmail | Local aggregate operation only; never training, selection, thresholding, or accuracy. |

The final manifest build ID, manifest SHA-256, checkpoint SHA-256, and
calibration SHA-256 are validated before evaluation. Every public evaluator
rejects private/unmarked examples. Calibration is also checked against the
resolved OCR-stack binding before calibrated layout inference begins.

## Commands

```powershell
$ocr = 'D:\OCR_Model_Assets\environments\ie-ocr\Scripts\python.exe'
$layout = 'D:\OCR_Model_Assets\environments\ie-layout\Scripts\python.exe'
$checkpoint = 'D:\OCR_Model_Assets\checkpoints\layoutxlm_multitask\ocr_upgrade_fresh_b_noise'

& $layout scripts/evaluate_multitask_model.py `
  --profile final --checkpoint $checkpoint --split test_in_domain `
  --streams ground_truth --device cuda --group-by dataset language `
  --calibration models\multitask_calibration.json `
  --report-name ocr_upgrade_locked_test_ground_truth.json
& $ocr scripts/evaluate_locked_ocr_upgrade.py `
  --checkpoint $checkpoint --device gpu:0
& $layout scripts/evaluate_layout_angles.py `
  --checkpoint $checkpoint --device cuda --pages-per-dataset 10
& $ocr scripts/evaluate_end_to_end_angles.py `
  --checkpoint $checkpoint --device gpu:0 --pages-per-dataset 1
& $ocr scripts/evaluate_unseen_coru.py `
  --checkpoint $checkpoint --device gpu:0 --limit 100
& $ocr scripts/evaluate_private_gmail.py `
  --layout-checkpoint $checkpoint --device gpu:0 --limit 2
& $ocr scripts/run_integration_smoke.py --device gpu:0 `
  --model-checkpoint $checkpoint
python scripts/compile_ocr_upgrade_reports.py
& $layout scripts/compile_final_reports.py `
  --heldout-report ocr_upgrade_locked_test_ground_truth.json
```

Private testing has a separate command and output boundary in
[private_testing.md](private_testing.md).

The reference-token and image-to-JSON `TEST_IN_DOMAIN` commands above are
audit records of one-time executions. The locked set has already been
consumed; do not rerun it to choose a checkpoint, OCR profile, calibration, or
threshold.

## Locked in-domain layout results

The ground-truth-token test covers 1,760 examples (FATURA 1,584, SROIE 146,
FUNSD 30) and 1,761 windows.

| Metric | Raw | Calibrated/abstained |
|---|---:|---:|
| Entity micro-F1 | 0.9827 | 0.9835 |
| Entity macro-F1 | 0.7718 | 0.7512 |
| Canonical-evidence micro-F1 | 0.9795 | 0.9860 |
| Relation micro-F1 | 0.5726 | 0.5603 |

Document accuracy is 1.0. Calibration improves entity and canonical-evidence
micro-F1 while slightly lowering relation micro-F1 under abstention. The final
report retains both raw and calibrated values.

Dataset interpretation:

- FATURA dominates the test and has entity F1 1.0 and canonical evidence
  0.9835 but no
  relation supervision;
- SROIE has entity F1 0.8969 and canonical-evidence F1 0.8538;
- FUNSD has entity F1 0.7559 and supplies the relation score (0.5726);
- B-HEADER F1 is 0.4088 and QUESTION_ANSWER relation F1 is 0.4776.

## Locked image-to-JSON results

The selected original OCR profile and fresh LayoutXLM checkpoint processed all
1,760 `TEST_IN_DOMAIN` pages once, with zero failures and nonempty output rate
1.0.

| Metric | Result |
|---|---:|
| Polygon precision / recall / F1 | 0.2780 / 0.6076 / 0.3815 |
| Recognized-text coverage | 0.1663 |
| CER / WER | 0.8337 / 0.9692 |
| Critical-field exact match | 0.3496 over 11,275 fields |
| Entity F1 | 0.0944 |
| Relation F1 | 0.0111 |
| Canonical-field accuracy | 0.2534 |
| Document-type accuracy | 0.9977 |
| Table availability | 0.9307 |
| Mean processing time | 2.0821 seconds/page |

The full locked run missed the requested OCR and end-to-end quality targets.
No post-test component, threshold, or checkpoint change was made. The earlier
three-page baseline is retained for provenance, but it is not a controlled
paired comparison with this 1,760-page result.

## Layout-only angle robustness

Thirty balanced public test pages are evaluated at:

```text
0, 1, 15, 30, 37, 45, 60, 89, 90, 91, 135, 179, 180, 225, 269, 270, 315, 359
```

The page and all target geometry rotate together, isolating the learned layout
heads from OCR. All 540 cases completed without failure. Across angles,
calibrated entity F1 is 0.7683–0.7973, canonical F1 0.9640–0.9748, relation
F1 0.3358–0.5553, and composite 0.7324–0.8005. Minimum retention versus
upright is 0.9669 entity, 0.9926 canonical, 0.6144 relation, and 0.9186
composite.

## End-to-end angle results

One deterministic public test page from each of FATURA, FUNSD, and SROIE plus
one synthetic Thai page runs at every required angle: 72 total cases. K-Means
is disabled for this test and never controls OCR.

| Metric across the 18 public angle aggregates | Range |
|---|---:|
| Nonempty output rate | 1.0–1.0 |
| Recognized-text coverage | 0.3068–0.3839 |
| WER | 0.7386–0.8540 |
| Polygon detection F1 | 0.1707–0.1941 |
| Entity F1 | 0.1326–0.1807 |
| Relation F1 | 0–0.0342 |
| Canonical-field accuracy | 0.4444–0.5556 |
| Orientation-selection accuracy | 0.3333–1.0000 |

Synthetic Thai text has 1.0 recognized-text coverage, 0 WER, a nonempty
result, and the Thai route at all 18 angles. It proves routing/rotation
integration only; there is no compatible labeled public Thai benchmark.

The large gap between reference-token and end-to-end scores is the central
quality result: learned layout heads work on aligned text/boxes, while OCR
coverage and detection on unfamiliar layouts constrain real extraction.

## OCR preprocessing selection

The public preprocessing ablation uses the fixed 400-page DEV_SELECT benchmark:
283 FATURA, 20 FUNSD, and 97 SROIE pages. `grayscale_normalized` wins the
preprocessing-only composite score at 0.521288, versus 0.517138 for original
preprocessing. Denoising is rejected at 0.461808. This preprocessing result is
retained for the optional adaptive profile rather than promoted on its own.

The full A-F comparison evaluates OCR and downstream extraction together on
the same 400 pages:

| Configuration | Status | Score | Entity F1 | WER | Seconds/page |
|---|---|---:|---:|---:|---:|
| A: all original | selected | 0.368311 | 0.137946 | 0.758754 | 2.573 |
| C: custom general recognizer | ineligible | 0.395077 | 0.436529 | 0.735421 | 7.457 |
| E: accepted registry choices | eligible | 0.368899 | 0.137946 | 0.758754 | 2.543 |
| F: adaptive stack | eligible | 0.339679 | 0.137002 | 0.761867 | 6.158 |

B and D are unavailable because no custom detector passed its bounded
hardware/acceptance gate. C is not eligible because its custom general
recognizer failed the WER and Turkish-character acceptance criteria. E is
effectively identical to A on this non-Thai benchmark and improves the
selection score by only 0.000588, below the material-gain threshold. The
frozen default is therefore A (`original`). All comparison rows have zero
failures and zero private rows; TEST, CORU, and Gmail were not used.

## Unseen CORU

CORU contributes zero fit, dev, calibration, or in-domain test rows. On a
deterministic 100-page sample from its 1,261-page unseen population:

| Metric | Result |
|---|---:|
| Successful / failed pages | 100 / 0 |
| Nonempty OCR rate | 1.0 |
| QA answers found in OCR | 78.53% of 4,001 |
| Canonical exact-match accuracy | 15.68% of 523 applicable fields |
| Mean entities / relations / non-null fields | 13.35 / 3.28 / 6.46 |
| Mean processing time | 25.96 seconds/page |

CORU QA has answer strings but no compatible token polygons. Entity and
relation F1 are therefore undefined rather than fabricated.

## Private and integration evidence

The fixed checkpoint completed 2/2 anonymous local Gmail documents and pages
with zero failures and nonempty output in 126.586 wall seconds. The public
artifact contains aggregates only and explicitly declares no filename, path,
OCR text, image, or per-document prediction. There is no private ground truth
and no accuracy claim.

The synthetic integration runner covers upright image, 45° image,
general-to-Thai two-page PDF, and Thai metadata routing. The durable report
records the four case assertions. Complete verification checks that summary
without requiring generated fixtures and outputs to remain on disk. Checkpoint,
calibration, model, and release provenance are verified by their dedicated
checks instead of being duplicated in the smoke-test report.

## Authoritative artifacts

- `reports/ocr_upgrade/final_ocr_model_card.md`
- `reports/ocr_upgrade/error_analysis.md`
- `reports/ocr_upgrade/final_upgrade_summary.md`
- `reports/ocr_upgrade/locked_test_{ocr,end_to_end}_metrics.json`
- `reports/final_model/{ocr,entity,relation,field,angle,language,dataset}_metrics.json`
- `reports/final_model/verification.json`
- `reports/final_model/evaluations/ocr_upgrade_locked_test_ground_truth.json`

These reports are bounded academic evidence. They do not establish safety for
automated financial, legal, identity, or other high-stakes decisions.

The complete data, detector, recognizer, adaptive-component, LayoutXLM trial,
calibration, cache, and reproduction record is
[`OCR_UPGRADE_RELEASE_NOTES.md`](OCR_UPGRADE_RELEASE_NOTES.md).

The learned canonical-evidence metrics cover the 14 configured
model-supervised fields. End-to-end canonical output may include any of the 27
schema-supported fields after learned evidence, rules, and hybrid resolution;
the two denominators must not be compared as if they were the same task.

See [`../reports/README.md`](../reports/README.md) for report authority and
lifecycle. Portable verification is generation-specific and is current only
when its source tree, `BUILD_INFO.json`, archive, sidecar, tag, and live
Release asset agree. The complete verifier recomputes every ledger
`evidence_sha256`, rejects a JSON report whose embedded source commit disagrees
with the ledger, and requires portable evidence to carry the exact source
commit, candidate-tree SHA-256, candidate count, and clean-build flag. Those
values must agree between the independently selected installed package's
`BUILD_INFO.json` and the portable report. Ledger rows bind to the exact report
by a recomputed evidence SHA-256 instead of duplicating those generation
fields. A self-consistent older report is not accepted as the current local
generation. The report cannot choose that trust anchor: complete mode
independently designates `D:\OCR_Model` (or an explicit `--portable-package`)
and the `v1.1.0-ocr-upgrade` Git tag, requires the report's package directory
to match, and requires `BUILD_INFO.json` to name the tag's exact commit.
