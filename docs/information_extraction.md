# Information-Extraction Workflow

> **Restoration-only workflow (archived 2026-09-06).** The referenced
> `D:\OCR_Model_Assets` environments and assets were removed. Recreate and
> revalidate them before running these commands. See
> [the archive record](../ARCHIVED.md).

## 1. Normalize and verify public annotations

```powershell
$ocr = 'D:\OCR_Model_Assets\environments\ie-ocr\Scripts\python.exe'
& $ocr scripts/normalize_ie_annotations.py --force
& $ocr scripts/verify_ie_annotations.py
```

The normalizer writes supported public records below the ignored
`data/processed/normalized_ie_annotations` tree and emits a public-safe
manifest, schema, mapping report, summary, and categorized error CSV. It never
modifies raw data. Eligibility is decided before materialization, so excluded
records do not create orphan outputs.

## 2. Build the leakage-safe final model dataset

```powershell
& $ocr scripts/prepare_model_dataset.py `
  --profile final --device gpu:0 --force `
  --streams ground_truth paddleocr hybrid ocr_noise `
  --ocr-variant-train-limit 1638 `
  --ocr-variant-dev-select-limit 400 `
  --manifest-output data\metadata\final_model_dataset_manifest_ocr_v2.csv `
  --ocr-profile original
& $ocr scripts/report_final_dataset.py
```

The split builder groups document and duplicate identities before assigning
train, `dev_select`, `dev_calibration`, and `test_in_domain`. CORU is reserved
as `unseen_domain_test`. The final manifest is profile/build-bound and records
source, stream, OCR model hashes, labels, split, and privacy status.

Executed build `final-8bfcf79fed04e375` contains 16,781 examples:

| Split | Examples |
|---|---:|
| train | 12,455 |
| dev_select | 1,913 |
| dev_calibration | 653 |
| test_in_domain | 1,760 |

The streams are 11,172 ground-truth, 2,038 PaddleOCR, 2,038 hybrid, and 1,533
train-only OCR-noise examples. TRAIN contains 7,646 ground-truth, 1,638
PaddleOCR, 1,638 hybrid, and 1,533 noise examples. Training targets include
783,680 entity tokens, 216,063 canonical-evidence tokens, 78,095 relation
pairs, and 10,535 positive relations. Gmail fit rows are zero. The manifest
SHA-256 is
`02d173cfcadeb6e7f0c061cba099568949423243c0b9f29a6628ce7750554133`.

## 3. Train the final multi-task checkpoint

```powershell
$layout = 'D:\OCR_Model_Assets\environments\ie-layout\Scripts\python.exe'
$manifest = 'data\metadata\final_model_dataset_manifest_ocr_v2_b_noise.csv'
$checkpoint = 'D:\OCR_Model_Assets\checkpoints\layoutxlm_multitask\ocr_upgrade_fresh_b_noise'
& $layout scripts/train_multitask_model.py `
  --profile final --manifest $manifest --checkpoint $checkpoint `
  --device cuda --epochs 4 --trial-id fresh_b_noise `
  --publish-canonical-report --encoder-learning-rate 0.00002 `
  --head-learning-rate 0.0001 --upright-probability 0.6 `
  --streams ground_truth paddleocr hybrid ocr_noise
& $layout scripts/report_multitask_training.py
```

The selected fresh run completed four epochs, 50,212 microsteps, and 12,556
optimizer steps. Checkpoint selection combines upright dev-select quality
(0.7) with a fixed 37° robustness slice (0.3); epoch 4 scored 0.843343 and was
selected. Reloaded logits match exactly. Continued A, continued B, continued
B-plus-noise, and fresh B-plus-noise were all evaluated against explicit
reference, canonical, document, and relation regression gates; the fresh run
had the highest downstream selection score, 0.800401.

Current local checkpoint:

```text
D:\OCR_Model_Assets\checkpoints\layoutxlm_multitask\ocr_upgrade_fresh_b_noise
```

`model.safetensors` SHA-256:

```text
f257538849bd2067a9df9df83385aa10ae468d0499510fb0621a03a5f0155180
```

The 1.1 GB weight file and resumable optimizer state remain on D: and are not
committed. The source/derived license is CC-BY-NC-SA-4.0.

## 4. Calibrate without touching test or private data

```powershell
& $layout scripts/calibrate_multitask_model.py `
  --profile final --checkpoint `
  'D:\OCR_Model_Assets\checkpoints\layoutxlm_multitask\ocr_upgrade_fresh_b_noise' `
  --manifest data\metadata\final_model_dataset_manifest_ocr_v2.csv `
  --device cuda --ocr-profile original `
  --streams ground_truth paddleocr `
  --output models\multitask_calibration.json
```

Calibration uses 653 public `dev_calibration` examples. It writes
`models/multitask_calibration.json` with temperatures and thresholds bound to
the exact build, manifest, checkpoint, and OCR-stack hashes. Runtime validates
that OCR binding before it starts calibrated layout inference. It records zero
private and zero Gmail rows. The calibration SHA-256 is
`81a55061554d760e42c492d16f283a78fea32c64fd697947cbeeb8c7c1e9fc44`.

## 5. Run inference

```powershell
& $ocr scripts/extract_document.py `
  --input 'path\to\document.pdf' `
  --output 'D:\OCR_Model_Assets\generated\run-001' `
  --language auto --device gpu:0 `
  --model-checkpoint `
  'D:\OCR_Model_Assets\checkpoints\layoutxlm_multitask\ocr_upgrade_fresh_b_noise'
```

The OCR process selects orientation, preprocessing, and general/Thai route. A
persistent CUDA layout worker produces calibrated entities, document type,
canonical evidence, and typed relations. Evidence rules validate fields,
arithmetic, generic key/value pairs, and tables. Results are schema-validated
before atomic write; page errors can be isolated with
`--continue-on-page-error`.

The configured checkpoint, calibration, and OCR stack are required by
default. Initialization or runtime worker failure aborts the run. For an
explicit degraded developer workflow, add
`--allow-generic-layout-fallback`; that path retains OCR plus evidence/rule
output but must not be reported as calibrated LayoutXLM inference.

The learned canonical-evidence head directly supervises 14 configured fields.
The result schema supports 27 fields after learned evidence, validation rules,
and hybrid resolution are combined.

Use `--save-visualization` for public/debug inputs only. For private inputs,
follow [private_testing.md](private_testing.md); detailed private outputs must
remain under the ignored D: root.

## 6. Reproduce final evidence

```powershell
& $layout scripts/evaluate_multitask_model.py `
  --profile final --checkpoint `
  'D:\OCR_Model_Assets\checkpoints\layoutxlm_multitask\ocr_upgrade_fresh_b_noise' `
  --split test_in_domain --streams ground_truth --device cuda `
  --group-by dataset language --calibration models\multitask_calibration.json `
  --report-name ocr_upgrade_locked_test_ground_truth.json
& $layout scripts/evaluate_layout_angles.py `
  --checkpoint 'D:\OCR_Model_Assets\checkpoints\layoutxlm_multitask\ocr_upgrade_fresh_b_noise' `
  --device cuda --pages-per-dataset 10
& $ocr scripts/evaluate_end_to_end_angles.py `
  --checkpoint 'D:\OCR_Model_Assets\checkpoints\layoutxlm_multitask\ocr_upgrade_fresh_b_noise' `
  --device gpu:0 --pages-per-dataset 1
& $ocr scripts/evaluate_unseen_coru.py `
  --checkpoint 'D:\OCR_Model_Assets\checkpoints\layoutxlm_multitask\ocr_upgrade_fresh_b_noise' `
  --device gpu:0 --limit 100
& $ocr scripts/evaluate_private_gmail.py `
  --layout-checkpoint 'D:\OCR_Model_Assets\checkpoints\layoutxlm_multitask\ocr_upgrade_fresh_b_noise' `
  --device gpu:0 --limit 2
& $ocr scripts/run_integration_smoke.py --device gpu:0 `
  --model-checkpoint 'D:\OCR_Model_Assets\checkpoints\layoutxlm_multitask\ocr_upgrade_fresh_b_noise'
python scripts/compile_ocr_upgrade_reports.py
& $layout scripts/compile_final_reports.py `
  --heldout-report ocr_upgrade_locked_test_ground_truth.json
```

The two `TEST_IN_DOMAIN` evaluations above have already executed once. They
are recorded for audit and must not be rerun to select, calibrate, or tune a
model. See [evaluation.md](evaluation.md) for results and limitations and
[OCR_UPGRADE_RELEASE_NOTES.md](OCR_UPGRADE_RELEASE_NOTES.md) for the exact OCR
data, training, rejection, cache, adaptive-component, and inference commands.
