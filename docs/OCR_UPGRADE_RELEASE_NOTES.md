# Domain-adapted OCR upgrade release notes

> **Historical release record.** The source was archived locally on 2026-09-06,
> and its runtime/model assets were removed. The hashes and results below remain
> the frozen release evidence. See [the archive record](../ARCHIVED.md).

**Source branch:** `feat/domain-adapted-ocr`

**Merged:** PR #2
into `main` at `c6303f6843de9af1c7c97fde1ef6ff43e01de553` on 2026-07-29

**Release:** `v1.1.0-ocr-upgrade`.
The July 29 initial asset targeted
`fcae32edc193ff6574bf99362da0e2368d5ef464`. The same release and asset names
now identify the GUI-state correction built from clean commit
`b7a2d10993cfd595c569797556e87eeed49aeaff`; the live ZIP and sidecar agree on
SHA-256 `660d56b9d64d7ddabeb1ea4ca945f6ea58e1d7b58031a8cffaacfedfdc3c3448`.

**Evidence window:** 2026-07-24 through 2026-07-30

**Status:** implemented and evaluated; not production-ready

This upgrade replaces the three-page OCR selection evidence with a
deterministic 400-page public `DEV_SELECT` benchmark, executes bounded
PaddleOCR detector and recognizer trials, rebuilds OCR-realistic LayoutXLM
streams, recalibrates the selected checkpoint, and records one locked
`TEST_IN_DOMAIN` image-to-JSON evaluation. `TEST_IN_DOMAIN`, CORU, and Gmail
were not used to fit or select any component.

The selected global OCR profile remains `original`. This is an evidence-based
result, not a failed release: the custom general recognizer missed a required
WER-improvement gate and catastrophically regressed Turkish characters, while
the custom detector trials could not produce an eligible checkpoint within the
verified 8,151 MiB GPU envelope. A custom Thai recognizer is available only in
the explicit `custom` and `adaptive` profiles; its evidence is synthetic.

## Baseline and compatibility

The frozen baseline is commit
`af0816b83fde1b3fb8a25802a1967350de654812`, tagged
`pre-ocr-upgrade-baseline`.

| Evidence | Frozen baseline |
|---|---:|
| Host tests | 244 passed, 2 skipped |
| OCR-runtime tests | 123 passed |
| Layout-runtime tests | 2 passed |
| Upright pages | 3 |
| Polygon F1 | 0.339511 |
| Recognized-text coverage | 0.402789 |
| WER | 0.699888 |
| End-to-end entity F1 | 0.177200 |
| End-to-end relation F1 | 0.007491 |
| Canonical-field accuracy | 0.444444 |
| Prior 18-angle duration | 335.749 seconds |

PaddlePaddle GPU and CUDA PyTorch remain isolated in separate Python 3.10
processes because their Windows wheels load incompatible CUDA/cuDNN
dependencies. Direct Paddle verification now registers the training
environment's NVIDIA DLL directories before importing Paddle, so the exact
standalone verifier works without relying on a setup shell's inherited
`PATH`.

The verified OCR training environment contains PaddlePaddle GPU 3.3.0,
PaddleOCR 3.7.0, PaddleX 3.7.2, CUDA 13 NVRTC 13.0.48, and a clean official
PaddleOCR v3.7.0 checkout at commit
`b03f46425e8ff4442b268ce449e3eef758146cd4`.

## Public benchmark and training data

The 400-page benchmark is a deterministic, group-aware subset of
`DEV_SELECT`. Its manifest SHA-256 is
`1786af155d1c2c3e69224c87f1885da3a62ef59617d7d57ff4d7d4bea661e3ec`.

| Dataset | Pages |
|---|---:|
| FATURA | 283 |
| FUNSD | 20 |
| SROIE | 97 |
| Total | 400 |

Only public `TRAIN` pages were materialized for OCR weight fitting:

| Corpus | Train | DEV_SELECT | Other facts |
|---|---:|---:|---|
| Detector pages | 7,646 | 1,109 | 8,755 unique pages, 163,679 regions, 4 exact duplicates skipped |
| Real recognition crops | 119,773 | 18,113 | 71,729 line crops and 66,157 critical-word crops |
| Dictionary-compatible real recognition crops | 119,772 | 18,111 | 3 private-use-code-point targets excluded |
| General synthetic crops | 29,943 | 4,528 | requested 20% raw synthetic share |
| Dictionary-compatible mixed general crops | 147,844 | 22,356 | 18.9876%/18.9882% synthetic |
| Thai synthetic crops | 12,000 | 2,000 | synthetic selection boundary only |

The generator produced 48,471 synthetic images and independently replayed
48,477 checksum entries. It uses Noto Sans and Noto Sans Thai from the official
Google Fonts repository at pinned commit
`9fab8b6cc7b2f20376914fd765d918c698c66d75`, under the SIL Open Font License
1.1. The candidate `openthaigpt/thai-ocr-evaluation` dataset was not
downloaded because its card did not provide per-sample source and licensing
provenance sufficient for this training and redistribution boundary.

All data reports record zero private rows. Raw inputs remain read-only, and
large crops, images, lists, caches, environments, and checkpoints remain on
`D:\OCR_Model_Assets`.

## Detector trials

| Trial | Outcome |
|---|---|
| Official `PP-OCRv6_medium_det` | Selected. Precision 0.446015, recall 0.648031, F1 0.528372, small-text recall 0.696776, critical-region recall 0.762317, 0.0610 seconds/page. |
| FP32 fine-tune | Bounded stop after 20 steps: 9,777 MiB allocated, 10,129 MiB reserved, projected runtime over ten hours. |
| Initial O2 AMP fine-tune | Failed before a valid window because CUDA 13 NVRTC was unavailable. |
| NVRTC-enabled O2 AMP fine-tune | Bounded stop after repeated inf/NaN loss-scale reductions; 12,108 MiB allocated, 13,639 MiB reserved, projected runtime over six hours. |

No custom detector was promoted. The selected original inference-tree SHA-256
is `eccf59cf53c201173dbabb4e45115d067414e4db8aeeb37a84e0b035afba494d`.

## Recognizer trials

| Track | Trial | WER | CER | Exact line | Decision |
|---|---|---:|---:|---:|---|
| General | Original PP-OCRv6 medium | 0.755776 | 0.594282 | 0.327339 | Selected |
| General | One-epoch real-public custom | 0.669095 | 0.571674 | 0.568393 | Rejected |
| Thai synthetic | Original PP-OCRv5 mobile | 0.234690 | 0.035907 | 0.544500 | Baseline |
| Thai synthetic | Custom | 0.071324 | 0.011100 | 0.866000 | Accepted for explicit custom/adaptive use |

The custom general model improved several aggregate and critical-field
development metrics, but its WER gain was below the declared 15% gate and its
Turkish-character accuracy fell from 0.803351 to 0.000882. The selected
original general recognizer inference-tree SHA-256 is
`6c46447e05189eb3f863dc75855f0cfccf16a4af188a249861377c69216f40b1`.

The custom Thai checkpoint SHA-256 is
`96bbda4bd94ebeee676e07a3f7c11339b707a506433bbb24f63b54f5459f26e9`;
its exported inference tree SHA-256 is
`0876e624221bf0ff2d888506c7b9fa84eacc99f98394b424e81c098769d91e73`.
Those values describe synthetic development performance, not real-world Thai
accuracy.

## Adaptive component evidence

The optional adaptive profile keeps every component bounded and
provenance-bound:

| Component | Result |
|---|---|
| Preprocessing | `grayscale_normalized` scored 0.521288 versus 0.517138 for original; retained only for `adaptive`. |
| Adaptive PDF DPI | 30 pages; 0 rerender triggers and 0 selected 300-DPI pages. At 200/adaptive DPI, coverage was 0.464355, WER 0.682442, critical exact match 0.625000. Forced 300 DPI measured 0.476432/0.669753/0.632813. |
| Tiling | 100 pages; 0 triggered, 0 selected, 0 duplicates; no quality gain and +1.1859 seconds/page. |
| Crop padding | Profile C selected: composite 0.404313 versus 0.380833 for A and 0.379396 for B. |
| Recognition retries | 26 pages, 40 words, 160 candidates, 28 non-original selections; entity F1 +0.000231, WER +0.000964, and +0.8085 seconds/page. |
| Orientation | 162 page-angle cases; 68.52% selection accuracy, 28.61° mean error, polygon F1 0.514075, coverage 0.459114, WER 0.656577. K-Means was never instantiated. |

The combined adaptive stack measured polygon F1 0.496780, text coverage
0.475706, WER 0.658165, critical exact match 0.307876, entity F1 0.232870,
canonical accuracy 0.236277, relation F1 0.062011, and 5.4849 seconds/page in
its component evaluation. It was not selected as the global default.

## OCR profile comparison

All eligible comparisons used the same 400 public `DEV_SELECT` pages:

| Configuration | Status | Score | Entity F1 | WER | Seconds/page |
|---|---|---:|---:|---:|---:|
| A: all original | selected | 0.368311 | 0.137946 | 0.758754 | 2.573 |
| B: custom detector, original recognizers | unavailable | — | — | — | — |
| C: original detector, custom recognizers | ineligible | 0.395077 | 0.436529 | 0.735421 | 7.457 |
| D: custom detector, custom recognizers | unavailable | — | — | — | — |
| E: registry-selected components | eligible | 0.368899 | 0.137946 | 0.758754 | 2.543 |
| F: adaptive stack | eligible | 0.339679 | 0.137002 | 0.761867 | 6.158 |

Configuration E's 0.000588 score gain over A was below the declared material
gain threshold. Configuration C contained the rejected general recognizer.
The default is therefore A (`original`). Original, custom, and adaptive
OCR experiments remain independently invocable, but only `original` is bound
to the shipped LayoutXLM calibration. Custom/adaptive runs require explicit
generic-layout fallback and are not calibrated LayoutXLM extraction.

OCR cache keys now include the selected model identities and hashes,
preprocessing/configuration hashes, language route, source hash, transform, and
profile. Old or mismatched entries fail validation. Private inputs are not
written to the public OCR cache.

## OCR-realistic LayoutXLM adaptation

The selected data build is `final-8bfcf79fed04e375`, manifest
`data/metadata/final_model_dataset_manifest_ocr_v2.csv`, SHA-256
`02d173cfcadeb6e7f0c061cba099568949423243c0b9f29a6628ce7750554133`.

| Split | Examples |
|---|---:|
| TRAIN | 12,455 |
| DEV_SELECT | 1,913 |
| DEV_CALIBRATION | 653 |
| TEST_IN_DOMAIN | 1,760 |
| Total | 16,781 |

The streams contain 11,172 ground-truth, 2,038 PaddleOCR, 2,038 hybrid, and
1,533 train-only OCR-noise examples. TRAIN contains 7,646 ground-truth,
1,638 PaddleOCR, 1,638 hybrid, and 1,533 noise examples. The v2 noise stream is
rotation-safe, affects 20% of eligible examples, caps changed tokens at 15%,
uses at most three transforms, and caps box jitter at 0.015. Across two planned
epochs, 3,066 of 3,066 deterministic rotation checks passed.

Five downstream candidates were compared:

| Trial | Reference entity F1 | Real-OCR entity F1 | Rotated real-OCR entity F1 | End-to-end entity F1 | Selection score |
|---|---:|---:|---:|---:|---:|
| Old baseline | 0.975842 | 0.761804 | 0.736446 | 0.137946 | 0.748084 |
| Continued A | 0.977986 | 0.824518 | 0.795247 | 0.128807 | 0.770244 |
| Continued B | 0.977389 | 0.836360 | 0.808532 | 0.122188 | 0.773621 |
| Continued B + noise | 0.977791 | 0.838234 | 0.811484 | 0.124906 | 0.775285 |
| Fresh B + noise | 0.980010 | 0.873922 | 0.862481 | 0.113191 | 0.800401 |

All declared reference, canonical, document, and relation regression gates
passed. The selected checkpoint is:

```text
D:\OCR_Model_Assets\checkpoints\layoutxlm_multitask\ocr_upgrade_fresh_b_noise
```

Its `model.safetensors` SHA-256 is
`f257538849bd2067a9df9df83385aa10ae468d0499510fb0621a03a5f0155180`.
The fresh run completed four epochs, 12,556 optimizer steps, selected epoch 4,
and reloaded with maximum logit difference 0.

## Calibration

Calibration used all 653 public `DEV_CALIBRATION` examples and no test, CORU,
or private data. The bound calibration SHA-256 is
`81a55061554d760e42c492d16f283a78fea32c64fd697947cbeeb8c7c1e9fc44`.

| Head | Temperature | Threshold |
|---|---:|---:|
| Entity | 1.095852 | 0.548886 |
| Canonical evidence | 1.295506 | 0.918779 |
| Relation | 0.666104 | 0.599866 |
| Document | 0.250000 | 1.000000 |

Post-calibration ECE is 0.001293 for entity, 0.006644 for canonical evidence,
0.005920 for relations, and 0 for document type.

## Locked and unseen evaluation

The reference-token `TEST_IN_DOMAIN` evaluation completed 1,760/1,760
examples with zero failures. Raw entity/canonical/relation F1 is
0.982673/0.979516/0.572585; calibrated values are
0.983478/0.986035/0.560279. Document accuracy is 1.0.

The one-time locked image-to-JSON run also completed 1,760/1,760 pages with
zero failures:

| Metric | Locked result |
|---|---:|
| Polygon precision / recall / F1 | 0.277989 / 0.607604 / 0.381456 |
| Recognized-text coverage | 0.166269 |
| CER / WER | 0.833731 / 0.969186 |
| Critical-field exact match | 0.349623 over 11,275 fields |
| End-to-end entity F1 | 0.094438 |
| End-to-end relation F1 | 0.011087 |
| Canonical-field accuracy | 0.253392 |
| Document-type accuracy | 0.997727 |
| Nonempty output / failure rate | 1.0 / 0 |
| Mean processing time | 2.0821 seconds/page |

These full locked metrics miss the requested quality targets. No component,
threshold, or checkpoint was changed from this result.

The 18-angle layout-only grid processed 540/540 examples without failure.
Across angles, calibrated entity F1 was 0.768340–0.797279, canonical F1
0.964029–0.974820, relation F1 0.335849–0.555324, and composite
0.732401–0.800490. Minimum retention versus upright was 0.966867 entity,
0.992593 canonical, 0.614422 relation, and 0.918572 composite.

The 18-angle end-to-end grid processed 54 public and 18 synthetic Thai cases
without failure. Public coverage was 0.306826–0.383885, WER
0.738562–0.854031, detector F1 0.170732–0.194107, entity F1
0.132597–0.180723, relation F1 0–0.034188, and canonical accuracy
0.444444–0.555556. Synthetic Thai routing, coverage, and exact text recovery
passed at all 18 angles. K-Means controlled no OCR decision.

The fixed unseen CORU run completed 100/100 pages with nonempty output. It
found 78.53% of 4,001 answer strings and exactly matched 15.68% of 523
applicable canonical fields. Mean processing time improved from 31.0417 to
25.9567 seconds/page, a 1.1959× speedup. CORU has no compatible token polygons,
so entity/relation F1 is not fabricated.

The final private operation was deliberately bounded to two anonymous Gmail
documents and two pages. Both completed with nonempty output in 126.586 wall
seconds. The public aggregate contains no filenames, paths, OCR text, images,
or per-document predictions. Gmail fit rows remain zero. This is an operation
check, not an accuracy estimate.

## Reproduction commands

The commands below use the verified paths. Commands containing `--force`
replace their exact generated target and should be used only when a rebuild is
intended.

### Setup and verification

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_ie_environment.ps1
powershell -ExecutionPolicy Bypass -File scripts\setup_ocr_training_environment.ps1

$train = 'D:\OCR_Model_Assets\environments\ie-ocr-train\Scripts\python.exe'
$ocr = 'D:\OCR_Model_Assets\environments\ie-ocr\Scripts\python.exe'
$layout = 'D:\OCR_Model_Assets\environments\ie-layout\Scripts\python.exe'

& $train scripts\verify_ocr_training_environment.py --device gpu:0 --write-report
& $ocr scripts\download_ocr_models.py
& $ocr scripts\verify_ocr_models.py --device gpu:0
python scripts\verify_ocr_benchmark.py
python scripts\verify_ie_annotations.py
```

### Data preparation and OCR training

```powershell
python scripts\build_ocr_benchmark.py --force
python scripts\build_ocr_detection_dataset.py --profile final --force
python scripts\build_ocr_recognition_dataset.py --profile final `
  --line-crops --critical-word-crops --force
python scripts\generate_synthetic_recognition_data.py `
  --real-corpus-root 'D:\OCR_Model_Assets\data\recognition_training' `
  --output-root 'D:\OCR_Model_Assets\data\synthetic_recognition' `
  --font-root 'D:\OCR_Model_Assets\fonts\google-fonts\9fab8b6cc7b2f20376914fd765d918c698c66d75' `
  --synthetic-fraction 0.20 --thai-train-count 12000 `
  --thai-validation-count 2000 --seed 42 --force
python scripts\prepare_ocr_trial_lists.py --force

powershell -ExecutionPolicy Bypass -File scripts\train_ocr_detector.ps1 `
  -Definition configs\ocr_upgrade\detector\det_ft_official_aug.yml
powershell -ExecutionPolicy Bypass -File scripts\train_ocr_detector.ps1 `
  -Definition configs\ocr_upgrade\detector\det_ft_official_amp.yml
powershell -ExecutionPolicy Bypass -File scripts\train_ocr_detector.ps1 `
  -Definition configs\ocr_upgrade\detector\det_ft_official_amp_nvrtc.yml
powershell -ExecutionPolicy Bypass -File scripts\train_ocr_recognizer.ps1 `
  -Definition configs\ocr_upgrade\recognizer\rec_general_real.yml -Track general
powershell -ExecutionPolicy Bypass -File scripts\train_ocr_recognizer.ps1 `
  -Definition configs\ocr_upgrade\recognizer\rec_thai_synthetic.yml -Track thai
```

### Selected LayoutXLM training and calibration

```powershell
$manifest = 'data\metadata\final_model_dataset_manifest_ocr_v2_b_noise.csv'
$checkpoint = 'D:\OCR_Model_Assets\checkpoints\layoutxlm_multitask\ocr_upgrade_fresh_b_noise'

& $layout scripts\train_multitask_model.py `
  --profile final --manifest $manifest --device cuda --epochs 4 `
  --checkpoint $checkpoint --trial-id fresh_b_noise `
  --publish-canonical-report --encoder-learning-rate 0.00002 `
  --head-learning-rate 0.0001 --upright-probability 0.6 `
  --streams ground_truth paddleocr hybrid ocr_noise
& $layout scripts\calibrate_multitask_model.py `
  --profile final --checkpoint $checkpoint `
  --manifest data\metadata\final_model_dataset_manifest_ocr_v2.csv `
  --device cuda --ocr-profile original --streams ground_truth paddleocr `
  --output models\multitask_calibration.json
```

### Evaluation

The two locked TEST commands below are audit records of one-time executions.
The locked set has already been consumed. Do not rerun them to choose or tune a
model.

```powershell
& $layout scripts\evaluate_multitask_model.py `
  --profile final --checkpoint $checkpoint `
  --manifest data\metadata\final_model_dataset_manifest_ocr_v2.csv `
  --split test_in_domain --streams ground_truth --device cuda `
  --calibration models\multitask_calibration.json `
  --report-name ocr_upgrade_locked_test_ground_truth.json
& $ocr scripts\evaluate_locked_ocr_upgrade.py `
  --checkpoint $checkpoint --device gpu:0

& $layout scripts\evaluate_layout_angles.py `
  --checkpoint $checkpoint --device cuda --pages-per-dataset 10
& $ocr scripts\evaluate_end_to_end_angles.py `
  --checkpoint $checkpoint --device gpu:0 --pages-per-dataset 1
& $ocr scripts\evaluate_unseen_coru.py `
  --checkpoint $checkpoint --device gpu:0 --limit 100
& $ocr scripts\evaluate_private_gmail.py `
  --layout-checkpoint $checkpoint --device gpu:0 --limit 2
```

### Inference profiles

```powershell
$input = 'path\to\document.pdf'
$output = 'D:\OCR_Model_Assets\generated\example'

& $ocr scripts\extract_document.py --input $input --output "$output-original" `
  --language auto --device gpu:0 --ocr-profile original `
  --model-checkpoint $checkpoint --save-visualization
& $ocr scripts\extract_document.py --input $input --output "$output-custom" `
  --language auto --device gpu:0 --ocr-profile custom `
  --model-checkpoint $checkpoint --allow-generic-layout-fallback `
  --save-visualization
& $ocr scripts\extract_document.py --input $input --output "$output-adaptive" `
  --language auto --device gpu:0 --ocr-profile adaptive `
  --model-checkpoint $checkpoint --allow-generic-layout-fallback `
  --save-visualization
```

The shipped calibration is bound to the original OCR stack. Therefore the
portable one-command CLI offers only `original`/`auto` calibrated extraction.
The two commands above are explicit OCR experiments using generic/rule-only
layout fallback; they must not be reported as calibrated LayoutXLM inference.

## Independent review closure

The late evidence-based review found no blocker and one medium defect: an
explicit custom-general selection could load the rejected general recognizer
even though the default and portable package remained on the original. The
fix binds registry promotion to the SHA-256 of the executed acceptance report
and candidate report, requires every acceptance criterion to pass, and filters
`accepted: false` custom artifacts before any runtime path or hash loading.
The permitted follow-up caught and closed that early-loading edge with a
missing rejected-model fixture. Focused regression tests cover both
registry-construction refusal and runtime refusal/fallback.

## In-place release correction

The July 30 correction retains the `v1.1.0-ocr-upgrade` version and replaces
its asset in place rather than publishing a patch version. The behavioral
changes are:

- calibrated layout inference validates the calibration's OCR-stack binding
  before worker startup and fails closed on a required runtime worker error;
- generic/rule-only layout fallback requires explicit
  `--allow-generic-layout-fallback` opt-in and is not labeled calibrated
  LayoutXLM output;
- runtime and report compilation resolve the selected checkpoint from
  `config.yaml` rather than probing a legacy `final` sibling;
- portable CLI/GUI private-document mode uses opaque private-root run IDs,
  defaults on in the GUI, disables preview, uses and removes an opaque
  short-lived worker input, redacts source filenames/paths from result
  surfaces, and disables visualizations and downloadable archives; the single
  reusable GUI session cache is removed when the GUI shuts down;
- the learned 14-field canonical-evidence scope is distinguished from the
  27-field schema output contract;
- the rotation-stage 10 GiB reserve is named separately from the 15 GiB
  OCR/model/setup reserve; and
- release construction uses a clean Git candidate-tree hash, a new isolated
  D: staging target that the builder never deletes, no-follow verified copies,
  sample-evidence binding, an all-file completed-payload privacy scan, a full
  payload manifest, and ZIP-integrity validation.

The external model-registry build stdout evidence was normalized from UTF-16LE
with BOM to UTF-8 without BOM; its JSON content remains parse-valid. Three
byte-identical final-model manifest aliases are retained deliberately for
historical command/provenance compatibility. Report authority and supersession
rules are documented in [`../reports/README.md`](../reports/README.md).

The correction kept the same release and stable asset names. The two assets
were replaced in place, the existing tag was moved to the clean build commit,
and both live assets were downloaded and re-hashed. The local archive,
sidecar, live digest, tag, `BUILD_INFO.json`, and payload manifest now agree.
No patch version, temporary release assets, deletion sentinel, or custom
release-state protocol is used.

## GUI state and layout repair

The July 30 GUI repair removes an extraction-to-upload feedback edge that
caused a completed private run to trigger the document-change reset and erase
its own output. Extraction now updates only the seven result components.
Changing **Private processing** updates only the preview and helper text, while
an actual document change remains the deliberate result-reset boundary. The
selected upload remains usable for repeat runs until the single GUI session
cache is removed at shutdown.

The Gradio 6.0.1 footer Settings control was removed because it raised a
browser-side `TypeError: Illegal invocation`. The repaired layout adds a
bounded centered canvas, a distinct settings heading, shorter private-mode
copy, a full-width extraction action, a status card, responsive spacing, and
a dedicated Download tab. Browser verification used the safe bundled
`unknown_upright.png`: one real GPU extraction produced five populated fields,
the result remained present after a private-mode toggle, the upload remained
selected, the Settings button was absent, and the browser console had zero
errors. Public preview also waits briefly for Gradio's upload copy to become
visible, preventing a transient missing-file message during first render.

## Remaining limitations

- The full locked OCR and end-to-end metrics missed the requested accuracy
  targets; OCR remains the dominant bottleneck.
- The selected detector and general recognizer are the upstream originals
  because the custom candidates were not eligible under the declared gates.
- Detector training is not practical on the verified 8,151 MiB laptop GPU
  without changing the training footprint or hardware.
- Thai selection evidence is synthetic and does not establish performance on
  real Thai documents.
- Relation supervision remains concentrated in FUNSD.
- The three-page baseline and the 1,760-page locked run are not a controlled
  paired comparison.
- K-Means remains display-only; the failed exact-angle estimator remains
  disabled.
- Physical Apple hardware remains untested. The macOS route is a
  Docker Linux/AMD64 CPU path.
- Human review is required for financial, legal, identity, medical, or other
  consequential use.

## Portable release package

The initial July 29 clean package was built from commit
`fcae32edc193ff6574bf99362da0e2368d5ef464`:

```text
D:\OCR_Model.zip
size: 1,159,061,897 bytes
SHA-256: d539c54f02c8e5bd204266eaed7e7372c4fd077d3cfa4062dccb1f894eb7d746
```

Its sidecar matches, its 181 ZIP entries have no duplicate or traversal path,
and its privacy audit finds no raw/private data, outputs, or credentials. A
fresh package-local CPU setup passed the doctor probe and real upright,
rotated, two-page PDF, custom Thai, and adaptive inference. Every result was
nonempty, schema-valid, and had one visualization per page. An additional GPU
run had stable semantic parity with CPU, custom English correctly fell back to
the selected original general recognizer, and the loopback GUI returned HTTP
200. The clean archive excludes `.runtime`, `runtime.local.json`, and outputs.
Those historical custom/adaptive probes established OCR routing and
schema-valid degraded output; they did not prove calibrated LayoutXLM
inference. The corrected portable CLI therefore narrows calibrated choices to
original/auto.
That initial archive and matching sidecar were published under
`v1.1.0-ocr-upgrade`;
the current live assets supersede this historical package identity. The
historical public
`v1.0.0-build-week`
Release remains available.

The current GUI-state correction has 183 ZIP entries and a 182-file payload
manifest:

```text
D:\OCR_Model.zip
size: 1,159,080,320 bytes
SHA-256: 660d56b9d64d7ddabeb1ea4ca945f6ea58e1d7b58031a8cffaacfedfdc3c3448
```

A clean isolated build passed payload privacy, CRC, root-layout, duplicate,
traversal, and manifest checks. The installed `D:\OCR_Model` matches all 182
payload records while preserving its machine-local runtime, configuration, and
existing outputs. Its GPU doctor and one-page GPU extraction passed. The exact
Linux/AMD64 Docker image passed HTTP readiness, doctor, and one-page CPU
extraction; its test container, image, and network were then removed.
Browser verification covered the exact repaired source and matching installed
GUI payload. Current-generation evidence is in `portable_verification.json`;
the initial package's broader custom, adaptive, PDF, and GPU probes remain
clearly labeled in
`portable_verification_initial_fcae32e.json`.

After current verification, C: had 54.863 GiB free and D: had 417.321 GiB
free, both above the 15 GiB reserve.

Authoritative machine-readable evidence is under `reports/ocr_upgrade/`,
`reports/final_model/`, and `reports/information_extraction/`. Portable
verification is generation-specific: accept it as current only when its source
commit/tree, `BUILD_INFO.json`, archive, sidecar, tag, and live Release asset
all agree.
