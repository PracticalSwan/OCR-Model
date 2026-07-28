# OCR and Layout Environment Setup

## Why there are three environments

PaddlePaddle GPU and CUDA PyTorch load incompatible cuDNN DLLs in one Windows
process. `scripts/setup_ie_environment.ps1` therefore creates separate
Python 3.10 inference environments on D:, and OCR fine-tuning uses a third,
source-bound environment:

```text
D:\CSX4201\vision-info-extraction-assets\environments\ie-ocr
D:\CSX4201\vision-info-extraction-assets\environments\ie-layout
D:\CSX4201\vision-info-extraction-assets\environments\ie-ocr-train
```

The OCR environment contains PaddlePaddle GPU 3.3.0, PaddleOCR 3.7.0, PaddleX
3.7.2, and CPU-only PyTorch 2.8.0 required by PaddleX/ModelScope. The layout
environment contains PyTorch 2.8.0+cu128, Transformers 4.57.6, SentencePiece,
and Accelerate. The training environment contains PaddlePaddle GPU 3.3.0,
PaddleOCR 3.7.0, PaddleX 3.7.2, CUDA 13 NVRTC, and a clean PaddleOCR v3.7.0
source checkout pinned to commit
`b03f46425e8ff4442b268ce449e3eef758146cd4`.

## Setup

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_ie_environment.ps1
powershell -ExecutionPolicy Bypass -File scripts/setup_ocr_training_environment.ps1
```

The setup scripts verify Python 3.10, C:/D: capacity, packages, independent GPU
runtimes, D:-backed cache variables, the clean source revision, model
initialization, a forward/backward/optimizer step, O2 AMP, checkpoint reload,
and dependency integrity. Use limits sparingly: run a bounded profile before a
large download, alignment job, or training run. The scripts discover Python
through `py -3.10`; pass `-Python310 <path>` only when the Windows launcher is
unavailable.

Run the training environment verifier directly after setup:

```powershell
$train = 'D:\CSX4201\vision-info-extraction-assets\environments\ie-ocr-train\Scripts\python.exe'
& $train scripts\verify_ocr_training_environment.py --device gpu:0 --write-report
```

The verifier registers the environment-local CUDA 13 and cuDNN DLL
directories before importing Paddle. It does not depend on a setup shell's
inherited `PATH`. The executed report passes GPU model initialization,
FP32/O2 forward-backward-optimizer steps, save/reload, and `pip check` in
5.141 seconds.

The configured external root is:

```text
D:\CSX4201\vision-info-extraction-assets
```

It contains environments, Paddle/Hugging Face/Torch/pip caches, temporary
files, OCR cache, aligned datasets, checkpoints, generated documents, and
private outputs. These are ignored by Git.

## Download and verify models

```powershell
$ocr = 'D:\CSX4201\vision-info-extraction-assets\environments\ie-ocr\Scripts\python.exe'
& $ocr scripts/download_ocr_models.py
& $ocr scripts/verify_ocr_models.py --device gpu:0
& $ocr scripts/print_environment_report.py `
  --layout-python 'D:\CSX4201\vision-info-extraction-assets\environments\ie-layout\Scripts\python.exe'
```

Required exact identities:

- detector: `PP-OCRv6_medium_det`;
- general recognizer: `PP-OCRv6_medium_rec`;
- Thai recognizer: `th_PP-OCRv5_mobile_rec`.

The download report records directories and per-file SHA-256 hashes. Model
verification re-hashes those files before initializing the routes. A partial,
missing, or changed artifact fails closed.

The rotated checks are end-to-end selector assertions. A 90-degree fixture
must return nonempty OCR, select a nonzero cardinal correction, and recover
`INVOICE`, `TOTAL`, and `123.45`. A separate 17-degree fixture must recover the
same phrase through automatic fine deskew with reliable line evidence. An
empty engine call cannot pass either check.

The 400-page public DEV_SELECT preprocessing ablation selected
`grayscale_normalized` for the optional adaptive stack: its composite score was
0.521288 versus 0.517138 for original preprocessing. The subsequent full
400-page A-F model comparison still retained configuration A as the global
default: original detector, original general recognizer, original Thai
recognizer, and original preprocessing scored 0.368311 at 2.573 seconds/page.
Registry-selected configuration E produced the same OCR and downstream
metrics and scored only 0.000588 higher, below the material-gain threshold.
The adaptive F pipeline scored 0.339679 at 6.158 seconds/page. The higher
scoring C experiment was ineligible because its custom general recognizer
failed the acceptance gate. The locked test and private documents were never
used for either selection.

The selected default artifacts remain the original detector and general
recognizer. Their inference-tree SHA-256 values are
`eccf59cf53c201173dbabb4e45115d067414e4db8aeeb37a84e0b035afba494d`
and `6c46447e05189eb3f863dc75855f0cfccf16a4af188a249861377c69216f40b1`.
The custom Thai recognizer passed its synthetic-only acceptance boundary and
remains available to explicit custom/adaptive profiles, with inference-tree
SHA-256
`0876e624221bf0ff2d888506c7b9fa84eacc99f98394b424e81c098769d91e73`.
It is not a claim of real-world Thai benchmark quality.

The selected fresh LayoutXLM checkpoint is
`D:\CSX4201\vision-info-extraction-assets\checkpoints\layoutxlm_multitask\ocr_upgrade_fresh_b_noise`
with `model.safetensors` SHA-256
`f257538849bd2067a9df9df83385aa10ae468d0499510fb0621a03a5f0155180`.
The bound calibration SHA-256 is
`81a55061554d760e42c492d16f283a78fea32c64fd697947cbeeb8c7c1e9fc44`.

Training-data counts, rejected trials, adaptive DPI, tiling, padding,
recognition retries, orientation selection, and exact reproduction commands
are recorded in
[`OCR_UPGRADE_RELEASE_NOTES.md`](OCR_UPGRADE_RELEASE_NOTES.md).

## CPU mode

Pass `--device cpu` to OCR verification/inference and `--device cpu` to the
layout training script where supported. CPU mode is slower. Do not install a
CUDA Torch wheel into the OCR environment; doing so reintroduces the DLL
collision.

## Troubleshooting

- Storage gate failure: free space or change the configured external root;
  never bypass the 15 GiB reserve.
- Paddle/Torch DLL error: verify that the command uses the correct interpreter
  and that the layout worker is a subprocess.
- Missing model/hash mismatch: rerun `download_ocr_models.py`; do not copy an
  unverified partial cache into place.
- Direct training verifier cannot find `cublasLt64_13.dll`: confirm the command
  uses `ie-ocr-train\Scripts\python.exe`. The verifier should record both
  environment-local NVIDIA DLL directories; do not repair this by copying DLLs
  into the repository.
- CUDA unavailable: rerun the environment report and inspect driver/runtime
  probes before falling back to CPU.
- K-Means maintenance artifacts remain in their original scikit-learn 1.8
  joblib form. Inference uses the hash-bound `inference_params.npz` numeric
  export, which was checked against all 7,520 public train/validation/test
  feature rows and avoids cross-version pickle loading in Python 3.10.
