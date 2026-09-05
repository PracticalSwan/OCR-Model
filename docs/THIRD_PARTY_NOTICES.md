# Distribution and third-party notices

> **Historical distribution notice.** The local package was removed when the
> project was archived on 2026-09-06. These upstream terms continue to apply to
> historical Release assets and any future restoration. See
> [the archive record](../ARCHIVED.md).

This portable package is intended for academic and noncommercial evaluation.
It contains original project code plus model artifacts with separate upstream
terms. No raw datasets or private Gmail documents are distributed.

## LayoutXLM checkpoint

The fine-tuned checkpoint is derived from `microsoft/layoutxlm-base`. The
project configuration records the model license as
**Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
(CC BY-NC-SA 4.0)**:

<https://creativecommons.org/licenses/by-nc-sa/4.0/>

This includes attribution, noncommercial-use, and share-alike conditions. Do
not use or redistribute the checkpoint for commercial purposes without
confirming that the required rights are available.

## PaddleOCR model artifacts

The included model cards for:

- `PP-OCRv6_medium_det`
- `PP-OCRv6_medium_rec`
- `th_PP-OCRv5_mobile_rec`

declare the **Apache License 2.0**:

<https://www.apache.org/licenses/LICENSE-2.0>

Their original `README.md` model cards are retained inside each bundled model
directory.

## Synthetic OCR fonts

The bounded general and Thai synthetic OCR corpora were rendered with Noto
Sans and Noto Sans Thai from the official Google Fonts repository at pinned
commit `9fab8b6cc7b2f20376914fd765d918c698c66d75`. Both fonts are licensed
under the SIL Open Font License 1.1:

<https://openfontlicense.org/>

The exact font and license hashes are recorded in
`reports/ocr_upgrade/synthetic_data_manifest.json`. Font binaries and
synthetic training images remain on the local D: asset volume and are not
committed to this repository. The selected custom Thai OCR artifact was
trained only on this deterministic OFL-font corpus; that evidence does not
establish accuracy on real Thai documents.

The candidate `openthaigpt/thai-ocr-evaluation` dataset declares CC BY-SA 4.0,
but its card did not provide per-sample source and license mapping for the
underlying images. It was not downloaded, trained on, evaluated on, or
redistributed by this project.

## Public OCR training datasets

- FATURA is sourced from Zenodo record 8261508, which declares CC BY 4.0.
- FUNSD's official terms restrict use to noncommercial research and
  education.
- The corrected public SROIE source repository declares MIT; underlying
  challenge and document-image rights may impose additional conditions.
- CORU is held out for unseen evaluation and contributes no fit or selection
  row.

Dataset source URLs, declared terms, and caveats are recorded in
`reports/ocr_upgrade/benchmark_summary.json`. No raw dataset is redistributed
inside the portable package.

## Runtime dependencies

The setup scripts install pinned or bounded Python dependencies from their
official package indexes. Each dependency keeps its own license. Docker,
Python, PyTorch, PaddlePaddle, PaddleOCR, Transformers, Gradio, the MCP Python
SDK, and the remaining packages are not relicensed by this project.

## Original project code

Original source code and documentation authored for this project are licensed
under the MIT License in the repository's `LICENSE` file. Trained weights,
datasets, and third-party components remain governed by their respective
upstream terms. In particular, the LayoutXLM-derived checkpoint's
CC BY-NC-SA 4.0 terms still restrict the complete weights-included package to
noncommercial use and require attribution and share-alike compliance.
