# Final vision information-extraction pre-model card

## Model

The checkpoint is a LayoutXLM-initialized multilingual text-plus-normalized-2D-layout encoder with trained entity, document-type, canonical-evidence, and real relation heads. PaddleOCR runs in an isolated process path with exact general and Thai recognizers. The K-Means model is display-only.

Checkpoint: `D:\CSX4201\vision-info-extraction-assets\checkpoints\layoutxlm_multitask\ocr_upgrade_fresh_b_noise`.
Checkpoint model SHA-256: `f257538849bd2067a9df9df83385aa10ae468d0499510fb0621a03a5f0155180`.
Training: 4 completed epochs, 12556 optimizer steps; best epoch 4 with selection score 0.8433429260632851.
License inherited from the base checkpoint: `CC-BY-NC-SA-4.0`.

## Training data and privacy

Public fit pages: 11172; examples: 16781; Gmail/private fit rows: 0.
Private Gmail documents are operational test only and never train, calibrate, or select the model.

## Measured quality

Held-out entity micro-F1: raw 0.9826732121770259; calibrated/abstained 0.983477623768298; raw macro-F1 0.7718406421253087.
Held-out relation F1: raw 0.5725853094274147; calibrated/abstained 0.5602787456445993.
Held-out canonical evidence F1: raw 0.9795157780195866; calibrated/abstained 0.9860346782445176.
Bounded upright end-to-end OCR text coverage: 0.31315974665728363; WER: 0.8191721132897604.
CORU unseen-domain answer-text recall: 0.7853036740814796 on 100 sampled pages.

## Intended use

Local extraction from images and PDFs containing receipts, invoices, forms, and unfamiliar documents. Outputs include OCR evidence, generic entities and relations, canonical fields with abstention, and geometry-based tables.

## Limitations

This is a bounded academic pre-model, not a production or high-stakes decision system. The visual backbone is unavailable on this Windows runtime, relation supervision is sparse, Thai quality lacks a labeled public benchmark, arbitrary-angle end-to-end evaluation is bounded, and low-confidence/unsupported fields return null. Review financial and legal outputs against the source document.
