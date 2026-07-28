# Final OCR and information-extraction model card

## Scope

This artifact is a document information-extraction pre-model for scanned forms, receipts, and invoices. It combines PaddleOCR detection and recognition with a LayoutXLM multi-task checkpoint. The four-zone K-Means output is display-only and never controls OCR or orientation.

The measured system is not production-ready and is not claimed to work accurately on every document.

## Selected stack

| Component | Selected artifact | SHA-256 | Decision |
|---|---|---|---|
| Detector | PP-OCRv6_medium_det | `eccf59cf53c201173dbabb4e45115d067414e4db8aeeb37a84e0b035afba494d` | Original retained after bounded fine-tuning attempts did not produce an eligible checkpoint. |
| General recognizer | PP-OCRv6_medium_rec | `6c46447e05189eb3f863dc75855f0cfccf16a4af188a249861377c69216f40b1` | Original retained by the declared development gate. |
| Thai recognizer | th_PP-OCRv5_mobile_rec | `0876e624221bf0ff2d888506c7b9fa84eacc99f98394b424e81c098769d91e73` | Custom synthetic-data model selected within its stated evidence boundary. |
| OCR profile | A (original) | `ce044d8893f214eebe1afd99327c7775a1cef4da9f83fe263050de3ef3975297` | Selected on public DEV_SELECT only. |
| LayoutXLM | fresh_b_noise | `f257538849bd2067a9df9df83385aa10ae468d0499510fb0621a03a5f0155180` | Best eligible downstream trial under the declared regression gates. |

Registry defaults: detector `original`, general recognizer `original`, Thai recognizer `custom`.

## Data and selection boundaries

- OCR model comparison used 400 public DEV_SELECT pages: fatura=283, funsd=20, sroie=97.
- Detector and recognizer fitting used public TRAIN only. DEV_SELECT selected models and adaptive components.
- DEV_CALIBRATION was reserved for confidence calibration.
- TEST_IN_DOMAIN, CORU, and Gmail/private documents contributed zero fit or selection rows.
- The Thai custom recognizer was trained and selected using synthetic OFL-font evidence. No real labeled public Thai accuracy benchmark was available, so its claim boundary remains synthetic-data performance plus integration behavior.
- Dataset and font licenses are recorded in the benchmark report, training-data report, third-party notices, and model registry.

## Adaptive OCR behavior

- Preprocessing: `grayscale_normalized`.
- Adaptive PDF rerender rate: 0.00%.
- Tiling metric deltas: cer=0.0000, critical_field_exact_match=0.0000, critical_region_recall=0.0000, end_to_end_canonical_accuracy=0.0000, end_to_end_entity_f1=0.0000, end_to_end_relation_f1=0.0000, page_failure_rate=0.0000, polygon_f1=0.0000, recognized_text_coverage=0.0000, small_text_recall=0.0000, tiling_duplicate_rate=0.0000, time_per_page_seconds=1.1859, wer=0.0000.
- Crop-padding profile: `C`.
- Recognition retry metric deltas: cer=0.0004, critical_field_exact_match=0.0000, critical_region_recall=0.0000, end_to_end_canonical_accuracy=0.0000, end_to_end_entity_f1=0.0002, end_to_end_relation_f1=-0.0000, page_failure_rate=0.0000, polygon_f1=0.0003, recognized_text_coverage=-0.0004, small_text_recall=0.0002, tiling_duplicate_rate=0.0000, time_per_page_seconds=0.8085, wer=0.0010.
- Orientation selection accuracy: 68.52%; K-Means instantiated: no.

## Locked public evaluation

| Metric | Result |
|---|---:|
| TEST pages | 1760 |
| OCR polygon F1 | 0.3815 |
| OCR recognized-text coverage | 0.1663 |
| OCR WER | 0.9692 |
| Critical-field exact match | 0.3496 |
| End-to-end entity F1 | 0.0944 |
| End-to-end relation F1 | 0.0111 |
| End-to-end canonical-field accuracy | 0.2534 |
| Calibration ECE after (entity / canonical / relation / document) | 0.0013 / 0.0066 / 0.0059 / 0.0000 |

## Fallback, cache, and portability

- The original detector and recognizers remain registered as explicit fallbacks.
- OCR caches are bound to model, preprocessing, and configuration hashes; mismatched entries are rejected.
- Portable packages contain selected inference artifacts and allowed fallbacks only. Training data, crops, caches, environments, logs, and private material are excluded.

## Known limitations

- OCR remains the main end-to-end bottleneck, especially for small, low-contrast, rotated, and tightly cropped text.
- Relation supervision is limited to FUNSD, so relation results do not generalize across every document family.
- CORU lacks compatible token polygons and is reported as an unseen operational QA/text-recall evaluation, not a fitting source.
- Thai accuracy is not established on a real labeled public benchmark.
- K-Means zones are diagnostic display output. The failed exact-angle estimator remains disabled for inference.
