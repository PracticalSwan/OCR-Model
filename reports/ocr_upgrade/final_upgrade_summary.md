# Final OCR upgrade summary

The upgrade completed a public-only OCR training and selection lifecycle, rebuilt OCR-realistic LayoutXLM streams, evaluated bounded downstream adaptation, recalibrated the frozen stack, and reserved locked TEST_IN_DOMAIN, unseen CORU, and private Gmail operation for their declared final stages.

## Decisions

- Selected OCR configuration: `A` (`original`).
- Selected preprocessing: `grayscale_normalized`.
- Selected LayoutXLM trial: `fresh_b_noise` with checkpoint `f257538849bd2067a9df9df83385aa10ae468d0499510fb0621a03a5f0155180`.
- Selected stream manifest: `C:\Assumption University\CSX4201\Project\data\metadata\final_model_dataset_manifest_ocr_v2_b_noise.csv` with 12455 TRAIN and 1913 DEV_SELECT examples.
- Original OCR remains available as a fallback. The custom Thai recognizer remains bounded to synthetic Thai selection evidence.

## Measured results

| Evidence | Baseline | Final |
|---|---:|---:|
| Upright OCR polygon F1 | 0.3395 | 0.3815 |
| Upright OCR recognized-text coverage | 0.4028 | 0.1663 |
| Upright OCR WER | 0.6999 | 0.9692 |
| End-to-end entity F1 | 0.1772 | 0.0944 |
| End-to-end relation F1 | 0.0075 | 0.0111 |
| End-to-end canonical-field accuracy | 0.4444 | 0.2534 |
| Unseen CORU QA answer-text recall | 0.7853 | 0.7853 |

The baseline upright OCR and end-to-end values used three pages, while the final locked result uses the full frozen TEST_IN_DOMAIN sample. The table preserves both executed results but does not treat that difference as a controlled paired experiment.

## Privacy and evaluation boundaries

- Locked public failures: 0 of 1760.
- CORU: 100 successful and 0 failed pages from a fixed 100-page sample.
- Private operation: 2 successful and 0 failed documents; published output is aggregate-only.
- Private/Gmail fit rows: 0.
- No decision or configuration change was made from TEST_IN_DOMAIN, CORU, or private output.

## Remaining limitations

- End-to-end extraction remains bounded by OCR quality.
- Detector fine-tuning was stopped for documented memory, numerical, and runtime constraints; the stronger eligible original detector was retained.
- The general custom recognizer improved several development metrics but missed the declared WER gate, so it was not made the default.
- Thai evidence is synthetic and does not establish real-world Thai accuracy.
- Physical Apple hardware remains untested. Docker Linux/AMD64 is the supported macOS route.

## Reproduction entry points

```powershell
python scripts/compile_ocr_upgrade_reports.py
python -m pytest -q
python -m compileall -q src scripts tests
D:\CSX4201\vision-info-extraction-assets\environments\ie-ocr\Scripts\python.exe scripts\verify_information_extraction.py --complete
```

The exact training, evaluation, inference, portable-build, and privacy commands are recorded in the repository documentation and executed reports. Reported hashes are authoritative for artifact identity.
