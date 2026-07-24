# OCR Benchmark Selection

Build: `ocr-benchmark-8854b5f20deb6611`

Manifest SHA-256: `1786af155d1c2c3e69224c87f1885da3a62ef59617d7d57ff4d7d4bea661e3ec`

## Selection

- Total DEV_SELECT pages: 400
- FUNSD: 20 (all available)
- SROIE: 97 (all available)
- FATURA: 283 (deterministic stratified sample)
- Private rows: 0

FATURA selection gives exact source hashes first priority and then uses seeded round-robin coverage across template family, document type, quality, density, character size, resolution, table, amount, date, and identifier strata.

A deterministic subset receives a 37-degree diagnostic transform flag; the raw source image is never changed or copied into a raw directory.

## Governance

Only DEV_SELECT pages are present. This benchmark may select OCR models and inference settings but may not fit weights or calibration. DEV_CALIBRATION, TEST_IN_DOMAIN, CORU, and Gmail are absent.

## License scope

- FATURA: `CC-BY-4.0-Zenodo-8261508` — Zenodo record declares Creative Commons Attribution 4.0. Source: https://zenodo.org/records/8261508
- FUNSD: `FUNSD-EPFL-NC-RESEARCH-EDUCATION` — Official terms restrict use to non-commercial research and education. Source: https://guillaumejaume.github.io/FUNSD/work/
- SROIE: `MIT-ZZZDAVID-ICDAR2019-SROIE-REPOSITORY` — The public source repository containing the corrected dataset declares MIT; the original challenge may retain additional underlying rights. Source: https://github.com/zzzDavid/ICDAR-2019-SROIE

The manifest contains paths and hashes only. Raw images, normalized annotations, benchmark renderings, and predictions are not committed by this selection step.
