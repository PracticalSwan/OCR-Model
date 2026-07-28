# Final public error analysis

This analysis uses executed public test predictions. Counts below come from the bounded end-to-end angle grid; no private text or filenames are included.

| Root-cause signal | Count |
|---|---:|
| entity f1 below 0 70 with reference | 54 |
| canonical field miss | 36 |
| detection f1 below 0 70 | 36 |
| ocr coverage below 0 70 | 36 |
| no table detected | 26 |
| relation f1 below 0 60 with reference | 18 |
| empty ocr | 0 |

## Weakest supported classes on the full locked layout test

| Head | Class | Support | F1 |
|---|---|---:|---:|
| entity | B-HEADER | 85 | 0.4088 |
| entity | I-HEADER | 173 | 0.5029 |
| entity | I-QUESTION | 796 | 0.6952 |
| entity | B-ANSWER | 564 | 0.7425 |
| entity | B-QUESTION | 613 | 0.7972 |
| relation | QUESTION_ANSWER | 595 | 0.4776 |
| relation | OTHER_RELATION | 183 | 0.8997 |
| canonical evidence | document_title | 730 | 0.9186 |
| canonical evidence | organization_name | 3902 | 0.9602 |
| canonical evidence | total_amount | 3038 | 0.9635 |
| canonical evidence | invoice_number | 3098 | 0.9777 |
| canonical evidence | email | 1854 | 0.9845 |

## Locked-test dataset slices

| Dataset | Examples | Entity F1 | Relation F1 | Canonical evidence F1 |
|---|---:|---:|---:|---:|
| fatura | 1584 | 1.0 | 0.0 | 0.9834768704238785 |
| funsd | 30 | 0.7558528428093646 | 0.5725853094274147 | 0.0 |
| sroie | 146 | 0.8968660968660969 | 0.0 | 0.8538011695906432 |

## Lowest bounded OCR coverage angles

| Angle | Pages | Recognized-text coverage | WER | Entity F1 |
|---:|---:|---:|---:|---:|
| 45 | 3 | 0.30682617874736107 | 0.8540305010893247 | 0.13559322033898305 |
| 135 | 3 | 0.30682617874736107 | 0.8474945533769063 | 0.14689265536723167 |
| 60 | 3 | 0.3078817733990148 | 0.8474945533769063 | 0.15909090909090912 |
| 315 | 3 | 0.3085855031667839 | 0.8453159041394336 | 0.13953488372093023 |
| 225 | 3 | 0.3089373680506685 | 0.8431372549019608 | 0.13559322033898305 |

These are bounded diagnostic signals, not all ground-truth error labels: for example, `no table detected` records output availability because the sampled annotations do not provide a compatible table benchmark. Implemented mitigations include cardinal-plus-polygon fine deskew, real PaddleOCR/hybrid training streams, class-weighted multi-task loss, calibrated abstention, arithmetic validation, and geometry table fallback. Remaining misses stay visible in the measured metrics.

Held-out entity micro-F1: 0.9826732121770259.
Held-out calibrated entity micro-F1: 0.983477623768298.
Held-out relation F1: 0.5725853094274147.
Held-out calibrated relation F1: 0.5602787456445993.
Held-out canonical evidence F1: 0.9795157780195866.
Held-out calibrated canonical evidence F1: 0.9860346782445176.

Known bottlenecks: relation labels exist only in FUNSD; the Windows runtime has no compatible Detectron2 visual backbone; CORU QA has no token polygons; and the public corpus has no compatible labeled Thai benchmark. These constraints are not treated as successes.
