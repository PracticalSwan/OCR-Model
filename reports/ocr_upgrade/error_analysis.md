# OCR upgrade public error analysis

This report uses the one-time public TEST_IN_DOMAIN OCR evaluation for detection, recognition, and downstream signals. Orientation signals come from the public DEV_SELECT rotation diagnostic. It contains no private filenames, OCR text, images, or per-document predictions.

The priority score is `frequency per evaluated page × downstream impact × implementation feasibility`. Impact and feasibility use a declared 1–5 engineering rubric. The score ranks bounded next steps; it is not a model-quality metric.

## Ranked priorities

| Rank | Group | Category | Count | Unit | Per page | Impact | Feasibility | Priority |
|---:|---|---|---:|---|---:|---:|---:|---:|
| 1 | Detection | False positives | 49974 | region signals | 28.3943 | 3 | 4 | 340.7318 |
| 2 | Detection | Split regions | 41023 | region-count signals | 23.3085 | 3 | 3 | 209.7767 |
| 3 | Detection | Missed small text | 6876 | missed-region signals | 3.9068 | 5 | 4 | 78.1364 |
| 4 | Downstream | Field evidence found but abstained | 4406 | field signals | 2.5034 | 4 | 4 | 40.0545 |
| 5 | Detection | Merged regions | 3475 | region-count signals | 1.9744 | 4 | 3 | 23.6932 |
| 6 | Downstream | OCR text wrong and entity wrong | 1760 | pages | 1.0000 | 5 | 3 | 15.0000 |
| 7 | Recognition | Low-resolution errors | 1634 | pages | 0.9284 | 4 | 4 | 14.8545 |
| 8 | Orientation | Wrong cardinal orientation | 62 | page-angle cases | 0.3827 | 5 | 4 | 7.6543 |
| 9 | Orientation | Close candidate ambiguity | 134 | page-angle cases | 0.8272 | 3 | 3 | 7.4444 |
| 10 | Recognition | Number confusion | 280 | token signals | 0.1591 | 5 | 4 | 3.1818 |
| 11 | Downstream | Relation missed | 684 | relation signals | 0.3886 | 4 | 2 | 3.1091 |
| 12 | Recognition | Punctuation loss | 289 | character signals | 0.1642 | 3 | 4 | 1.9705 |
| 13 | Recognition | Identifier corruption | 120 | token signals | 0.0682 | 5 | 3 | 1.0227 |
| 14 | Recognition | Decimal confusion | 59 | token signals | 0.0335 | 5 | 4 | 0.6705 |
| 15 | Recognition | Spacing errors | 1 | pages | 0.0006 | 2 | 4 | 0.0045 |
| 16 | Downstream | Arithmetic inconsistency | 0 | pages | 0.0000 | 5 | 4 | 0.0000 |
| 17 | Recognition | Crop too tight | 0 | retry signals | 0.0000 | 4 | 5 | 0.0000 |
| 18 | Recognition | Language-route errors | 0 | pages | 0.0000 | 5 | 4 | 0.0000 |
| 19 | Detection | Low-contrast misses | 0 | pages | 0.0000 | 4 | 4 | 0.0000 |
| 20 | Orientation | Multi-orientation page | 0 | page-angle cases | 0.0000 | 5 | 2 | 0.0000 |
| 21 | Downstream | OCR text correct but entity wrong | 0 | pages | 0.0000 | 5 | 3 | 0.0000 |
| 22 | Detection | Rotated misses | 0 | pages | 0.0000 | 5 | 3 | 0.0000 |
| 23 | Downstream | Table grouping failure | 0 | pages | 0.0000 | 5 | 3 | 0.0000 |
| 24 | Detection | Table misses | 0 | pages | 0.0000 | 5 | 3 | 0.0000 |
| 25 | Recognition | Turkish-character errors | 0 | character signals | 0.0000 | 4 | 4 | 0.0000 |
| 26 | Orientation | Wrong deskew | 0 | page-angle cases | 0.0000 | 4 | 3 | 0.0000 |

## Measurement boundaries

- Missed-small-text, false-positive, number, punctuation, identifier, field, and relation values are occurrence signals and can exceed the evaluated page count.
- Merged and split region values are deterministic region-count imbalance signals, not manually labeled merge/split events.
- Low-contrast, low-resolution, rotation, arithmetic, and table values are reproducible page-level diagnostics from the executed pipeline.
- A zero means the bounded diagnostic did not observe that signal. It does not prove the failure mode is impossible.
- TEST_IN_DOMAIN was evaluated only after the OCR and LayoutXLM stack was frozen. CORU and private outputs were not used for selection or subsequent tuning.
