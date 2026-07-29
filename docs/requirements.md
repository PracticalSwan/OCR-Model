# Information-Extraction Requirements

Acceptance criteria and executed evidence for the final working academic
pre-model. Passing a functional requirement does not imply production-quality
accuracy; measured quality is reported separately.

## Scope

The controlling result is structured extraction from raster images and
single/multipage PDFs, including rotated, multilingual, and unfamiliar
documents. The preserved four-cluster K-Means model is a display-only
diagnostic. Its cluster, zone, confidence, and failed exact-angle experiment
cannot control or block OCR/extraction.

## Functional requirements

| ID | Requirement | Final validated state |
|---|---|---|
| IE-001 | Accept supported raster images and single/multipage PDFs. | Pass in unit tests and executable image/PDF integration; PNG/JPEG/TIFF/BMP/WebP paths covered. |
| IE-002 | Normalize EXIF, grayscale, RGB/RGBA, transparent, and CMYK inputs while preserving geometry. | Pass; transparency flattens on white and geometry transforms are tested. |
| IE-003 | Use the exact `PP-OCRv6_medium_det` artifact. | Pass; inventory SHA-256 and GPU initialization verified. |
| IE-004 | Use exact `PP-OCRv6_medium_rec` for the general route. | Pass; model identity and known-phrase recovery verified. |
| IE-005 | Use exact `th_PP-OCRv5_mobile_rec` for Thai. | Pass; Thai Unicode and mixed-language multipage routing verified. |
| IE-006 | Select OCR correction independently of K-Means. | Pass; cardinal candidates plus reliable automatic/supplied fine deskew use OCR evidence only. |
| IE-007 | Rotate pixels, polygons, boxes, entities, and relations with one transform. | Pass in homogeneous-transform, clipping, and arbitrary-angle tests. |
| IE-008 | Produce learned entities, document type, typed relations, canonical evidence, fields, and tables. | Pass functionally with calibrated abstention; quality limitations remain explicit. |
| IE-009 | Preserve useful generic output for unfamiliar types. | Pass; OCR, entities, generic key/value pairs, and schema-valid unknown type remain available. |
| IE-010 | Validate every result against a versioned JSON Schema before write. | Pass; unsupported/conflicted fields are explicit `null`. |
| IE-011 | Keep multipage geometry/output isolated by page. | Pass, including continued output after a configured page-level failure. |
| IE-012 | Make K-Means display-only and failure-isolated. | Pass; disabled/missing/wrong artifacts cannot block extraction. |
| IE-013 | Fail fast when a required final checkpoint/calibration is missing, mismatched, bound to another OCR stack, or fails at runtime. | Pass in CLI/worker regressions; generic/rule-only fallback requires an explicit opt-in and is never labeled calibrated LayoutXLM output. |
| IE-014 | Provide an explicit portable private-document mode. | Pass; GUI-private is the default and disables preview, uses opaque private-root output and worker input, removes session upload/input caches, redacts paths/names, provides no visualization/archive, and blocks Gradio access. |
| IE-015 | Distinguish learned field supervision from the output contract. | Pass; 14 configured fields have direct model supervision and the schema supports 27 fields after learned, rule, and hybrid resolution. |

## Data, training, and evaluation requirements

| ID | Requirement | Final validated state |
|---|---|---|
| DT-001 | Treat `data/raw` as read-only. | Pass; raw verifier remains unchanged. |
| DT-002 | Normalize supported FATURA/SROIE/FUNSD/CORU records without fabricating labels. | Pass: 12,433 public records; exclusions/source defects categorized. |
| DT-003 | Use document/duplicate-safe split identities. | Pass: 29,886 identities, zero cross-split violations. |
| DT-004 | Reserve CORU as wholly unseen domain. | Pass: all 1,261 pages are `unseen_domain_test`. |
| DT-005 | Keep Gmail private-test only. | Pass: zero private/Gmail rows in model data, training, calibration, selection, and public evaluation. |
| DT-006 | Build a full public labeled final profile with multiple OCR streams. | Pass: 16,781 examples, including 11,172 ground-truth, 2,038 PaddleOCR, 2,038 hybrid, and 1,533 train-only OCR-noise examples. |
| DT-007 | Apply continuous rotations with aligned targets. | Pass: final training uses 60% upright/40% arbitrary-angle geometry. |
| DT-008 | Train real entity/document/canonical/relation targets. | Pass: 783,680 entity tokens, 216,063 canonical tokens, 78,095 relation pairs, 10,535 positives. |
| DT-009 | Select checkpoints without private or test data. | Pass: five bounded adaptation candidates; selection uses public DEV_SELECT upright/37° plus predetermined end-to-end gates only. |
| DT-010 | Calibrate without private or test data and bind the result. | Pass: 653 public DEV_CALIBRATION examples; exact checkpoint/build/manifest/OCR-stack hashes. |
| DT-011 | Save/reload model, tokenizer, labels, heads, and resumable state. | Pass; checkpoint reload maximum difference 0.0; final resume state retained on D:. |
| DT-012 | Execute a locked in-domain test once, without tuning from it. | Pass: 1,760 public ground-truth examples; report is hash-bound. |

## Safety and reproducibility requirements

| ID | Requirement | Final validated state |
|---|---|---|
| SF-001 | Preserve the stage-specific free-space reserve: 10 GiB for the bounded rotation materializer and 15 GiB for OCR/model setup, training, and portable setup. | Pass; the rotation key is explicit and backward-compatible, and after fresh portable setup C: had 23.71 GiB and D: 391.685 GiB free. |
| SF-002 | Keep large assets below the configured D: root. | Pass for environments, caches, examples, checkpoint, generated and private output. |
| SF-003 | Detect incomplete/hash-mismatched OCR artifacts. | Pass in registry, downloader, verifier, and tests. |
| SF-004 | Isolate Paddle CUDA from CUDA PyTorch on Windows. | Pass with persistent subprocess inference and separate environment partitions. |
| SF-005 | Bind public OCR caches to source/model/profile/route/transform provenance and exclude private data. | Pass in cache and privacy tests. |
| SF-006 | Return actionable input/model/storage/protocol errors without fabricated output. | Pass for missing, corrupt, encrypted, oversized, invalid-checkpoint, and worker failures. |
| SF-007 | Constrain detailed private outputs to ignored private roots. | Pass; public path rejection, anonymous IDs, no public visualization, aggregate-only report. |
| SF-008 | Preserve historical rotation evidence. | Historical pass: the frozen July 28 report is 20/20. A July 30 rerun is 18/20 because verified cleanup removed 203 derived private page renders, so 812 retained private rotations cannot re-hash their source render; all public rows and all rotation files remain present. |
| SF-009 | Make integration evidence executable and tamper-evident. | Pass: 11 source/model/config/checkpoint/fixture/output artifacts independently re-hashed and semantically checked. |
| SF-010 | Reject large/unexpected/publication-risk Git candidates. | Pass in the complete IE verifier; final staged audit is still mandatory before push. |
| SF-011 | Build portable releases from an exact clean source candidate tree without modifying an installed working copy. | Pass in builder/provenance tests; only marked isolated D: staging is allowed, installed-target and reparse-point copies are rejected, source/sample/payload hashes are bound, all payload bytes are privacy-scanned, and every ZIP entry must match the frozen payload manifest before sidecar creation. |

## Measured quality

### Locked reference-token layout test

On 1,760 public `test_in_domain` ground-truth examples (1,761 windows):

| Head | Raw micro-F1 | Calibrated/abstained micro-F1 | Raw macro-F1 |
|---|---:|---:|---:|
| Entity | 0.9827 | 0.9835 | 0.7718 |
| Canonical evidence | 0.9795 | 0.9860 | 0.9736 |
| Relation | 0.5726 | 0.5603 | 0.6887 |

Document accuracy and calibrated coverage/selective accuracy are 1.0. The
micro/macro gap and per-dataset slices expose class and dataset imbalance;
B-HEADER and QUESTION_ANSWER are the weakest supported entity/relation
classes.

### Locked image-to-JSON test

The one-time full `TEST_IN_DOMAIN` run processed 1,760/1,760 pages with zero
failures. Polygon F1 is 0.3815, recognized-text coverage 0.1663, WER 0.9692,
critical-field exact match 0.3496, entity F1 0.0944, relation F1 0.0111, and
canonical-field accuracy 0.2534. These results missed the requested accuracy
targets and were not used for tuning.

### Rotation robustness

The 18-angle layout-only grid uses 30 dataset-balanced test pages and rotates
reference geometry with the page. Minimum calibrated scores are entity 0.7683,
canonical 0.9640, relation 0.3358, and composite 0.7324. Minimum retention
versus upright is 96.69% entity, 99.26% canonical, 61.44% relation, and 91.86%
composite.

The bounded 18-angle end-to-end grid uses one real page from each labeled
dataset plus one synthetic Thai page at each angle. All 72 cases are nonempty.
Across real pages, OCR text coverage is 0.3068–0.3839, detection F1
0.1707–0.1941, entity F1 0.1326–0.1807, relation F1 0–0.0342, and field
accuracy 0.4444–0.5556. Synthetic Thai text/routing succeeds 18/18. This gap
shows that OCR, not only the layout heads, limits usable extraction.

### Unseen and private operation

All 100 deterministic CORU pages succeed with nonempty OCR. The model finds
78.53% of 4,001 QA answer strings and exactly matches 15.68% of 523 applicable
canonical fields. CORU lacks compatible token polygons, so entity/relation F1
is undefined.

The current OCR-upgrade private check succeeds on 2/2 anonymous Gmail
documents/pages with zero failures and aggregate-only reporting. There is no
private ground truth, so the aggregate is an operation check, not accuracy.

## Open acceptance decisions

- professor-approved canonical fields/document types and minimum thresholds;
- whether the required four zones specifically demand K-Means, a supervised
  angle model, or only quadrant display after orientation estimation;
- exact 90/180/270 boundary ownership;
- approved labeled Thai benchmark and broader OCR/domain adaptation plan;
- final deliverable format and whether/how to redistribute the
  CC-BY-NC-SA-4.0 checkpoint.
