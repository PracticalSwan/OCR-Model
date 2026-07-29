# Report lifecycle and authority

Reports are retained as evidence of distinct project stages. A newer report
does not rewrite an older experiment, and historical files are not deleted or
renamed merely because the selected model changed.

Use these sources for the current OCR-upgrade state:

- `ocr_upgrade/` contains the July 24–29 public-only OCR upgrade, model
  selection, selected `fresh_b_noise` LayoutXLM checkpoint, locked evaluation,
  CORU operation, private aggregate, and verification evidence.
- `ocr_upgrade/final_upgrade_summary.md` and
  `ocr_upgrade/final_ocr_model_card.md` are the reader-facing summaries.
- `ocr_upgrade/verification_executions.json` is the executed command ledger;
  `ocr_upgrade/verification.json` and
  `verification/information_extraction_verification.json` are complete
  verifier outputs.
- `final_model/` contains the selected final dataset, training, calibration,
  and evaluation reports consumed by the complete verifier. Files whose names
  contain `development` or `smoke` are retained historical trial evidence.

Release evidence is generation-specific. In particular,
`ocr_upgrade/portable_verification.json` is authoritative only when its source
commit/tree provenance, `BUILD_INFO.json`, package manifest, archive digest,
sidecar, Git tag, and live Release asset all agree. Rebuilding or replacing an
asset supersedes the prior portable-verification record without changing the
historical model/evaluation reports.

The complete execution verifier does not accept a portable report merely
because it is internally self-consistent. Its commit, candidate-tree SHA-256,
candidate count, and clean-build flag must agree between the independently
selected installed `BUILD_INFO.json` and the portable report. Each ledger row
is bound to its evidence file by a recomputed SHA-256, so these generation
fields are not duplicated in the ledger. The report does not select its own
package path: complete mode independently anchors to `D:\OCR_Model` (or an
explicit verifier override) and the current release tag commit.

The frozen July 28 rotation report is also historical evidence. It recorded
20/20 when all bounded derived sources existed. The July 30 rerun reaches
18/20 because verified cleanup had already removed 203 derived private page
renders; therefore 812 retained private rotations cannot re-hash those source
renders. The 8,332 rotations and all 7,520 public rows remain present. Do not
rewrite the historical PASS report or regenerate private derived material
merely to make that historical check reproducible.

The following directories are historical or auxiliary:

- `model_evaluation/` records the July 15 smoke-model evaluation.
- `angle_estimation/`, `feature_analysis/`, `kmeans_evaluation/`,
  `rotation/`, and `rotation_preparation/` record the July 13 rotation
  baseline. K-Means remains display-only.
- `information_extraction/` and `ocr/` include earlier integration and model
  setup evidence still referenced by the current verifier.

When reports appear to disagree, first compare their stage, timestamp,
checkpoint hash, calibration hash, OCR stack binding, source commit/tree
provenance, split, and build ID. Do not combine metrics across those
boundaries.
