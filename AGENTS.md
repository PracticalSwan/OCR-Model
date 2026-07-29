# AGENTS.md — Project source of truth (cross-host)

> **How this file fits together with the others**
> - This is the **shared, cross-host** source of truth for the `CSX4201/Project` workspace. It is read by Codex, Claude Code, and Gemini CLI alike.
> - `CLAUDE.md` (sibling file) layers Claude Code-specific behavior on top of this one. The two **complement** each other: shared rules live here, Claude-only rules live there.
> - User-level global rules still apply (`~/.claude/CLAUDE.md`, `~/.codex/AGENTS.md`).
> - **Priority when rules conflict:** project files > global files > host defaults. For Claude Code specifically, `CLAUDE.md` > `AGENTS.md` at equal scope.

## Priority system
**MUST** > **SHOULD** > **OPTIONAL**. At equal priority, the narrower scope wins.

---

## Project identity
- **Course:** CSX4201 — Artificial Intelligence Concepts (Assumption University, Bangkok). Spring 2026.
- **Workspace:** `c:\Assumption University\CSX4201\Project`
- **Data root:** `data/raw/` (organized 2026-07-13). The original empty
  `vision_info_extraction_data/` husk was removed during verified cleanup on
  2026-07-21; canonical data remains under `data/`.
- **Domain (confirmed by professor, 2026-07-13):** vision information extraction — build a model that extracts information correctly and accurately from images, documents, and any file that contains information. The datasets on disk are scanned forms, receipts, and invoices (OCR + IE + possibly DocVQA).
- **Status:** Dataset organization and the bounded rotation baseline remain
  verified. The 2026-07-24 through 2026-07-28 OCR upgrade froze a 400-page
  public DEV_SELECT benchmark, built 8,755 detector pages/163,679 regions and
  137,886 real recognition crops, generated 48,471 licensed synthetic crops,
  and executed bounded detector, general-recognizer, and Thai-recognizer
  trials. The global default remains the original detector/general recognizer;
  the custom Thai recognizer is synthetic-only and limited to explicit
  custom/adaptive OCR experiments with explicit generic-layout fallback; only
  the original stack is bound to the shipped calibration. The selected fresh
  four-epoch LayoutXLM checkpoint
  used 12,455 public TRAIN examples and has SHA-256
  `f257538849bd2067a9df9df83385aa10ae468d0499510fb0621a03a5f0155180`;
  its public-only calibration SHA-256 is
  `81a55061554d760e42c492d16f283a78fea32c64fd697947cbeeb8c7c1e9fc44`.
  The locked 1,760-page image-to-JSON run had zero failures but measured only
  0.3815 polygon F1, 0.1663 text coverage, 0.9692 WER, 0.0944 entity F1,
  0.0111 relation F1, and 0.2534 canonical accuracy. These miss the requested
  quality targets; do not describe the system as production-ready or accurate
  on every document. The 100-page CORU run remains wholly unseen, and the
  current private operation is a two-document/page aggregate-only check with
  zero Gmail fit rows. K-Means remains display-only and the failed exact-angle
  estimator remains disabled. PR #2 merged `feat/domain-adapted-ocr` into
  `main` at `c6303f6843de9af1c7c97fde1ef6ff43e01de553` on 2026-07-29.
  The July 29 initial `v1.1.0-ocr-upgrade` archive was built from clean commit
  `fcae32edc193ff6574bf99362da0e2368d5ef464`; its historical size and digest
  remain in the release notes. For the current in-place correction, the live
  Release asset and sidecar are authoritative and must agree with the tag,
  `BUILD_INFO.json`, source candidate-tree hash, and generation-specific
  portable verification. Calibrated layout inference now fails closed on an
  OCR-stack binding mismatch or required worker error; generic/rule fallback
  is explicit. Portable private-document mode uses opaque output IDs and
  redacts source paths/names. The public `v1.0.0-build-week` package and
  Devpost submission `1102544` remain historical July 21 publication evidence.

## Project goal and model requirements (confirmed by professor, 2026-07-13)

The professor has now stated the project goal. Treat it as authoritative for downstream work; do not re-litigate it without the user.

- **Goal:** Build a **vision pre-model** that extracts information correctly and accurately from pictures, files, and anything that contains information.
- **Rotation-zone clustering (required auxiliary feature):** group each document by its rotation angle into four zones for display/diagnostics. It is included alongside the main pre-model but must not control OCR or extraction:

| Zone (cluster) | Rotation angle range | Example |
|----------------|----------------------|---------|
| 1              | 0 to 90 degrees      | 45 degrees -> Zone 1 |
| 2              | 90 to 180 degrees    |         |
| 3              | 180 to 270 degrees   |         |
| 4              | 270 to 360 degrees   |         |

- The rotation requirement has an executed baseline. It uses deterministic balanced augmentation, K-Means with four clusters, a training-only Hungarian cluster-to-zone mapping, and an experimental zone-guided exact-angle search. Its held-out quality is weak, so inference exposes only the K-Means display value and disables exact-angle correction by default. This implementation does not resolve the professor's open method questions.
- **Open sub-questions to confirm with the professor (do NOT assume answers):**
  - Angle-range boundaries: is exactly 90 / 180 / 270 degrees in the lower or upper zone? Default convention if unspecified: half-open intervals [0,90), [90,180), [180,270), [270,360).
  - Method: the professor said "clustering," but the four zones are predefined angular quadrants. Confirm whether this means K-Means (k=4) on the estimated rotation angle, or a deterministic 4-way classifier / regression head. These are different implementations.
  - Angle source: zoning requires a per-document rotation-angle estimate first (orientation estimation). Confirm the angle-estimation approach before zoning.
  - What "pre-model" denotes: interpreted as a preprocessing/precursor model (orientation handling that runs before extraction). Confirm with the professor.

## Verified workspace structure (2026-07-13)
```text
data/
  raw/                     # original public/private inputs; read-only and ignored
  metadata/                # organization plus page/split/rotation/feature manifests
  processed/
    private/page_images/   # anonymized private PDF renders; ignored
    rotated_images/full/   # 8,332 bounded full-profile rotations; ignored
    features/full/         # raw/transformed feature caches; ignored
  splits/                  # train, validation, test, private_test page manifests
models/kmeans_rotation/    # scaler, PCA, K-Means, mapping, provenance
reports/                   # preparation, features, K-Means, angles, verification
schemas/                   # versioned inference-output JSON Schema
scripts/                   # organization plus rotation-stage CLI entry points
src/                       # organization, rotation, OCR, IE, inference, evaluation
tests/                     # synthetic/regression tests; final counts must come from the current verification ledger
```
Large OCR/layout assets live below
`D:\CSX4201\vision-info-extraction-assets` in separate Python 3.10
environments. Raw totals remain 128,793 files and 35,459,126,772 bytes. The
bounded rotation run generated 8,332 rotations with 0 failures and 2,083 rows
per zone. The July 28 frozen rotation evidence passed 20/20 checks. A July 30
rerun currently reaches 18/20 because the earlier verified cleanup removed
203 derived private page renders, so 812 retained private rotations can no
longer re-hash their source render; all 8,332 rotation files and the public
7,520-row surface remain present. This is historical-evidence drift, not a
K-Means or inference change, and private renders must not be regenerated
solely to make the old verifier green. Use
`reports/ocr_upgrade/verification_executions.json` for the current host,
OCR-runtime, layout-runtime, compilation, schema, privacy, storage, cache,
profile, and portable verification evidence rather than copying an older test
count into new reports.

---

## Session-start protocol (MUST, every session)
1. **Read `AGENT_MEMORY.md`.** Treat its contents as hints written at a point in time. **Verify any fact against the live source before relying on it** (paths may have moved, files may have been added/removed). Reading it is orientation; it does **not** count as completing a task.
2. **Read `LESSONS.md`.** Do not repeat a mistake that has already been recorded there.
3. If anything in either file contradicts the current filesystem or the user's current request, trust the live evidence and the user — then update the file.

## Core workflow (MUST)
- **Classify the task first:** advisory/review, read-only investigation, local mutation, cross-workspace mutation, or document/writing. Behavior differs by type.
- **Read before edit.** Never edit a file you have not seen.
- **Loop:** Read -> Plan minimal change -> Implement -> Test/verify -> Refine only if needed.
- **Minimal diffs.** Change only what the task requires. No reformatting or "clean up" outside scope.
- **Evidence over assumption.** Prefer tests, execution, and logs over inference. State which verification level was performed (static review / local execution / test execution / live external verification / inference).
- **Surface uncertainty immediately.** Distinguish verified fact from inference in every report.
- **No implied work.** Do not claim an edit, validation, sync, or doc update happened unless it actually did.

## Data and git hygiene
- The repo is initialized on `main` with an existing public GitHub remote. Before every commit or push:
  - `.gitignore` already excludes OS junk (`__MACOSX/`, `.DS_Store`), Python build/venv artifacts, and IDE folders — keep it that way.
  - Large binaries are present (`.zip`, `.pdf`, image archives). Before committing, decide policy: **Git LFS**, gitignore + documented download steps, or commit directly. Do not blindly push hundreds of MB.
  - Derived rotations, features, private page renders, private operational manifests, and large operational manifests are local ignored outputs. Classical artifacts under `models/kmeans_rotation/` are legitimate for this stage, but verify their provenance and contents before staging.
  - Public reports contain public row-level predictions and private aggregate-only metrics. Even sanitized outputs require a final privacy review before publication.

## Privacy (CRITICAL — read before any commit or external upload)
- `data/raw/private/gmail/` contains **real personal financial and legal documents** (invoices, agreements, risk disclosures, terms, fee details). (Moved here from the original `vision_info_extraction_data/gmail_private_test/`.)
- **Default posture:** these MUST NOT be uploaded to a public GitHub repo. `.gitignore` excludes `data/raw/`, `data/**/gmail/`, and `private_file_inventory.csv` by default.
- The commitable `file_inventory.csv` anonymizes Gmail filenames (`gmail_<id>.<ext>`); real filenames live only in the gitignored `private_file_inventory.csv`.
- Before any commit/push, confirm the repo visibility. If the user explicitly needs Gmail data in the repo, require a **private** repo and explicit opt-in. Surface this risk every time a commit/share action is pending.
- Beyond this folder, watch for personal identities, names, emails, account numbers, and signatures inside any image/PDF before generating derived artifacts (annotations, cropped images, logs) that might be shared.

## Pending from user (blockers for fuller documentation)
- [x] High-level project goal — CONFIRMED 2026-07-13 by professor (vision info-extraction pre-model + rotation-zone clustering). See *Project goal and model requirements*.
- [ ] Confirm whether the implemented provisional canonical fields and document types match the professor's final required scope.
- [ ] Rotation-zone open sub-questions (boundary inclusivity; clustering vs classification; angle-estimation source; meaning of "pre-model"). The current baseline uses half-open zones, K-Means, and a zone-guided handcrafted estimator only as provisional implementation choices.
- [ ] Confirm official quality thresholds and test protocol. Current smoke and natural CORU-holdout results are not final benchmarks.
- [ ] Is `gmail_private_test` the private leaderboard set? Should derived outputs be derived from it at all?
- [x] Target deliverable — the final trained model, portable Windows/Docker-macOS package, public demo, and OpenAI Build Week submission were completed by 2026-07-21. Devpost submission `1102544` is `Submitted`.
- [x] Repo visibility for GitHub — confirmed public after the owner-authorized 2026-07-21 transition; recheck before every future upload.
- [x] README.md — updated for the final working information-extraction and portable-product lifecycle (2026-07-19), while preserving historical metrics, limitations, and open research decisions.

> When the user provides the above, update this section, `AGENT_MEMORY.md`, and then `README.md`.
