# AGENTS.md — OCR Model project instructions

## Project identity

- **Project:** OCR Model: Local Document Intelligence.
- **Workspace:** `D:\Side Projects\OCR Model`.
- **Type:** independent personal/open-source side project.
- **Purpose:** local document OCR and structured information extraction from images and PDFs.
- **Primary stack:** PaddleOCR + LayoutXLM + deterministic evidence/rule validation.
- **Historical large asset alias:** `D:\OCR_Model_Assets` (removed during archival cleanup).
- Do not infer or add academic, classroom, instructor, grading, or university requirements to this repository.

## Archived state

- The project was archived locally on 2026-09-06. See `ARCHIVED.md`.
- Preserve this repository as source and historical evidence. Do not recreate
  deleted runtimes, model assets, generated data, or MCP configuration unless
  the user explicitly reopens the project.
- Commands and local paths in technical documents are historical/restoration
  instructions and are not evidence that those environments still exist.
- The public GitHub repository, Releases, Devpost entry, and video are external
  historical records; do not delete or alter them without separate explicit
  authorization.

## Current product boundary

- The pipeline performs local OCR, entity extraction, document-type prediction, canonical-field extraction, relations, evidence checks, table geometry, and JSON Schema validation.
- The preserved PCA/K-Means rotation experiment is diagnostic/display-only and must not control OCR or extraction.
- Calibrated LayoutXLM inference is fail-closed when its checkpoint, calibration, OCR-stack binding, or required worker is invalid.
- Generic/rule fallback must remain explicit and must not be described as calibrated LayoutXLM inference.
- Current end-to-end quality is measured and limited; do not describe the system as production-ready or universally accurate.

## Session-start protocol

1. Read `AGENT_MEMORY.md` for orientation, then verify relevant facts against the live workspace.
2. Read `LESSONS.md` before changing code or data workflows.
3. Inspect Git status and the nearest relevant source, tests, config, and docs before editing.
4. Keep changes minimal and avoid unrelated refactors or dependency upgrades.

## Data and privacy

- `data/raw/` and private document inputs are local data, not normal Git content.
- Never publish private documents, private OCR text, private predictions, credentials, local runtime configuration, or user outputs.
- Prefer synthetic or clearly redistributable public samples in tests and documentation.
- Treat generated reports as evidence from a particular execution; do not silently rewrite historical metrics to match a newer run.
- Historical reports may contain old machine-local paths. They are provenance records, not current project configuration.

## Verification

- Prefer executable evidence: targeted tests, compilation, model/runtime probes, schema checks, and end-to-end samples where appropriate.
- Do not claim a model, package, release, or workflow works without relevant verification.
- For broad Python changes, run the strongest practical subset of `pytest` plus `python -m compileall -q src scripts tests`.

## Git and release hygiene

- Before commit or push, inspect status and intended diff; stage only intended files.
- Keep large checkpoints, environments, caches, and generated user data out of normal Git history unless an established release process explicitly packages them.
- Preserve third-party license notices and checkpoint provenance.
- External publication, GitHub repository changes, release replacement, or history rewriting require explicit user scope.

## Architecture guardrails

- Reuse existing OCR, layout-worker, schema, privacy, and packaging abstractions before adding new ones.
- Keep Paddle and CUDA PyTorch runtimes isolated where required by their Windows library compatibility.
- Keep private-mode output surfaces opaque and non-shareable by default.
- Preserve the distinction between reference-token model metrics and full image-to-JSON end-to-end metrics.

## Definition of done

The requested scope is implemented, relevant verification has been executed, material limitations are stated, and unrelated user work remains untouched.
