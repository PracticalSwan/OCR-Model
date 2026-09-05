# AGENT_MEMORY.md — OCR Model working memory

> Orientation only. Verify live paths, Git state, model files, and metrics before relying on them.

## Archive state — 2026-09-06

- The source repository remains at `D:\Side Projects\OCR Model`.
- `D:\OCR_Model`, `D:\OCR_Model_Assets`, and
  `D:\CSX4201\vision-info-extraction-assets` were deleted; the empty
  `D:\CSX4201` parent was then removed.
- The global Codex `ocr_model` MCP registration, its obsolete trust entry, and
  four matching server processes were removed.
- The final cleanup snapshots measured 55,699,529,728 bytes (51.87 GiB)
  reclaimed.
- Pre-clean source verification: compileall passed; 421 tests passed, 2 skipped,
  and 1 dependency-gated `ImageHash` test was deselected. No post-clean model
  inference is possible without restoring weights and environments.
- GitHub, historical Releases, Devpost, and YouTube remain external publication
  records and were not removed during local cleanup.

## Project scope

- `D:\Side Projects\OCR Model` is an independent local-document-intelligence side project.
- Do not use it as a workspace for unrelated model-training exercises.
- Historical OCR/layout assets were referenced through `D:\OCR_Model_Assets`;
  that junction and its target are no longer present.
- The project extracts structured information from local images and PDFs without requiring an OpenAI API key.

## Current architecture

- PaddleOCR handles text detection/recognition with general and Thai routes.
- A fine-tuned LayoutXLM model handles entity, document-type, canonical-evidence, and relation heads.
- Output is validated against a versioned JSON Schema and augmented with evidence/rule checks.
- A preserved PCA/K-Means rotation-quadrant branch is display-only.
- Private-document mode uses opaque outputs, suppresses preview/share artifacts, and redacts source names and paths.

## Verified model snapshot

- Selected LayoutXLM checkpoint SHA-256: `f257538849bd2067a9df9df83385aa10ae468d0499510fb0621a03a5f0155180`.
- Public locked image-to-JSON evaluation previously measured polygon F1 0.3815, text coverage 0.1663, WER 0.9692, entity F1 0.0944, relation F1 0.0111, and canonical accuracy 0.2534.
- Those end-to-end results are substantially weaker than reference-token layout-head metrics; keep the distinction explicit.
- The custom Thai recognizer is experimental/synthetic-selected and is not a blanket replacement for the original calibrated OCR stack.
- The preserved K-Means rotation baseline remains weak and diagnostic only.

## Runtime and packaging

- Windows OCR and CUDA layout inference use separate Python 3.10 runtimes because of binary-library compatibility.
- Portable Windows and Docker-backed CPU packaging exist; host-specific macOS behavior was not validated on physical Apple hardware in the recorded evidence.
- The current source checkout may rely on large local assets that are intentionally outside normal Git storage.
- Release provenance, hashes, and historical reports are execution-specific and must be rechecked before publication or replacement.

## Working rules

- Preserve private-data boundaries and never publish real private documents or derived private text/predictions.
- Prefer measured evidence over claims of accuracy.
- Do not alter historical reports merely to make old paths or metrics look current.
- Use synthetic or redistributable public fixtures for tests.
- Keep the active project identity, configuration, and documentation independent from unrelated workspaces.

## Historical evidence

Older generated reports can retain machine-local paths and experiment labels from the environment in which they were produced. Treat those as immutable provenance, not as current workspace identity or configuration. New work must use the current independent project paths and terminology.
