# OCR Model archive status

OCR Model was archived locally on 2026-09-06. The source repository is retained
at `D:\Side Projects\OCR Model` as a read-only historical and reproducibility
record. It no longer has a runnable local installation or model-asset tree on
this computer.

The public [GitHub repository](https://github.com/PracticalSwan/OCR-Model),
historical GitHub Releases, Devpost submission, and YouTube demonstration were
not deleted or changed by the local cleanup. They remain historical publication
records. This project performs extraction locally and does not require an
OpenAI API key.

## Local cleanup completed

The following project-specific runtime state was removed:

- `D:\OCR_Model` — portable Windows installation and embedded runtime;
- `D:\OCR_Model_Assets` — junction to the large asset tree;
- `D:\CSX4201\vision-info-extraction-assets` — checkpoints, OCR models,
  development environments, generated data, and release-build inputs;
- `D:\CSX4201` — removed after it was verified empty;
- the global Codex `ocr_model` MCP registration and its obsolete trusted-project
  entry; and
- four live Python MCP server processes whose command lines targeted
  `D:\OCR_Model\mcp_server.py`.

The previously used `D:\OCR_Model.zip`, checksum sidecar, and
`D:\OCR_Model_release_build` paths were already absent. The final before/after
drive snapshots measured 55,699,529,728 bytes (51.87 GiB) reclaimed.

The independent rotation-classification coursework repository was outside this
cleanup boundary and was left intact.

## Final source verification

Verification was run before the OCR development environments were removed:

- `python -m compileall -q src scripts tests` completed successfully.
- The corrected critical-field evaluation module passed both focused tests.
- The executable repository suite completed with 421 passed, 2 skipped, and
  1 deselected. The deselected test requires the declared optional `ImageHash`
  dependency, which was not installed in any remaining local environment.
- A first unfiltered run completed with 420 passed, 2 skipped, and 2 failures:
  one inconsistent anonymized test fixture (corrected before the final run) and
  the same unavailable `ImageHash` dependency.

No post-cleanup end-to-end inference was attempted: the model weights and OCR
runtime were deliberately deleted. Historical evaluation and Release evidence
remain in this repository and on GitHub, but they are not proof of a currently
installed local runtime.

## Restoring the project

To reactivate the project, start from a fresh clone and treat the setup commands
in `README.md`, `docs/ocr_setup.md`, and `docs/PORTABLE_USAGE.md` as restoration
instructions. Recreate the Python environments, obtain every model artifact
under its upstream license, rebuild the asset registry, rerun the full tests,
and verify a real image/PDF extraction before claiming the project is runnable.
The optional Codex MCP integration must also be registered again if it is
wanted.

Do not publish private documents, OCR caches, extracted text, credentials, or
personal paths while restoring or reviewing the project.
