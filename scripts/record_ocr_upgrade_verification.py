#!/usr/bin/env python3
"""Upsert one executed OCR-upgrade verification command into its final ledger."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.verify_information_extraction import FINAL_EXECUTION_CHECKS  # noqa: E402
from src.release_provenance import git_worktree_provenance  # noqa: E402
from src.rotation_common import atomic_write_json, sha256_file  # noqa: E402

RESERVED_HASH_KEYS = {
    "evidence_sha256",
    "source_commit",
    "source_tree_dirty_at_record_start",
    "source_tree_sha256",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ledger",
        default=str(
            PROJECT_ROOT
            / "reports"
            / "ocr_upgrade"
            / "verification_executions.json"
        ),
    )
    parser.add_argument("--name", choices=FINAL_EXECUTION_CHECKS, required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--status", choices=("passed", "failed"), required=True)
    parser.add_argument("--evidence-path", required=True)
    parser.add_argument(
        "--hash",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Additional relevant hash or immutable identifier.",
    )
    parser.add_argument(
        "--detail-json",
        default="{}",
        help="Compact JSON object with counts or result details.",
    )
    args = parser.parse_args()

    ledger_path = _resolve(args.ledger)
    evidence_path = _resolve(args.evidence_path)
    if not evidence_path.exists():
        raise FileNotFoundError(evidence_path)
    command = str(args.command).strip()
    if not command:
        parser.error("--command must not be empty")
    try:
        detail = json.loads(args.detail_json)
    except json.JSONDecodeError as exc:
        parser.error(f"--detail-json is invalid: {exc}")
    if not isinstance(detail, dict):
        parser.error("--detail-json must decode to an object")

    provenance = git_worktree_provenance(PROJECT_ROOT)
    hashes = {
        "source_commit": provenance["source_commit"],
        "source_tree_sha256": provenance["source_tree_sha256"],
        "source_tree_dirty_at_record_start": str(
            provenance["source_tree_dirty"]
        ).lower(),
    }
    if evidence_path.is_file():
        hashes["evidence_sha256"] = sha256_file(evidence_path)
    if evidence_path.name == "portable_verification.json":
        evidence_payload = json.loads(evidence_path.read_text(encoding="utf-8"))
        portable_provenance = {
            "source_commit": evidence_payload.get("source_commit"),
            "source_tree_sha256": evidence_payload.get(
                "source_tree_sha256"
            ),
            "source_candidate_file_count": evidence_payload.get(
                "source_candidate_file_count"
            ),
            "source_tree_dirty_at_build": evidence_payload.get(
                "source_tree_dirty_at_build"
            ),
        }
        if (
            any(
                value in (None, "")
                for key, value in portable_provenance.items()
                if key != "source_tree_dirty_at_build"
            )
            or portable_provenance["source_tree_dirty_at_build"] is not False
        ):
            raise ValueError(
                "portable verification lacks clean generation provenance"
            )
    try:
        additional_hashes = _parse_additional_hashes(
            args.hash,
            reserved_keys=set(hashes) | RESERVED_HASH_KEYS,
        )
    except ValueError as exc:
        parser.error(str(exc))
    for key, raw in additional_hashes.items():
        hashes[key] = raw

    payload = _load_ledger(ledger_path)
    records = [
        record
        for record in payload["checks"]
        if record.get("name") != args.name
    ]
    records.append(
        {
            "name": args.name,
            "command": command,
            "status": args.status,
            "evidence_path": _portable_path(evidence_path),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "relevant_hashes": hashes,
            "detail": detail,
        }
    )
    records.sort(key=lambda record: FINAL_EXECUTION_CHECKS.index(record["name"]))
    payload.update(
        {
            "schema_version": "1.0",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_commit_at_write_start": provenance["source_commit"],
            "source_tree_dirty_at_write_start": provenance[
                "source_tree_dirty"
            ],
            "source_tree_sha256_at_write_start": provenance[
                "source_tree_sha256"
            ],
            "source_candidate_file_count_at_write_start": provenance[
                "source_candidate_file_count"
            ],
            "checks": records,
        }
    )
    payload.pop("portable_generation", None)
    atomic_write_json(ledger_path, payload)
    print(
        json.dumps(
            {
                "ledger": str(ledger_path),
                "name": args.name,
                "status": args.status,
                "record_count": len(records),
            },
            indent=2,
        )
    )
    return 0


def _parse_additional_hashes(
    values: list[str],
    *,
    reserved_keys: set[str],
) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        key, separator, raw = value.partition("=")
        key = key.strip()
        raw = raw.strip()
        if not separator or not key or not raw:
            raise ValueError("--hash values must use nonempty NAME=VALUE")
        if key in reserved_keys:
            raise ValueError(
                f"--hash cannot override recorder-controlled key: {key}"
            )
        if key in parsed:
            raise ValueError(f"--hash key was provided more than once: {key}")
        parsed[key] = raw
    return parsed


def _load_ledger(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"schema_version": "1.0", "checks": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("checks"), list):
        raise ValueError("verification ledger must contain a checks list")
    names = [str(record.get("name", "")) for record in payload["checks"]]
    if len(names) != len(set(names)):
        raise ValueError("verification ledger contains duplicate check names")
    if any(name not in FINAL_EXECUTION_CHECKS for name in names):
        raise ValueError("verification ledger contains an unknown check name")
    return payload


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _portable_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
