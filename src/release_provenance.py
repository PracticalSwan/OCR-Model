"""Deterministic Git working-tree provenance for verified release builds."""
from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any


def git_worktree_provenance(project_root: Path) -> dict[str, Any]:
    """Return the exact commit and candidate-file state of a Git worktree.

    The candidate digest covers tracked files and non-ignored untracked files.
    Ignored raw data, environments, caches, and private artifacts are excluded
    by Git's own ignore rules and therefore cannot silently affect a release
    source snapshot.
    """
    root = project_root.expanduser().resolve()
    source_commit = _git_text(root, "rev-parse", "HEAD")
    status = _git_text(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    listed = _git_bytes(
        root,
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "-z",
    )
    relative_paths = sorted(
        {
            value.decode("utf-8", errors="surrogateescape").replace("\\", "/")
            for value in listed.split(b"\0")
            if value
        }
    )

    digest = hashlib.sha256()
    missing_paths: list[str] = []
    for relative in relative_paths:
        encoded_path = relative.encode("utf-8", errors="surrogateescape")
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        path = root / Path(relative)
        if path.is_symlink():
            target = os.readlink(path).encode("utf-8", errors="surrogateescape")
            digest.update(b"L")
            digest.update(len(target).to_bytes(8, "big"))
            digest.update(target)
            continue
        if not path.is_file():
            digest.update(b"MISSING")
            missing_paths.append(relative)
            continue
        digest.update(b"F")
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(chunk)

    return {
        "source_commit": source_commit,
        "source_tree_dirty": bool(status),
        "source_tree_sha256": digest.hexdigest(),
        "source_candidate_file_count": len(relative_paths),
        "source_missing_candidate_paths": missing_paths,
    }


def require_clean_git_worktree(project_root: Path) -> dict[str, Any]:
    """Return provenance only when the release source is completely clean."""
    provenance = git_worktree_provenance(project_root)
    if provenance["source_tree_dirty"]:
        raise RuntimeError(
            "release build requires a clean Git worktree; commit or remove "
            "tracked/untracked source changes before building"
        )
    if provenance["source_missing_candidate_paths"]:
        raise RuntimeError(
            "release source contains missing Git candidate paths: "
            + ", ".join(provenance["source_missing_candidate_paths"])
        )
    return provenance


def _git_text(root: Path, *args: str) -> str:
    return _git_bytes(root, *args).decode("utf-8", errors="replace").strip()


def _git_bytes(root: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"git {' '.join(args)} failed with exit code "
            f"{completed.returncode}: {stderr}"
        )
    return completed.stdout
