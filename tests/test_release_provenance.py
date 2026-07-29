from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from src.release_provenance import (
    git_worktree_provenance,
    require_clean_git_worktree,
)


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=root,
        stdin=subprocess.DEVNULL,
        check=True,
        capture_output=True,
    )


def test_worktree_provenance_binds_candidate_bytes_and_ignores_cache(
    tmp_path: Path,
) -> None:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.invalid")
    _git(tmp_path, "config", "user.name", "Release Test")
    (tmp_path / ".gitignore").write_text("cache/\n", encoding="utf-8")
    source = tmp_path / "source.txt"
    source.write_text("first\n", encoding="utf-8")
    _git(tmp_path, "add", ".gitignore", "source.txt")
    _git(tmp_path, "commit", "-m", "baseline")

    baseline = require_clean_git_worktree(tmp_path)
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "ignored.bin").write_bytes(b"ignored")
    ignored = require_clean_git_worktree(tmp_path)

    assert baseline["source_commit"] == ignored["source_commit"]
    assert baseline["source_tree_sha256"] == ignored["source_tree_sha256"]
    assert baseline["source_candidate_file_count"] == 2
    assert baseline["source_tree_dirty"] is False

    source.write_text("second\n", encoding="utf-8")
    modified = git_worktree_provenance(tmp_path)

    assert modified["source_tree_dirty"] is True
    assert modified["source_tree_sha256"] != baseline["source_tree_sha256"]
    with pytest.raises(RuntimeError, match="clean Git worktree"):
        require_clean_git_worktree(tmp_path)
