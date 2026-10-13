"""Derive evaluation ground truth from committed Git history."""

import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from cohere_workshops.dataset import FULL_COMMIT_PATTERN


@dataclass(frozen=True)
class DiffGroundTruth:
    """Changed files derived from an exact Git commit range."""

    pre_change_commit: str
    post_change_commit: str
    files: tuple[str, ...]
    test_files: tuple[str, ...]


def _git(repository: Path, *arguments: str) -> str:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError as error:
        message = error.stderr.strip() or "Git command failed"
        raise ValueError(message) from error


def _verify_commit(repository: Path, commit: str, field: str) -> str:
    if not FULL_COMMIT_PATTERN.fullmatch(commit):
        raise ValueError(f"{field} must be a lowercase 40-character Git commit")

    resolved = _git(repository, "rev-parse", "--verify", f"{commit}^{{commit}}")
    if resolved != commit:
        raise ValueError(f"{field} did not resolve to the requested commit")
    return resolved


def is_test_path(path: str) -> bool:
    """Identify conventional Python test paths deterministically."""
    parsed = PurePosixPath(path)
    name = parsed.name
    return (
        "tests" in parsed.parts
        or (name.startswith("test_") and name.endswith(".py"))
        or name.endswith("_test.py")
    )


def extract_diff_ground_truth(
    repository: Path,
    pre_change_commit: str,
    post_change_commit: str,
) -> DiffGroundTruth:
    """Extract changed paths without using a language model."""
    repository = repository.resolve()
    if not repository.is_dir():
        raise ValueError(f"Repository does not exist: {repository}")
    if _git(repository, "rev-parse", "--is-inside-work-tree") != "true":
        raise ValueError(f"Not a Git repository: {repository}")

    pre_change = _verify_commit(
        repository,
        pre_change_commit,
        "pre_change_commit",
    )
    post_change = _verify_commit(
        repository,
        post_change_commit,
        "post_chane_commit",
    )
    output = _git(
        repository,
        "diff",
        "--name-only",
        "--diff-filter=ACMRD",
        pre_change,
        post_change,
        "--",
    )
    files = tuple(sorted(line for line in output.splitlines() if line))
    if not files:
        raise ValueError("commit range contains no changed files")

    return DiffGroundTruth(
        pre_change_commit=pre_change,
        post_change_commit=post_change,
        files=files,
        test_files=tuple(path for path in files if is_test_path(path)),
    )
