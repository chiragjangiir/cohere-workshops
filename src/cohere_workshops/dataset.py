"""Load and validate reproducible pull-request evaluation manifests."""

import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

FULL_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class EvaluationCase:
    """One pre-change repository case with diff-derived ground truth."""

    case_id: str
    repository_url: str
    pull_request: int
    pre_change_commit: str
    post_change_commit: str
    change_request: str
    ground_truth_files: tuple[str, ...]
    ground_truth_test_files: tuple[str, ...]


@dataclass(frozen=True)
class DatasetManifest:
    """A versioned collection of independent evaluation cases."""

    dataset_version: str
    cases: tuple[EvaluationCase, ...]


def _required_string(payload: dict[str, object], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _repository_paths(
    value: object,
    field: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ValueError(f"{field} must be a non-empty list")
    if not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{field} must contain non-empty strings")

    paths = tuple(value)
    if len(set(paths)) != len(paths):
        raise ValueError(f"{field} must not contain duplicates")

    for path in paths:
        parsed = PurePosixPath(path)
        if parsed.is_absolute() or ".." in parsed.parts or path != parsed.as_posix():
            raise ValueError(
                f"{field} must contain normalized repository-relative paths"
            )

    return paths


def _parse_case(payload: object) -> EvaluationCase:
    if not isinstance(payload, dict):
        raise TypeError("each dataset case must be an object")

    case_id = _required_string(payload, "case_id")
    repository_url = _required_string(payload, "repository_url")
    if not repository_url.startswith("https://github.com/"):
        raise ValueError("repository_url must be a public GitHub HTTPS URL")

    pull_request = payload.get("pull_request")
    if (
        not isinstance(pull_request, int)
        or isinstance(pull_request, bool)
        or pull_request < 1
    ):
        raise ValueError("pull_request must be a positive integer")

    pre_change_commit = _required_string(payload, "pre_change_commit")
    if not FULL_COMMIT_PATTERN.fullmatch(pre_change_commit):
        raise ValueError(
            "pre_change_commit must be a lowercase 40-character Git commit"
        )

    post_change_commit = _required_string(payload, "post_change_commit")
    if not FULL_COMMIT_PATTERN.fullmatch(post_change_commit):
        raise ValueError(
            "post_change_commit must be a lowercase 40-character Git commit"
        )

    change_request = _required_string(payload, "change_request")
    ground_truth_files = _repository_paths(
        payload.get("ground_truth_files"),
        "ground_truth_files",
    )
    ground_truth_test_files = _repository_paths(
        payload.get("ground_truth_test_files"),
        "ground_truth_test_files",
        allow_empty=True,
    )

    if not set(ground_truth_test_files).issubset(ground_truth_files):
        raise ValueError(
            "test ground truth must be a subset of ground_truth_files"
        )

    return EvaluationCase(
        case_id=case_id,
        repository_url=repository_url,
        pull_request=pull_request,
        pre_change_commit=pre_change_commit,
        post_change_commit=post_change_commit,
        change_request=change_request,
        ground_truth_files=ground_truth_files,
        ground_truth_test_files=ground_truth_test_files,
    )


def load_dataset_manifest(path: Path) -> DatasetManifest:
    """Load a versioned JSON manifest and reject ambiguous cases."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("dataset manifest must be a JSON object")

    dataset_version = _required_string(payload, "dataset_version")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("cases must be a non-empty list")

    cases = tuple(_parse_case(case) for case in raw_cases)
    case_ids = [case.case_id for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("dataset contains a duplicate case_id")

    repository_prs = [
        (case.repository_url, case.pull_request) for case in cases
    ]
    if len(set(repository_prs)) != len(repository_prs):
        raise ValueError("dataset contains a duplicate pull request")

    return DatasetManifest(
        dataset_version=dataset_version,
        cases=cases,
    )
