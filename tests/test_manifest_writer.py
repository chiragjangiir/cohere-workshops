from pathlib import Path

import pytest

from cohere_workshops.dataset import EvaluationCase
from cohere_workshops.manifest_writer import append_dataset_case


def make_case(case_id: str, pull_request: int) -> EvaluationCase:
    return EvaluationCase(
        case_id=case_id,
        repository_url="https://github.com/example/project.git",
        pull_request=pull_request,
        pre_change_commit="a" * 40,
        post_change_commit="b" * 40,
        change_request="Update repository scanning.",
        ground_truth_files=("src/scanner.py",),
        ground_truth_test_files=(),
    )


def test_creates_and_appends_validated_manifest(tmp_path: Path) -> None:
    path = tmp_path / "dataset.json"

    append_dataset_case(
        path,
        make_case("example-pr-1", 1),
        dataset_version="0.1.0",
    )
    result = append_dataset_case(
        path,
        make_case("example-pr-2", 2),
        dataset_version="0.1.0",
    )

    assert result.dataset_version == "0.1.0"
    assert [case.case_id for case in result.cases] == [
        "example-pr-1",
        "example-pr-2",
    ]
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_rejects_duplicate_case_without_replacing_manifest(
    tmp_path: Path,
) -> None:
    path = tmp_path / "dataset.json"
    case = make_case("example-pr-1", 1)
    append_dataset_case(path, case, dataset_version="0.1.0")
    original = path.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate case_id"):
        append_dataset_case(path, case, dataset_version="0.1.0")

    assert path.read_text(encoding="utf-8") == original