import json
from pathlib import Path

import pytest

from cohere_workshops.dataset import load_dataset_manifest


def write_manifest(path: Path, cases: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps({"dataset_version": "0.1.0", "cases": cases}),
        encoding="utf-8",
    )


def valid_case(case_id: str = "example-101") -> dict[str, object]:
    return {
        "case_id": case_id,
        "repository_url": "https://github.com/example/project.git",
        "pull_request": 101,
        "pre_change_commit": "a" * 40,
        "change_request": "Respect Git ignore rules during scanning.",
        "ground_truth_files": [
            "src/project/scanner.py",
            "tests/test_scanner.py",
        ],
        "ground_truth_test_files": ["tests/test_scanner.py"],
    }


def test_loads_reproducible_dataset_manifest(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    write_manifest(manifest_path, [valid_case()])

    manifest = load_dataset_manifest(manifest_path)

    assert manifest.dataset_version == "0.1.0"
    assert len(manifest.cases) == 1
    case = manifest.cases[0]
    assert case.case_id == "example-101"
    assert case.pull_request == 101
    assert case.pre_change_commit == "a" * 40
    assert case.ground_truth_test_files == ("tests/test_scanner.py",)


def test_rejects_duplicate_case_identifiers(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    write_manifest(
        manifest_path,
        [valid_case(), valid_case()],
    )

    with pytest.raises(ValueError, match="duplicate case_id"):
        load_dataset_manifest(manifest_path)


def test_requires_test_files_to_be_in_ground_truth(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    case = valid_case()
    case["ground_truth_test_files"] = ["tests/test_other.py"]
    write_manifest(manifest_path, [case])

    with pytest.raises(ValueError, match="test ground truth"):
        load_dataset_manifest(manifest_path)


def test_requires_full_pre_change_commit(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    case = valid_case()
    case["pre_change_commit"] = "abc123"
    write_manifest(manifest_path, [case])

    with pytest.raises(ValueError, match="40-character"):
        load_dataset_manifest(manifest_path)
