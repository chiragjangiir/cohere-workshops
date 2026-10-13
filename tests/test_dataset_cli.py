import json
from pathlib import Path

from cohere_workshops.dataset_cli import build_parser, main


def write_manifest(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "dataset_version": "0.1.0",
                "cases": [
                    {
                        "case_id": "example-1",
                        "repository_url": (
                            "https://github.com/example/project.git"
                        ),
                        "pull_request": 1,
                        "pre_change_commit": "a" * 40,
                        "post_change_commit": "b" * 40,
                        "change_request": "Update the scanner.",
                        "ground_truth_files": ["scanner.py"],
                        "ground_truth_test_files": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_accepts_manifest_path() -> None:
    options = build_parser().parse_args(["dataset.json"])

    assert options.manifest == "dataset.json"
    assert options.case_id is None
    assert options.repository is None


def test_prints_schema_validation_summary(
    tmp_path: Path,
    capsys,
) -> None:
    manifest_path = tmp_path / "dataset.json"
    write_manifest(manifest_path)

    main([str(manifest_path)])

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "dataset_version": "0.1.0",
        "case_count": 1,
        "case_ids": ["example-1"],
        "schema_valid": True,
    }
