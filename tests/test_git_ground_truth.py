import subprocess
from pathlib import Path

from cohere_workshops.git_ground_truth import extract_diff_ground_truth


def git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def commit_all(repository: Path, message: str) -> str:
    git(repository, "add", ".")
    git(
        repository,
        "-c",
        "user.name=Dataset Test",
        "-c",
        "user.email=dataset@example.invalid",
        "commit",
        "-m",
        message,
    )
    return git(repository, "rev-parse", "HEAD")


def test_extracts_diff_derived_file_and_test_ground_truth(
    tmp_path: Path,
) -> None:
    git(tmp_path, "init")
    source = tmp_path / "src" / "project"
    tests = tmp_path / "tests"
    source.mkdir(parents=True)
    tests.mkdir()
    (source / "scanner.py").write_text("LIMIT = 1\n", encoding="utf-8")
    (tests / "test_scanner.py").write_text(
        "def test_limit(): pass\n", encoding="utf-8"
    )
    pre_change = commit_all(tmp_path, "Initial state")

    (source / "scanner.py").write_text("LIMIT = 2\n", encoding="utf-8")
    (tests / "test_scanner.py").write_text(
        "def test_limit(): assert True\n", encoding="utf-8"
    )
    (tmp_path / "README.md").write_text("# Example\n", encoding="utf-8")
    post_change = commit_all(tmp_path, "Update scanner")

    result = extract_diff_ground_truth(
        tmp_path,
        pre_change,
        post_change,
    )

    assert result.files == (
        "README.md",
        "src/project/scanner.py",
        "tests/test_scanner.py",
    )
    assert result.test_files == ("tests/test_scanner.py",)
    assert result.pre_change_commit == pre_change
    assert result.post_change_commit == post_change


def test_includes_deleted_files_in_ground_truth(tmp_path: Path) -> None:
    git(tmp_path, "init")
    removed = tmp_path / "removed.py"
    removed.write_text("VALUE = 1\n", encoding="utf-8")
    pre_change = commit_all(tmp_path, "Add file")
    removed.unlink()
    post_change = commit_all(tmp_path, "Remove file")

    result = extract_diff_ground_truth(
        tmp_path,
        pre_change,
        post_change,
    )

    assert result.files == ("removed.py",)

def test_verifies_manifest_ground_truth_against_git(tmp_path: Path) -> None:
    from cohere_workshops.dataset import EvaluationCase
    from cohere_workshops.git_ground_truth import verify_case_ground_truth

    git(tmp_path, "init")
    source = tmp_path / "app.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    pre_change = commit_all(tmp_path, "Initial state")
    source.write_text("VALUE = 2\n", encoding="utf-8")
    post_change = commit_all(tmp_path, "Update app")
    case = EvaluationCase(
        case_id="example-1",
        repository_url="https://github.com/example/project.git",
        pull_request=1,
        pre_change_commit=pre_change,
        post_change_commit=post_change,
        change_request="Update the application value.",
        ground_truth_files=("app.py",),
        ground_truth_test_files=(),
    )

    result = verify_case_ground_truth(tmp_path, case)

    assert result.files == ("app.py",)


def test_rejects_manifest_ground_truth_that_disagrees_with_git(
    tmp_path: Path,
) -> None:
    import pytest

    from cohere_workshops.dataset import EvaluationCase
    from cohere_workshops.git_ground_truth import verify_case_ground_truth

    git(tmp_path, "init")
    source = tmp_path / "app.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    pre_change = commit_all(tmp_path, "Initial state")
    source.write_text("VALUE = 2\n", encoding="utf-8")
    post_change = commit_all(tmp_path, "Update app")
    case = EvaluationCase(
        case_id="example-1",
        repository_url="https://github.com/example/project.git",
        pull_request=1,
        pre_change_commit=pre_change,
        post_change_commit=post_change,
        change_request="Update the application value.",
        ground_truth_files=("wrong.py",),
        ground_truth_test_files=(),
    )

    with pytest.raises(ValueError, match="does not match Git diff"):
        verify_case_ground_truth(tmp_path, case)
