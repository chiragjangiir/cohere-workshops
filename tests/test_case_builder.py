import subprocess
from pathlib import Path

from cohere_workshops.case_builder import build_evaluation_case


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
        "user.name=Test Researcher",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-m",
        message,
    )
    return git(repository, "rev-parse", "HEAD")


def test_builds_case_with_diff_derived_ground_truth(
    tmp_path: Path,
) -> None:
    git(tmp_path, "init")

    source = tmp_path / "src" / "app.py"
    source.parent.mkdir()
    source.write_text("def run():\n    return 1\n", encoding="utf-8")
    pre_change_commit = commit_all(tmp_path, "Initial version")

    source.write_text("def run():\n    return 2\n", encoding="utf-8")
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_app.py").write_text(
        "def test_run():\n    assert True\n",
        encoding="utf-8",
    )
    post_change_commit = commit_all(tmp_path, "Update run behavior")

    case = build_evaluation_case(
        tmp_path,
        case_id="example-pr-7",
        repository_url="https://github.com/example/project.git",
        pull_request=7,
        pre_change_commit=pre_change_commit,
        post_change_commit=post_change_commit,
        change_request="Update the run behavior and cover it with a test.",
    )

    assert case.ground_truth_files == (
        "src/app.py",
        "tests/test_app.py",
    )
    assert case.ground_truth_test_files == ("tests/test_app.py",)
    assert case.pre_change_commit == pre_change_commit
    assert case.post_change_commit == post_change_commit