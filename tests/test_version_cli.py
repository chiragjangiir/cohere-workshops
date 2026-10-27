import json
from pathlib import Path

import pytest

from cohere_workshops import version_cli
from cohere_workshops.demo_repo import create_demo_repository
from cohere_workshops.demo_repo import main as demo_main
from cohere_workshops.versions import (
    CURRENT,
    REMOVED,
    SUPERSEDED,
    build_versioned_corpus,
)


@pytest.fixture
def demo_repository(tmp_path: Path) -> Path:
    return create_demo_repository(tmp_path / "demo_repo")


def test_demo_repository_has_expected_versions(demo_repository: Path) -> None:
    corpus = build_versioned_corpus(demo_repository, "v1", "v2")
    labels = {
        (item.revision, item.chunk.symbol): item.status
        for item in corpus.chunks
        if item.chunk.kind == "function"
    }

    assert labels[("v2", "calculate_transfer_fee")] == CURRENT
    assert labels[("v1", "calculate_transfer_fee")] == SUPERSEDED
    assert labels[("v1", "validate_phone_number")] == REMOVED
    assert ("v1", "format_receipt") not in labels


def test_demo_command_refuses_to_overwrite(
    demo_repository: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        demo_main([str(demo_repository)])

    assert "already exists" in capsys.readouterr().err


def test_offline_run_prints_report_and_records_results(
    demo_repository: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "runs.jsonl"

    version_cli.main(
        [
            str(demo_repository),
            "--old", "v1",
            "--new", "v2",
            "--offline",
            "--top-k", "3",
            "--output", str(output),
        ]
    )

    printed = capsys.readouterr().out
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert "Indexed at symbol level: 4 current, 2 superseded, 1 removed" in printed
    assert "Skipped 1 unchanged older chunks" in printed
    assert "Top result is current:" in printed
    assert len(records) == len(version_cli.DEMO_QUERIES)
    assert all(record["embed_model"] == "offline-word-hashing" for record in records)
    assert all(len(record["results"]) == 3 for record in records)


def test_file_level_uses_only_file_chunks(
    demo_repository: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "runs.jsonl"

    version_cli.main(
        [
            str(demo_repository),
            "--old", "v1",
            "--new", "v2",
            "--offline",
            "--level", "file",
            "--query", "transfer fee",
            "--output", str(output),
        ]
    )

    printed = capsys.readouterr().out
    [record] = [json.loads(line) for line in output.read_text().splitlines()]
    assert "Indexed at file level: 2 current, 1 superseded, 0 removed" in printed
    assert "Skipped 1 unchanged older chunks" in printed
    assert {result["kind"] for result in record["results"]} == {"file"}


def test_live_mode_requires_api_key(
    demo_repository: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.setattr(version_cli, "load_dotenv", lambda: None)

    with pytest.raises(RuntimeError, match="COHERE_API_KEY is missing"):
        version_cli.main([str(demo_repository), "--old", "v1", "--new", "v2"])
