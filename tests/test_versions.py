import subprocess
from pathlib import Path

import pytest

from cohere_workshops.versions import (
    CURRENT,
    REMOVED,
    SUPERSEDED,
    build_versioned_corpus,
    load_revision,
    resolve_commit,
)

OLD_PAYMENTS = '''def fee(amount):
    return amount * 0.02


def validate(phone):
    return len(phone) == 10
'''

NEW_PAYMENTS = '''def fee(amount):
    return 0 if amount <= 1000 else amount * 0.01


def normalize(phone):
    return "254" + phone[1:]
'''

RECEIPTS = '''def receipt(amount):
    return f"Sent {amount}"
'''


def git(repository: Path, *arguments: str) -> None:
    subprocess.run(
        [
            "git",
            "-c", "user.name=Test",
            "-c", "user.email=test@example.com",
            "-c", "commit.gpgsign=false",
            *arguments,
        ],
        cwd=repository,
        check=True,
        capture_output=True,
    )


def commit_files(repository: Path, files: dict[str, str], tag: str) -> None:
    for name, content in files.items():
        path = repository / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    git(repository, "add", ".")
    git(repository, "commit", "-m", tag)
    git(repository, "tag", tag)


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    git(tmp_path, "init")
    commit_files(
        tmp_path,
        {"payments.py": OLD_PAYMENTS, "receipts.py": RECEIPTS},
        "v1",
    )
    commit_files(tmp_path, {"payments.py": NEW_PAYMENTS}, "v2")
    return tmp_path


def statuses(corpus, kind: str = "function") -> dict[tuple[str, str], str]:
    return {
        (item.revision, item.chunk.symbol): item.status
        for item in corpus.chunks
        if item.chunk.kind == kind
    }


def test_labels_current_superseded_and_removed_symbols(repository: Path) -> None:
    corpus = build_versioned_corpus(repository, "v1", "v2")

    assert statuses(corpus) == {
        ("v2", "fee"): CURRENT,
        ("v2", "normalize"): CURRENT,
        ("v2", "receipt"): CURRENT,
        ("v1", "fee"): SUPERSEDED,
        ("v1", "validate"): REMOVED,
    }


def test_skips_unchanged_previous_chunks(repository: Path) -> None:
    corpus = build_versioned_corpus(repository, "v1", "v2")

    # receipts.py file chunk and its receipt function did not change.
    assert corpus.unchanged_count == 2
    assert not any(
        item.revision == "v1" and item.chunk.path == "receipts.py"
        for item in corpus.chunks
    )


def test_records_full_commit_hashes(repository: Path) -> None:
    corpus = build_versioned_corpus(repository, "v1", "v2")

    assert corpus.previous.commit == resolve_commit(repository, "v1")
    assert corpus.current.commit == resolve_commit(repository, "v2")
    assert all(len(item.commit) == 40 for item in corpus.chunks)


def test_reads_old_revision_without_checkout(repository: Path) -> None:
    snapshot = load_revision(repository, "v1")

    symbols = {chunk.symbol for chunk in snapshot.chunks if chunk.symbol}
    assert symbols == {"fee", "validate", "receipt"}
    assert (repository / "payments.py").read_text() == NEW_PAYMENTS


def test_excludes_secrets_and_sensitive_paths(repository: Path) -> None:
    fake_key = "-----BEGIN " + "PRIVATE KEY-----"
    commit_files(
        repository,
        {"config.py": f'KEY = """{fake_key}"""\n', ".env": "TOKEN=x\n"},
        "v3",
    )

    snapshot = load_revision(repository, "v3")

    assert {(item.path, item.reason) for item in snapshot.exclusions} == {
        (".env", "sensitive_path"),
        ("config.py", "suspected_secret"),
    }
    assert not any(chunk.path == "config.py" for chunk in snapshot.chunks)


def test_records_invalid_python_as_error(repository: Path) -> None:
    commit_files(repository, {"broken.py": "def broken(:\n"}, "v3")

    snapshot = load_revision(repository, "v3")

    assert [(error.path, error.reason) for error in snapshot.errors] == [
        ("broken.py", "invalid_python_syntax")
    ]


def test_rejects_unknown_revision(repository: Path) -> None:
    with pytest.raises(ValueError):
        load_revision(repository, "does-not-exist")


def test_rejects_option_like_revision(repository: Path) -> None:
    with pytest.raises(ValueError, match="Invalid revision"):
        resolve_commit(repository, "--all")