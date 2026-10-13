from pathlib import Path

from cohere_workshops.corpus import build_repository_corpus


def test_builds_safe_repository_corpus(tmp_path: Path) -> None:
    (tmp_path / "safe.py").write_text(
        "def run():\n    return True\n",
        encoding="utf-8",
    )
    (tmp_path / "unsafe.py").write_text(
        'KEY = """-----BEGIN ' + 'PRIVATE KEY-----"""\n',
        encoding="utf-8",
    )
    (tmp_path / "broken.py").write_text(
        "def broken(:\n",
        encoding="utf-8",
    )
    (tmp_path / "binary.py").write_bytes(b"\xff\xfe")
    (tmp_path / ".env").write_text(
        "TOKEN=secret",
        encoding="utf-8",
    )

    corpus = build_repository_corpus(tmp_path)

    assert corpus.root == tmp_path.name
    assert [
        (chunk.path, chunk.kind, chunk.symbol)
        for chunk in corpus.chunks
    ] == [
        ("safe.py", "file", None),
        ("safe.py", "function", "run"),
    ]
    assert [
        (excluded.path, excluded.reason)
        for excluded in corpus.exclusions
    ] == [
        (".env", "sensitive_path"),
        ("unsafe.py", "suspected_secret"),
    ]
    assert [
        (error.path, error.stage, error.reason)
        for error in corpus.errors
    ] == [
        ("binary.py", "load", "invalid_utf8"),
        ("broken.py", "chunk", "invalid_python_syntax"),
    ]
