from pathlib import Path

import pytest

from cohere_workshops.chunks import chunk_document
from cohere_workshops.documents import load_document


def test_creates_stable_file_and_symbol_chunks(tmp_path: Path) -> None:
    source = """import os

class Greeter:
    def greet(self) -> str:
        return os.getcwd()
"""
    source_file = tmp_path / "example.py"
    source_file.write_text(source, encoding="utf-8")
    document = load_document(source_file, tmp_path)

    first = chunk_document(document)
    second = chunk_document(document)

    assert first == second
    assert len({chunk.chunk_id for chunk in first}) == 3
    assert [
        (
            chunk.kind,
            chunk.symbol,
            chunk.start_line,
            chunk.end_line,
        )
        for chunk in first
    ] == [
        ("file", None, 1, 5),
        ("class", "Greeter", 3, 5),
        ("method", "Greeter.greet", 4, 5),
    ]
    assert all(chunk.source_hash == document.content_hash for chunk in first)


def test_refuses_to_chunk_suspected_secrets(tmp_path: Path) -> None:
    source_file = tmp_path / "unsafe.py"
    source_file.write_text(
        'KEY = """-----BEGIN PRIVATE KEY-----"""\n',
        encoding="utf-8",
    )
    document = load_document(source_file, tmp_path)

    with pytest.raises(ValueError, match="Refusing to chunk suspected secret"):
        chunk_document(document)
