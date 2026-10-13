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



def test_splits_large_documents_without_losing_content(
    tmp_path: Path,
) -> None:
    content = "".join(
        f"line {number:02d} contains text\n"
        for number in range(1, 11)
    )
    source_file = tmp_path / "notes.md"
    source_file.write_text(content, encoding="utf-8")
    document = load_document(source_file, tmp_path)

    chunks = chunk_document(document, max_characters=45)

    assert len(chunks) > 1
    assert "".join(chunk.content for chunk in chunks) == content
    assert all(len(chunk.content) <= 45 for chunk in chunks)
    assert [chunk.part_index for chunk in chunks] == list(
        range(1, len(chunks) + 1)
    )
    assert all(chunk.part_count == len(chunks) for chunk in chunks)
    assert chunks[0].start_line == 1
    assert chunks[-1].end_line == 10


def test_splits_single_line_longer_than_limit(
    tmp_path: Path,
) -> None:
    content = "x" * 25
    source_file = tmp_path / "data.json"
    source_file.write_text(content, encoding="utf-8")
    document = load_document(source_file, tmp_path)

    chunks = chunk_document(document, max_characters=10)

    assert [len(chunk.content) for chunk in chunks] == [10, 10, 5]
    assert "".join(chunk.content for chunk in chunks) == content
    assert all(
        (chunk.start_line, chunk.end_line) == (1, 1)
        for chunk in chunks
    )
