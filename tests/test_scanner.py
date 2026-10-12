from pathlib import Path

from cohere_workshops.scanner import MAX_FILE_SIZE_BYTES, discover_source_files


def test_discovers_supported_files_and_ignores_generated_directories(
    tmp_path: Path,
) -> None:
    (tmp_path / "app.py").write_text("print('hello')", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Example", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("Ignore me", encoding="utf-8")

    virtual_environment = tmp_path / ".venv"
    virtual_environment.mkdir()
    (virtual_environment / "ignored.py").write_text("", encoding="utf-8")

    discovered = [
        path.relative_to(tmp_path)
        for path in discover_source_files(tmp_path)
    ]

    assert discovered == [
        Path("README.md"),
        Path("app.py"),
    ]

def test_ignores_files_larger_than_limit(tmp_path: Path) -> None:
    oversized_file = tmp_path / "oversized.py"
    oversized_file.write_bytes(b"x" * (MAX_FILE_SIZE_BYTES + 1))

    assert discover_source_files(tmp_path) == []


def test_ignores_sensitive_file_names(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text('{"safe": true}', encoding="utf-8")
    (tmp_path / "credentials.json").write_text(
        '{"api_key": "secret"}',
        encoding="utf-8",
    )

    discovered = [
        path.relative_to(tmp_path)
        for path in discover_source_files(tmp_path)
    ]

    assert discovered == [Path("config.json")]
