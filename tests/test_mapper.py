from pathlib import Path

from cohere_workshops.mapper import map_repository
from cohere_workshops.structure import CodeSymbol


def test_builds_repository_map_from_supported_files(
    tmp_path: Path,
) -> None:
    (tmp_path / "README.md").write_text("# Example", encoding="utf-8")
    (tmp_path / "app.py").write_text(
        "import json\n\ndef run() -> None:\n    pass\n",
        encoding="utf-8",
    )

    result = map_repository(tmp_path)

    assert result.root == tmp_path.name
    assert [file.path for file in result.files] == [
        "README.md",
        "app.py",
    ]

    python_file = result.files[1]
    assert python_file.imports == ("json",)
    assert python_file.symbols == (
        CodeSymbol(
            name="run",
            qualified_name="run",
            kind="function",
            line=3,
            end_line=4,
        ),
    )

    markdown_file = result.files[0]
    assert markdown_file.symbols == ()
    assert markdown_file.imports == ()

def test_includes_scanner_exclusions_in_repository_map(
    tmp_path: Path,
) -> None:
    (tmp_path / "app.py").write_text("pass", encoding="utf-8")
    (tmp_path / ".env").write_text("TOKEN=secret", encoding="utf-8")

    result = map_repository(tmp_path)

    assert [file.path for file in result.files] == ["app.py"]
    assert [
        (excluded.path, excluded.reason)
        for excluded in result.exclusions
    ] == [(".env", "sensitive_path")]


def test_records_processing_errors_without_stopping_repository_map(
    tmp_path: Path,
) -> None:
    (tmp_path / "valid.py").write_text("def run():\n    pass\n", encoding="utf-8")
    (tmp_path / "broken.py").write_text("def broken(:\n", encoding="utf-8")
    (tmp_path / "binary.py").write_bytes(b"\xff\xfe")

    result = map_repository(tmp_path)

    assert [file.path for file in result.files] == [
        "broken.py",
        "valid.py",
    ]
    assert [
        (error.path, error.stage, error.reason)
        for error in result.errors
    ] == [
        ("binary.py", "load", "invalid_utf8"),
        ("broken.py", "structure", "invalid_python_syntax"),
    ]


def test_excludes_files_containing_suspected_secrets(
    tmp_path: Path,
) -> None:
    (tmp_path / "safe.py").write_text("value = 1\n", encoding="utf-8")
    (tmp_path / "unsafe.py").write_text(
        'PRIVATE_KEY = """-----BEGIN '
        + 'PRIVATE KEY-----"""\n',
        encoding="utf-8",
    )

    result = map_repository(tmp_path)

    assert [file.path for file in result.files] == ["safe.py"]
    assert [
        (excluded.path, excluded.reason)
        for excluded in result.exclusions
    ] == [("unsafe.py", "suspected_secret")]
