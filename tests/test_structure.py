from pathlib import Path

from cohere_workshops.documents import load_document
from cohere_workshops.structure import CodeSymbol, analyze_python


def test_extracts_python_symbols_and_imports(tmp_path: Path) -> None:
    source = """import os
from pathlib import Path

class Mapper:
    def build(self) -> Path:
        return Path.cwd()

async def scan() -> str:
    return os.getcwd()
"""
    source_file = tmp_path / "example.py"
    source_file.write_text(source, encoding="utf-8")

    structure = analyze_python(load_document(source_file, tmp_path))

    assert structure.imports == ("os", "pathlib")
    assert structure.symbols == (
        CodeSymbol(
            name="Mapper",
            qualified_name="Mapper",
            kind="class",
            line=4,
            end_line=6,
        ),
        CodeSymbol(
            name="build",
            qualified_name="Mapper.build",
            kind="method",
            line=5,
            end_line=6,
        ),
        CodeSymbol(
            name="scan",
            qualified_name="scan",
            kind="function",
            line=8,
            end_line=9,
        ),
    )