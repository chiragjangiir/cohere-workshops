from pathlib import Path

from cohere_workshops.mapper import map_repository
from cohere_workshops.relationships import (
    related_paths,
    resolve_import_relationships,
)


def test_resolves_local_import_relationships(tmp_path: Path) -> None:
    package = tmp_path / "src" / "example"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "reader.py").write_text(
        "from example.writer import write\nimport os\n",
        encoding="utf-8",
    )
    (package / "writer.py").write_text(
        "def write():\n    pass\n",
        encoding="utf-8",
    )

    relationships = resolve_import_relationships(
        map_repository(tmp_path)
    )

    assert [
        (
            relationship.source_path,
            relationship.target_path,
            relationship.import_name,
        )
        for relationship in relationships
    ] == [
        (
            "src/example/reader.py",
            "src/example/writer.py",
            "example.writer",
        )
    ]


def test_finds_both_incoming_and_outgoing_neighbors(
    tmp_path: Path,
) -> None:
    package = tmp_path / "src" / "example"
    package.mkdir(parents=True)
    (package / "a.py").write_text(
        "import example.b\n",
        encoding="utf-8",
    )
    (package / "b.py").write_text(
        "import example.c\n",
        encoding="utf-8",
    )
    (package / "c.py").write_text("pass\n", encoding="utf-8")

    relationships = resolve_import_relationships(
        map_repository(tmp_path)
    )

    assert related_paths(
        {"src/example/b.py"},
        relationships,
    ) == (
        "src/example/a.py",
        "src/example/c.py",
    )
