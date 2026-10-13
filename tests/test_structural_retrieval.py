from cohere_workshops.chunks import RepositoryChunk
from cohere_workshops.embeddings import EmbeddedChunk
from cohere_workshops.relationships import ImportRelationship
from cohere_workshops.retrieval import RetrievedChunk
from cohere_workshops.structural_retrieval import (
    expand_with_import_relationships,
)


def make_embedded(
    name: str,
    vector: tuple[float, ...],
) -> EmbeddedChunk:
    chunk = RepositoryChunk(
        chunk_id=name,
        path=f"src/{name}.py",
        language="python",
        kind="file",
        symbol=None,
        start_line=1,
        end_line=1,
        content=name,
        content_hash=f"{name}-content",
        source_hash=f"{name}-source",
    )
    return EmbeddedChunk(
        chunk=chunk,
        vector=vector,
        model="embed-v4.0",
    )


def test_expands_candidates_with_related_files() -> None:
    reader = make_embedded("reader", (1.0, 0.0))
    writer = make_embedded("writer", (0.8, 0.2))
    unrelated = make_embedded("unrelated", (0.9, 0.1))
    retrieved = (
        RetrievedChunk(embedded_chunk=reader, score=1.0),
    )
    relationships = (
        ImportRelationship(
            source_path="src/reader.py",
            target_path="src/writer.py",
            import_name="writer",
        ),
    )

    expanded = expand_with_import_relationships(
        (1.0, 0.0),
        retrieved,
        (reader, writer, unrelated),
        relationships,
    )

    assert [
        result.embedded_chunk.chunk.chunk_id
        for result in expanded
    ] == ["reader", "writer"]


def test_can_disable_structural_expansion() -> None:
    reader = make_embedded("reader", (1.0, 0.0))

    assert expand_with_import_relationships(
        (1.0, 0.0),
        (RetrievedChunk(reader, 1.0),),
        (reader,),
        (),
        max_expansions=0,
    ) == (RetrievedChunk(reader, 1.0),)
