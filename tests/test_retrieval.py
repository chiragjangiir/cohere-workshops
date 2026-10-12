import pytest

from cohere_workshops.chunks import RepositoryChunk
from cohere_workshops.embeddings import EmbeddedChunk
from cohere_workshops.retrieval import cosine_similarity, retrieve_chunks


def make_embedded_chunk(
    chunk_id: str,
    vector: tuple[float, ...],
) -> EmbeddedChunk:
    chunk = RepositoryChunk(
        chunk_id=chunk_id,
        path=f"src/{chunk_id}.py",
        language="python",
        kind="function",
        symbol=chunk_id,
        start_line=1,
        end_line=1,
        content=f"def {chunk_id}(): pass",
        content_hash=f"{chunk_id}-content",
        source_hash=f"{chunk_id}-source",
    )
    return EmbeddedChunk(
        chunk=chunk,
        vector=vector,
        model="embed-v4.0",
    )


def test_calculates_cosine_similarity() -> None:
    assert cosine_similarity((1.0, 0.0), (1.0, 0.0)) == pytest.approx(1.0)
    assert cosine_similarity((1.0, 0.0), (0.0, 1.0)) == pytest.approx(0.0)


def test_retrieves_most_similar_chunks() -> None:
    chunks = (
        make_embedded_chunk("opposite", (-1.0, 0.0)),
        make_embedded_chunk("partial", (1.0, 1.0)),
        make_embedded_chunk("exact", (1.0, 0.0)),
    )

    results = retrieve_chunks((1.0, 0.0), chunks, top_k=2)

    assert [
        result.embedded_chunk.chunk.chunk_id
        for result in results
    ] == ["exact", "partial"]
    assert results[0].score == pytest.approx(1.0)


def test_rejects_incompatible_vectors() -> None:
    with pytest.raises(ValueError, match="equal dimensions"):
        cosine_similarity((1.0,), (1.0, 2.0))

    with pytest.raises(ValueError, match="non-zero magnitude"):
        cosine_similarity((0.0, 0.0), (1.0, 0.0))
