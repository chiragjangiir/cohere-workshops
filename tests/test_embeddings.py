from dataclasses import dataclass

import pytest

from cohere_workshops.chunks import RepositoryChunk
from cohere_workshops.embeddings import (
    EMBED_MODEL,
    embed_chunks,
    embed_query,
)


@dataclass
class FakeEmbeddings:
    float: list[list[float]]


@dataclass
class FakeResponse:
    embeddings: FakeEmbeddings


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def embed(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        texts = kwargs["texts"]
        assert isinstance(texts, list)
        return FakeResponse(
            FakeEmbeddings(
                [[float(index), 1.0] for index, _ in enumerate(texts)]
            )
        )


def make_chunk(chunk_id: str) -> RepositoryChunk:
    return RepositoryChunk(
        chunk_id=chunk_id,
        path=f"src/{chunk_id}.py",
        language="python",
        kind="function",
        symbol="run",
        start_line=1,
        end_line=2,
        content="def run():\n    pass",
        content_hash="chunk-hash",
        source_hash="source-hash",
    )


def test_embeds_repository_chunks_in_document_mode() -> None:
    client = FakeClient()
    chunks = (make_chunk("one"), make_chunk("two"))

    results = embed_chunks(client, chunks, batch_size=1)

    assert [result.chunk.chunk_id for result in results] == ["one", "two"]
    assert all(result.model == EMBED_MODEL for result in results)
    assert len(client.calls) == 2
    assert all(
        call["input_type"] == "search_document"
        for call in client.calls
    )
    assert all(
        call["embedding_types"] == ["float"]
        for call in client.calls
    )


def test_embeds_change_request_in_query_mode() -> None:
    client = FakeClient()

    vector = embed_query(client, "Update the repository scanner")

    assert vector == (0.0, 1.0)
    assert client.calls[0]["input_type"] == "search_query"


def test_reads_cohere_sdk_float_field() -> None:
    from cohere.core.unchecked_base_model import construct_type
    from cohere.types import EmbedByTypeResponse

    class SdkClient:
        def embed(self, **kwargs: object) -> object:
            return construct_type(
                type_=EmbedByTypeResponse,
                object_={
                    "id": "test",
                    "response_type": "embeddings_by_type",
                    "texts": kwargs["texts"],
                    "embeddings": {"float": [[0.5, 0.25]]},
                },
            )

    assert embed_query(SdkClient(), "Find the scanner") == (0.5, 0.25)


def test_rejects_empty_query() -> None:
    with pytest.raises(ValueError, match="query must not be empty"):
        embed_query(FakeClient(), "   ")