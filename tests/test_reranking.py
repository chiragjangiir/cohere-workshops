from dataclasses import dataclass

import pytest

from cohere_workshops.chunks import RepositoryChunk
from cohere_workshops.embeddings import EmbeddedChunk
from cohere_workshops.reranking import RERANK_MODEL, rerank_chunks
from cohere_workshops.retrieval import RetrievedChunk


@dataclass
class FakeRerankResult:
    index: int
    relevance_score: float


@dataclass
class FakeRerankResponse:
    results: list[FakeRerankResult]


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def rerank(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return FakeRerankResponse(
            [
                FakeRerankResult(index=1, relevance_score=0.95),
                FakeRerankResult(index=0, relevance_score=0.70),
            ]
        )


def make_candidate(name: str, score: float) -> RetrievedChunk:
    chunk = RepositoryChunk(
        chunk_id=name,
        path=f"src/{name}.py",
        language="python",
        kind="function",
        symbol=name,
        start_line=1,
        end_line=1,
        content=f"def {name}(): pass",
        content_hash=f"{name}-content",
        source_hash=f"{name}-source",
    )
    embedded = EmbeddedChunk(
        chunk=chunk,
        vector=(1.0, 0.0),
        model="embed-v4.0",
    )
    return RetrievedChunk(embedded_chunk=embedded, score=score)


def test_reranks_only_retrieved_candidates() -> None:
    client = FakeClient()
    candidates = (
        make_candidate("first", 0.90),
        make_candidate("second", 0.80),
    )

    results = rerank_chunks(client, "Update scanning", candidates)

    assert [
        result.retrieved_chunk.embedded_chunk.chunk.chunk_id
        for result in results
    ] == ["second", "first"]
    assert [result.original_rank for result in results] == [2, 1]
    assert all(result.model == RERANK_MODEL for result in results)
    assert client.calls[0]["model"] == RERANK_MODEL
    assert client.calls[0]["top_n"] == 2


def test_returns_empty_result_without_api_call() -> None:
    client = FakeClient()

    assert rerank_chunks(client, "Update scanning", ()) == ()
    assert client.calls == []


def test_rejects_empty_query() -> None:
    with pytest.raises(ValueError, match="query must not be empty"):
        rerank_chunks(FakeClient(), " ", ())
