"""Rerank retrieved repository chunks with Cohere."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from cohere_workshops.embeddings import format_chunk_for_embedding
from cohere_workshops.retrieval import RetrievedChunk

RERANK_MODEL = "rerank-v4.0-pro"


class RerankClient(Protocol):
    """Minimal client interface required by the reranking layer."""

    def rerank(self, **kwargs: object) -> object:
        """Return a reranking response."""


@dataclass(frozen=True)
class RerankedChunk:
    """A retrieved chunk with its Cohere relevance score."""

    retrieved_chunk: RetrievedChunk
    rerank_score: float
    model: str
    original_rank: int


def rerank_chunks(
    client: RerankClient,
    query: str,
    candidates: Sequence[RetrievedChunk],
    *,
    model: str = RERANK_MODEL,
    top_n: int = 10,
) -> tuple[RerankedChunk, ...]:
    """Rerank retrieved candidates against the original change request."""
    if not query.strip():
        raise ValueError("query must not be empty")

    if top_n < 1:
        raise ValueError("top_n must be at least 1")

    if not candidates:
        return ()

    requested_top_n = min(top_n, len(candidates))
    response = client.rerank(
        model=model,
        query=query,
        documents=[
            format_chunk_for_embedding(
                candidate.embedded_chunk.chunk
            )
            for candidate in candidates
        ],
        top_n=requested_top_n,
    )
    results = getattr(response, "results", None)

    if results is None:
        raise ValueError("Cohere response did not contain rerank results")

    reranked: list[RerankedChunk] = []
    seen_indices: set[int] = set()

    for result in results:
        index = int(result.index)
        score = float(result.relevance_score)

        if index < 0 or index >= len(candidates):
            raise ValueError("Cohere returned an invalid document index")

        if index in seen_indices:
            raise ValueError("Cohere returned a duplicate document index")

        seen_indices.add(index)
        reranked.append(
            RerankedChunk(
                retrieved_chunk=candidates[index],
                rerank_score=score,
                model=model,
                original_rank=index + 1,
            )
        )

    if len(reranked) != requested_top_n:
        raise ValueError(
            "Cohere returned a different number of rerank results than requested"
        )

    return tuple(reranked)
