"""Rank embedded repository chunks by vector similarity."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import sqrt

from cohere_workshops.embeddings import EmbeddedChunk


@dataclass(frozen=True)
class RetrievedChunk:
    """A repository chunk with its semantic similarity score."""

    embedded_chunk: EmbeddedChunk
    score: float


def cosine_similarity(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    """Calculate cosine similarity between equal-length vectors."""
    if not left or not right:
        raise ValueError("vectors must not be empty")

    if len(left) != len(right):
        raise ValueError("vectors must have equal dimensions")

    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))

    if left_norm == 0 or right_norm == 0:
        raise ValueError("vectors must have non-zero magnitude")

    dot_product = sum(
        left_value * right_value
        for left_value, right_value in zip(left, right, strict=True)
    )
    return dot_product / (left_norm * right_norm)


def retrieve_chunks(
    query_vector: Sequence[float],
    embedded_chunks: Sequence[EmbeddedChunk],
    *,
    top_k: int = 20,
) -> tuple[RetrievedChunk, ...]:
    """Return the most similar chunks with deterministic tie-breaking."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    ranked = [
        RetrievedChunk(
            embedded_chunk=embedded_chunk,
            score=cosine_similarity(
                query_vector,
                embedded_chunk.vector,
            ),
        )
        for embedded_chunk in embedded_chunks
    ]

    ranked.sort(
        key=lambda result: (
            -result.score,
            result.embedded_chunk.chunk.chunk_id,
        )
    )
    return tuple(ranked[:top_k])
