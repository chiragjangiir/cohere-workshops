"""Expand semantic candidates with deterministic repository structure."""

from collections.abc import Sequence

from cohere_workshops.embeddings import EmbeddedChunk
from cohere_workshops.relationships import (
    ImportRelationship,
    related_paths,
)
from cohere_workshops.retrieval import (
    RetrievedChunk,
    cosine_similarity,
)


def expand_with_import_relationships(
    query_vector: Sequence[float],
    retrieved: Sequence[RetrievedChunk],
    embedded_chunks: Sequence[EmbeddedChunk],
    relationships: Sequence[ImportRelationship],
    *,
    max_expansions: int = 10,
) -> tuple[RetrievedChunk, ...]:
    """Add semantically strongest chunks from one-hop related files."""
    if max_expansions < 0:
        raise ValueError("max_expansions must not be negative")

    if not retrieved or max_expansions == 0:
        return tuple(retrieved)

    seed_paths = {
        result.embedded_chunk.chunk.path
        for result in retrieved
    }
    neighbor_paths = set(
        related_paths(seed_paths, relationships)
    )
    existing_chunk_ids = {
        result.embedded_chunk.chunk.chunk_id
        for result in retrieved
    }

    expansions = [
        RetrievedChunk(
            embedded_chunk=embedded_chunk,
            score=cosine_similarity(
                query_vector,
                embedded_chunk.vector,
            ),
        )
        for embedded_chunk in embedded_chunks
        if embedded_chunk.chunk.path in neighbor_paths
        and embedded_chunk.chunk.chunk_id not in existing_chunk_ids
    ]
    expansions.sort(
        key=lambda result: (
            -result.score,
            result.embedded_chunk.chunk.chunk_id,
        )
    )

    return (*retrieved, *expansions[:max_expansions])
