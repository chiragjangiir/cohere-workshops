"""Create Cohere embeddings for repository retrieval."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from cohere_workshops.chunks import RepositoryChunk

EMBED_MODEL = "embed-v4.0"
DEFAULT_BATCH_SIZE = 96


class EmbedClient(Protocol):
    """Minimal client interface required by the embedding layer."""

    def embed(self, **kwargs: object) -> object:
        """Return an embedding response."""


@dataclass(frozen=True)
class EmbeddedChunk:
    """A repository chunk paired with its Cohere embedding."""

    chunk: RepositoryChunk
    vector: tuple[float, ...]
    model: str


def format_chunk_for_embedding(chunk: RepositoryChunk) -> str:
    """Add traceable repository context to chunk content."""
    symbol = chunk.symbol or "not_applicable"
    return (
        f"path: {chunk.path}\n"
        f"language: {chunk.language}\n"
        f"kind: {chunk.kind}\n"
        f"symbol: {symbol}\n"
        f"lines: {chunk.start_line}-{chunk.end_line}\n"
        f"part: {chunk.part_index}/{chunk.part_count}\n\n"
        f"{chunk.content}"
    )


def _extract_float_vectors(
    response: object,
    expected_count: int,
) -> tuple[tuple[float, ...], ...]:
    embeddings = getattr(response, "embeddings", None)
    # The Cohere SDK exposes float vectors as `float_` because `float` is a
    # Python built-in. `float` is kept for simple test doubles.
    vectors = getattr(embeddings, "float_", None)
    if vectors is None:
        vectors = getattr(embeddings, "float", None)

    if vectors is None:
        raise ValueError("Cohere response did not contain float embeddings")

    normalized = tuple(
        tuple(float(value) for value in vector)
        for vector in vectors
    )

    if len(normalized) != expected_count:
        raise ValueError(
            "Cohere returned a different number of embeddings than requested"
        )

    if any(not vector for vector in normalized):
        raise ValueError("Cohere returned an empty embedding vector")

    return normalized


def embed_chunks(
    client: EmbedClient,
    chunks: Sequence[RepositoryChunk],
    *,
    model: str = EMBED_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[EmbeddedChunk, ...]:
    """Embed repository chunks using Cohere's search-document mode."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    results: list[EmbeddedChunk] = []

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        response = client.embed(
            model=model,
            texts=[format_chunk_for_embedding(chunk) for chunk in batch],
            input_type="search_document",
            embedding_types=["float"],
        )
        vectors = _extract_float_vectors(response, len(batch))
        results.extend(
            EmbeddedChunk(chunk=chunk, vector=vector, model=model)
            for chunk, vector in zip(batch, vectors, strict=True)
        )

    return tuple(results)


def embed_query(
    client: EmbedClient,
    query: str,
    *,
    model: str = EMBED_MODEL,
) -> tuple[float, ...]:
    """Embed one change request using Cohere's search-query mode."""
    if not query.strip():
        raise ValueError("query must not be empty")

    response = client.embed(
        model=model,
        texts=[query],
        input_type="search_query",
        embedding_types=["float"],
    )
    return _extract_float_vectors(response, 1)[0]