"""Run evidence-preserving semantic repository retrieval."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from cohere_workshops.corpus import (
    CorpusError,
    build_repository_corpus,
)
from cohere_workshops.embeddings import (
    EmbedClient,
    embed_chunks,
    embed_query,
)
from cohere_workshops.reranking import (
    RerankClient,
    RerankedChunk,
    rerank_chunks,
)
from cohere_workshops.retrieval import retrieve_chunks
from cohere_workshops.scanner import ExcludedFile


class SemanticClient(EmbedClient, RerankClient, Protocol):
    """Client supporting Cohere Embed and Rerank."""


@dataclass(frozen=True)
class SemanticSearchResult:
    """Traceable output from semantic repository retrieval."""

    root: str
    chunk_count: int
    exclusions: tuple[ExcludedFile, ...]
    errors: tuple[CorpusError, ...]
    results: tuple[RerankedChunk, ...]


def semantic_search_repository(
    client: SemanticClient,
    repository: Path,
    query: str,
    *,
    retrieval_k: int = 20,
    rerank_k: int = 10,
) -> SemanticSearchResult:
    """Retrieve and rerank repository evidence for a change request."""
    if not query.strip():
        raise ValueError("query must not be empty")

    corpus = build_repository_corpus(repository)

    if not corpus.chunks:
        return SemanticSearchResult(
            root=corpus.root,
            chunk_count=0,
            exclusions=corpus.exclusions,
            errors=corpus.errors,
            results=(),
        )

    embedded_chunks = embed_chunks(client, corpus.chunks)
    query_vector = embed_query(client, query)
    retrieved = retrieve_chunks(
        query_vector,
        embedded_chunks,
        top_k=retrieval_k,
    )
    reranked = rerank_chunks(
        client,
        query,
        retrieved,
        top_n=rerank_k,
    )

    return SemanticSearchResult(
        root=corpus.root,
        chunk_count=len(corpus.chunks),
        exclusions=corpus.exclusions,
        errors=corpus.errors,
        results=reranked,
    )
