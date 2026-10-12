"""Deterministic BM25 keyword retrieval baseline."""

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from cohere_workshops.chunks import RepositoryChunk

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
CAMEL_CASE_BOUNDARY = re.compile(r"([a-z0-9])([A-Z])")


@dataclass(frozen=True)
class KeywordResult:
    """A repository chunk with its BM25 score."""

    chunk: RepositoryChunk
    score: float


def tokenize(text: str) -> tuple[str, ...]:
    """Tokenize prose and code identifiers deterministically."""
    expanded = CAMEL_CASE_BOUNDARY.sub(r"\1 \2", text)
    expanded = expanded.replace("_", " ")
    return tuple(
        token.lower()
        for token in TOKEN_PATTERN.findall(expanded)
    )


def keyword_retrieve(
    query: str,
    chunks: Sequence[RepositoryChunk],
    *,
    top_k: int = 20,
    k1: float = 1.5,
    b: float = 0.75,
) -> tuple[KeywordResult, ...]:
    """Rank repository chunks using the BM25 lexical baseline."""
    if not query.strip():
        raise ValueError("query must not be empty")

    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    if not chunks:
        return ()

    query_terms = set(tokenize(query))
    documents = [tokenize(chunk.content) for chunk in chunks]
    average_length = (
        sum(len(document) for document in documents) / len(documents)
    )

    document_frequency = Counter(
        term
        for document in documents
        for term in set(document)
        if term in query_terms
    )

    ranked: list[KeywordResult] = []

    for chunk, document in zip(chunks, documents, strict=True):
        frequencies = Counter(document)
        document_length = len(document)
        score = 0.0

        for term in query_terms:
            frequency = frequencies[term]

            if frequency == 0:
                continue

            containing_documents = document_frequency[term]
            inverse_document_frequency = math.log(
                1
                + (
                    len(documents)
                    - containing_documents
                    + 0.5
                )
                / (containing_documents + 0.5)
            )
            length_normalization = frequency + k1 * (
                1 - b + b * document_length / max(average_length, 1)
            )
            score += inverse_document_frequency * (
                frequency * (k1 + 1)
            ) / length_normalization

        if score > 0:
            ranked.append(KeywordResult(chunk=chunk, score=score))

    ranked.sort(key=lambda result: (-result.score, result.chunk.chunk_id))
    return tuple(ranked[:top_k])
