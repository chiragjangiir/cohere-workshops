"""Search code across Git versions and report which version ranks first."""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from cohere_workshops.embeddings import (
    EMBED_MODEL,
    EmbedClient,
    embed_chunks,
    embed_query,
)
from cohere_workshops.retrieval import retrieve_chunks
from cohere_workshops.versions import CURRENT, VersionedChunk, VersionedCorpus

LEVELS = ("symbol", "file")


@dataclass(frozen=True)
class VersionHit:
    """One ranked search result with its version label."""

    rank: int
    score: float
    item: VersionedChunk


@dataclass(frozen=True)
class VersionSummary:
    """Whether current code or outdated code won for one query."""

    top_result_is_current: bool
    outdated_in_top_k: int
    best_current_score: float | None
    best_outdated_score: float | None
    score_gap: float | None


@dataclass(frozen=True)
class QueryResult:
    """Ranked hits and their summary for one natural language query."""

    query: str
    hits: tuple[VersionHit, ...]
    summary: VersionSummary


def select_level(
    chunks: Sequence[VersionedChunk],
    level: str,
) -> tuple[VersionedChunk, ...]:
    """Keep symbol chunks (functions, classes) or whole-file chunks."""
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}")
    if level == "file":
        return tuple(item for item in chunks if item.chunk.kind == "file")
    return tuple(item for item in chunks if item.chunk.kind != "file")


def summarize_hits(hits: Sequence[VersionHit]) -> VersionSummary:
    """Compare the best current result with the best outdated result."""
    if not hits:
        raise ValueError("hits must not be empty")

    current = [hit.score for hit in hits if hit.item.status == CURRENT]
    outdated = [hit.score for hit in hits if hit.item.status != CURRENT]
    best_current = max(current) if current else None
    best_outdated = max(outdated) if outdated else None
    gap = (
        best_current - best_outdated
        if best_current is not None and best_outdated is not None
        else None
    )

    return VersionSummary(
        top_result_is_current=hits[0].item.status == CURRENT,
        outdated_in_top_k=len(outdated),
        best_current_score=best_current,
        best_outdated_score=best_outdated,
        score_gap=gap,
    )


def search_versions(
    client: EmbedClient,
    corpus: VersionedCorpus,
    queries: Sequence[str],
    *,
    level: str = "symbol",
    top_k: int = 5,
    model: str = EMBED_MODEL,
) -> tuple[QueryResult, ...]:
    """Embed versioned chunks once, then rank them for every query."""
    if not queries:
        raise ValueError("at least one query is required")

    items = select_level(corpus.chunks, level)
    if not items:
        return ()

    by_chunk_id = {item.chunk.chunk_id: item for item in items}
    if len(by_chunk_id) != len(items):
        raise ValueError("versioned chunks must have unique chunk IDs")

    embedded = embed_chunks(client, [item.chunk for item in items], model=model)
    results: list[QueryResult] = []

    for query in queries:
        query_vector = embed_query(client, query, model=model)
        retrieved = retrieve_chunks(query_vector, embedded, top_k=top_k)
        hits = tuple(
            VersionHit(
                rank=rank,
                score=result.score,
                item=by_chunk_id[result.embedded_chunk.chunk.chunk_id],
            )
            for rank, result in enumerate(retrieved, start=1)
        )
        results.append(QueryResult(query, hits, summarize_hits(hits)))

    return tuple(results)


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 6)


def serialize_query_result(
    result: QueryResult,
    corpus: VersionedCorpus,
    *,
    run_id: str,
    level: str,
    model: str,
) -> dict[str, object]:
    """Create an auditable record without vectors or source content."""
    summary = result.summary
    return {
        "run_id": run_id,
        "embed_model": model,
        "level": level,
        "previous": {
            "revision": corpus.previous.revision,
            "commit": corpus.previous.commit,
        },
        "current": {
            "revision": corpus.current.revision,
            "commit": corpus.current.commit,
        },
        "query": result.query,
        "summary": {
            "top_result_is_current": summary.top_result_is_current,
            "outdated_in_top_k": summary.outdated_in_top_k,
            "best_current_score": _rounded(summary.best_current_score),
            "best_outdated_score": _rounded(summary.best_outdated_score),
            "score_gap": _rounded(summary.score_gap),
        },
        "results": [
            {
                "rank": hit.rank,
                "score": _rounded(hit.score),
                "status": hit.item.status,
                "revision": hit.item.revision,
                "commit": hit.item.commit,
                "path": hit.item.chunk.path,
                "kind": hit.item.chunk.kind,
                "symbol": hit.item.chunk.symbol,
                "start_line": hit.item.chunk.start_line,
                "end_line": hit.item.chunk.end_line,
                "chunk_id": hit.item.chunk.chunk_id,
                "content_hash": hit.item.chunk.content_hash,
            }
            for hit in result.hits
        ],
    }


def append_records(path: Path, records: Sequence[dict[str, object]]) -> None:
    """Append one JSON line per query for later evaluation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, sort_keys=True) + "\n")