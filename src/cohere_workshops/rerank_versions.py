"""Rerank versioned code candidates and check them against Git labels."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from cohere_workshops.reranking import RERANK_MODEL, RerankClient
from cohere_workshops.version_search import VersionHit
from cohere_workshops.versions import CURRENT, VersionedChunk

FORMATS = ("code", "metadata")


@dataclass(frozen=True)
class RankedCandidate:
    """One candidate in a ranking, with where it started."""

    rank: int
    score: float
    retrieval_rank: int
    item: VersionedChunk


@dataclass(frozen=True)
class RankingCheck:
    """Deterministic Git check of one ranking."""

    top_result_is_current: bool
    current_rank: int | None
    best_stale_rank: int | None
    stale_above_current: bool
    current_in_top_k: bool
    stale_rate_at_k: float


def is_current(item: VersionedChunk) -> bool:
    """Git decides validity: only code from the current revision is valid."""
    return item.status == CURRENT


def format_candidate(
    item: VersionedChunk,
    style: str,
    commit_dates: Mapping[str, str] | None = None,
) -> str:
    """Build the text Rerank reads, with or without Git metadata."""
    if style not in FORMATS:
        raise ValueError(f"style must be one of {FORMATS}")

    chunk = item.chunk
    lines = [f"path: {chunk.path}", f"symbol: {chunk.symbol or '<file>'}"]

    if style == "metadata":
        dates = commit_dates or {}
        lines += [
            f"revision: {item.revision}",
            f"commit: {item.commit[:12]}",
            f"commit date: {dates.get(item.commit, 'unknown')}",
        ]

    return "\n".join(lines) + "\n\n" + chunk.content


def retrieval_order(hits: Sequence[VersionHit]) -> tuple[RankedCandidate, ...]:
    """Keep the Embed retrieval order, so it can be compared with Rerank."""
    return tuple(
        RankedCandidate(hit.rank, hit.score, hit.rank, hit.item) for hit in hits
    )


def rerank_hits(
    client: RerankClient,
    query: str,
    hits: Sequence[VersionHit],
    *,
    style: str = "code",
    commit_dates: Mapping[str, str] | None = None,
    model: str = RERANK_MODEL,
) -> tuple[RankedCandidate, ...]:
    """Reorder retrieved candidates by Cohere Rerank relevance."""
    if not query.strip():
        raise ValueError("query must not be empty")
    if not hits:
        return ()

    response = client.rerank(
        model=model,
        query=query,
        documents=[format_candidate(hit.item, style, commit_dates) for hit in hits],
        top_n=len(hits),
    )
    results = getattr(response, "results", None)
    if results is None:
        raise ValueError("Cohere response did not contain rerank results")

    ranked: list[RankedCandidate] = []
    seen: set[int] = set()

    for result in results:
        index = int(result.index)
        if index < 0 or index >= len(hits):
            raise ValueError("Cohere returned an invalid document index")
        if index in seen:
            raise ValueError("Cohere returned a duplicate document index")
        seen.add(index)
        hit = hits[index]
        ranked.append(
            RankedCandidate(
                rank=len(ranked) + 1,
                score=float(result.relevance_score),
                retrieval_rank=hit.rank,
                item=hit.item,
            )
        )

    if len(ranked) != len(hits):
        raise ValueError("Cohere returned a different number of results than sent")

    return tuple(ranked)


def check_ranking(
    ranking: Sequence[RankedCandidate],
    *,
    k: int = 3,
) -> RankingCheck:
    """Compare where current and stale code landed in one ranking."""
    if not ranking:
        raise ValueError("ranking must not be empty")
    if k < 1:
        raise ValueError("k must be at least 1")

    current_ranks = [c.rank for c in ranking if is_current(c.item)]
    stale_ranks = [c.rank for c in ranking if not is_current(c.item)]
    current_rank = min(current_ranks) if current_ranks else None
    best_stale_rank = min(stale_ranks) if stale_ranks else None
    top_k = ranking[:k]

    return RankingCheck(
        top_result_is_current=is_current(ranking[0].item),
        current_rank=current_rank,
        best_stale_rank=best_stale_rank,
        stale_above_current=(
            best_stale_rank is not None
            and (current_rank is None or best_stale_rank < current_rank)
        ),
        current_in_top_k=any(is_current(c.item) for c in top_k),
        stale_rate_at_k=sum(not is_current(c.item) for c in top_k) / len(top_k),
    )


def temporal_filter(ranking: Sequence[RankedCandidate]) -> tuple[RankedCandidate, ...]:
    """After Rerank, drop stale code and renumber what is left."""
    kept = [c for c in ranking if is_current(c.item)]
    return tuple(
        RankedCandidate(rank, c.score, c.retrieval_rank, c.item)
        for rank, c in enumerate(kept, start=1)
    )


def _check_record(check: RankingCheck) -> dict[str, object]:
    return {
        "top_result_is_current": check.top_result_is_current,
        "current_rank": check.current_rank,
        "best_stale_rank": check.best_stale_rank,
        "stale_above_current": check.stale_above_current,
        "current_in_top_k": check.current_in_top_k,
        "stale_rate_at_k": round(check.stale_rate_at_k, 6),
    }


def _ranking_record(ranking: Sequence[RankedCandidate]) -> list[dict[str, object]]:
    return [
        {
            "rank": c.rank,
            "score": round(c.score, 6),
            "retrieval_rank": c.retrieval_rank,
            "status": c.item.status,
            "revision": c.item.revision,
            "commit": c.item.commit,
            "path": c.item.chunk.path,
            "symbol": c.item.chunk.symbol,
            "chunk_id": c.item.chunk.chunk_id,
        }
        for c in ranking
    ]


def serialize_rerank_result(
    *,
    run_id: str,
    query: str,
    style: str,
    k: int,
    embed_model: str,
    rerank_model: str,
    previous: tuple[str, str],
    current: tuple[str, str],
    retrieval: Sequence[RankedCandidate],
    reranked: Sequence[RankedCandidate],
) -> dict[str, object]:
    """Create one auditable record without source code or vectors."""
    filtered = temporal_filter(reranked)
    return {
        "run_id": run_id,
        "query": query,
        "format": style,
        "k": k,
        "embed_model": embed_model,
        "rerank_model": rerank_model,
        "previous": {"revision": previous[0], "commit": previous[1]},
        "current": {"revision": current[0], "commit": current[1]},
        "retrieval_check": _check_record(check_ranking(retrieval, k=k)),
        "rerank_check": _check_record(check_ranking(reranked, k=k)),
        "filtered_top": _ranking_record(filtered[:1]),
        "retrieval": _ranking_record(retrieval),
        "reranked": _ranking_record(reranked),
    }
