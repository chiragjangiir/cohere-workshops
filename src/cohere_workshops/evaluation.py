"""Evaluate ranked file predictions against pull-request ground truth."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RankingMetrics:
    """File-level metrics for one evaluation case."""

    predicted_files: tuple[str, ...]
    ground_truth_files: tuple[str, ...]
    recall_at_k: dict[int, float]
    precision_at_k: dict[int, float]
    reciprocal_rank: float
    average_precision: float


def deduplicate_paths(paths: Iterable[str]) -> tuple[str, ...]:
    """Preserve the first occurrence of every predicted file."""
    seen: set[str] = set()
    unique: list[str] = []

    for path in paths:
        if path not in seen:
            seen.add(path)
            unique.append(path)

    return tuple(unique)


def evaluate_file_ranking(
    predicted_paths: Sequence[str],
    ground_truth_paths: Iterable[str],
    *,
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> RankingMetrics:
    """Calculate deterministic file-ranking metrics for one case."""
    if any(k < 1 for k in k_values):
        raise ValueError("all K values must be at least 1")

    predictions = deduplicate_paths(predicted_paths)
    ground_truth = frozenset(ground_truth_paths)

    if not ground_truth:
        raise ValueError("ground truth must contain at least one file")

    recall_at_k: dict[int, float] = {}
    precision_at_k: dict[int, float] = {}

    for k in k_values:
        relevant = sum(
            path in ground_truth
            for path in predictions[:k]
        )
        recall_at_k[k] = relevant / len(ground_truth)
        precision_at_k[k] = relevant / k

    relevant_ranks = [
        rank
        for rank, path in enumerate(predictions, start=1)
        if path in ground_truth
    ]
    reciprocal_rank = (
        1 / relevant_ranks[0]
        if relevant_ranks
        else 0.0
    )

    precision_sum = 0.0
    relevant_seen = 0

    for rank, path in enumerate(predictions, start=1):
        if path in ground_truth:
            relevant_seen += 1
            precision_sum += relevant_seen / rank

    average_precision = precision_sum / len(ground_truth)

    return RankingMetrics(
        predicted_files=predictions,
        ground_truth_files=tuple(sorted(ground_truth)),
        recall_at_k=recall_at_k,
        precision_at_k=precision_at_k,
        reciprocal_rank=reciprocal_rank,
        average_precision=average_precision,
    )
