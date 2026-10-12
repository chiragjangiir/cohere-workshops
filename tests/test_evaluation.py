import pytest

from cohere_workshops.evaluation import (
    deduplicate_paths,
    evaluate_file_ranking,
)


def test_deduplicates_chunk_paths_without_changing_rank() -> None:
    assert deduplicate_paths(
        ("scanner.py", "scanner.py", "test_scanner.py")
    ) == ("scanner.py", "test_scanner.py")


def test_calculates_file_level_ranking_metrics() -> None:
    metrics = evaluate_file_ranking(
        (
            "src/scanner.py",
            "src/scanner.py",
            "tests/test_scanner.py",
            "README.md",
        ),
        {
            "src/scanner.py",
            "tests/test_scanner.py",
        },
        k_values=(1, 3),
    )

    assert metrics.predicted_files == (
        "src/scanner.py",
        "tests/test_scanner.py",
        "README.md",
    )
    assert metrics.recall_at_k == {1: 0.5, 3: 1.0}
    assert metrics.precision_at_k == {1: 1.0, 3: pytest.approx(2 / 3)}
    assert metrics.reciprocal_rank == 1.0
    assert metrics.average_precision == 1.0


def test_scores_missing_relevant_files_as_zero() -> None:
    metrics = evaluate_file_ranking(
        ("README.md",),
        {"src/scanner.py"},
        k_values=(1,),
    )

    assert metrics.recall_at_k == {1: 0.0}
    assert metrics.precision_at_k == {1: 0.0}
    assert metrics.reciprocal_rank == 0.0
    assert metrics.average_precision == 0.0


def test_requires_non_empty_ground_truth() -> None:
    with pytest.raises(ValueError, match="ground truth"):
        evaluate_file_ranking(("README.md",), ())
