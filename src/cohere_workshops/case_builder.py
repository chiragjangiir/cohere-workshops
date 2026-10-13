"""Build evaluation cases from exact Git commit ranges."""

from pathlib import Path

from cohere_workshops.dataset import EvaluationCase
from cohere_workshops.git_ground_truth import extract_diff_ground_truth


def build_evaluation_case(
    repository: Path,
    *,
    case_id: str,
    repository_url: str,
    pull_request: int,
    pre_change_commit: str,
    post_change_commit: str,
    change_request: str,
) -> EvaluationCase:
    """Create one evaluation case with diff-derived ground truth."""
    if not case_id.strip():
        raise ValueError("case_id must be a non-empty string")
    if not repository_url.startswith("https://github.com/"):
        raise ValueError("repository_url must be a public GitHub HTTPS URL")
    if (
        not isinstance(pull_request, int)
        or isinstance(pull_request, bool)
        or pull_request < 1
    ):
        raise ValueError("pull_request must be a positive integer")
    if not change_request.strip():
        raise ValueError("change_request must be a non-empty string")

    ground_truth = extract_diff_ground_truth(
        repository,
        pre_change_commit,
        post_change_commit,
    )

    return EvaluationCase(
        case_id=case_id,
        repository_url=repository_url,
        pull_request=pull_request,
        pre_change_commit=ground_truth.pre_change_commit,
        post_change_commit=ground_truth.post_change_commit,
        change_request=change_request,
        ground_truth_files=ground_truth.files,
        ground_truth_test_files=ground_truth.test_files,
    )