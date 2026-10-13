"""Command-line interface for building one evaluation case."""

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from cohere_workshops.case_builder import build_evaluation_case
from cohere_workshops.manifest_writer import append_dataset_case


def build_parser() -> argparse.ArgumentParser:
    """Create the evaluation-case command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Build one research evaluation case from an exact Git "
            "commit range."
        )
    )
    parser.add_argument("repository", help="Local Git repository.")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--repository-url", required=True)
    parser.add_argument("--pull-request", required=True, type=int)
    parser.add_argument("--pre-change-commit", required=True)
    parser.add_argument("--post-change-commit", required=True)
    parser.add_argument("--change-request", required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Append the case to this dataset manifest.",
    )
    parser.add_argument(
        "--dataset-version",
        default="0.1.0",
        help="Dataset version used with --manifest.",
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> None:
    """Build one case, optionally append it, and print JSON."""
    options = build_parser().parse_args(arguments)
    case = build_evaluation_case(
        Path(options.repository),
        case_id=options.case_id,
        repository_url=options.repository_url,
        pull_request=options.pull_request,
        pre_change_commit=options.pre_change_commit,
        post_change_commit=options.post_change_commit,
        change_request=options.change_request,
    )

    if options.manifest is None:
        payload = asdict(case)
    else:
        manifest = append_dataset_case(
            options.manifest,
            case,
            dataset_version=options.dataset_version,
        )
        payload = {
            "manifest": str(options.manifest),
            "dataset_version": manifest.dataset_version,
            "case_count": len(manifest.cases),
            "added_case_id": case.case_id,
        }

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()