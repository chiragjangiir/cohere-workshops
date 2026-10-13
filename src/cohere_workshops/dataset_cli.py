"""Validate evaluation dataset manifests and Git ground truth."""

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from cohere_workshops.dataset import load_dataset_manifest
from cohere_workshops.git_ground_truth import verify_case_ground_truth


def build_parser() -> argparse.ArgumentParser:
    """Create the dataset validation parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate a research dataset manifest and optionally "
            "verify one case against a local Git repository."
        )
    )
    parser.add_argument("manifest", help="Path to the dataset manifest.")
    parser.add_argument("--case-id", help="Case identifier to verify.")
    parser.add_argument(
        "--repository",
        help="Local repository containing both recorded commits.",
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> None:
    """Validate a manifest and print a machine-readable summary."""
    options = build_parser().parse_args(arguments)
    manifest = load_dataset_manifest(Path(options.manifest))

    if bool(options.case_id) != bool(options.repository):
        raise ValueError(
            "--case-id and --repository must be provided together"
        )

    payload: dict[str, object] = {
        "dataset_version": manifest.dataset_version,
        "case_count": len(manifest.cases),
        "case_ids": [case.case_id for case in manifest.cases],
        "schema_valid": True,
    }

    if options.case_id:
        matches = [
            case for case in manifest.cases if case.case_id == options.case_id
        ]
        if not matches:
            raise ValueError(f"Unknown case_id: {options.case_id}")
        ground_truth = verify_case_ground_truth(
            Path(options.repository),
            matches[0],
        )
        payload["verified_case"] = options.case_id
        payload["git_ground_truth"] = asdict(ground_truth)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
