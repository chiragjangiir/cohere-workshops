"""Write validated dataset manifests atomically."""

import json
from dataclasses import asdict
from pathlib import Path

from cohere_workshops.dataset import (
    DatasetManifest,
    EvaluationCase,
    load_dataset_manifest,
)


def write_dataset_manifest(
    path: Path,
    manifest: DatasetManifest,
) -> DatasetManifest:
    """Validate and atomically write one dataset manifest."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    payload = {
        "dataset_version": manifest.dataset_version,
        "cases": [asdict(case) for case in manifest.cases],
    }

    try:
        temporary.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        validated = load_dataset_manifest(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)

    return validated


def append_dataset_case(
    path: Path,
    case: EvaluationCase,
    *,
    dataset_version: str,
) -> DatasetManifest:
    """Append one case while preserving full manifest validation."""
    if path.exists():
        current = load_dataset_manifest(path)
        if current.dataset_version != dataset_version:
            raise ValueError("dataset version does not match existing manifest")
        cases = (*current.cases, case)
    else:
        cases = (case,)

    return write_dataset_manifest(
        path,
        DatasetManifest(
            dataset_version=dataset_version,
            cases=cases,
        ),
    )