"""Discover source files inside a repository."""

from dataclasses import dataclass
from pathlib import Path

MAX_FILE_SIZE_BYTES = 1_000_000

SENSITIVE_FILE_NAMES = {
    ".env",
    "credentials.json",
    "secrets.json",
    "secrets.yaml",
    "secrets.yml",
    "service-account.json",
}

IGNORED_DIRECTORIES = {
    ".git",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    ".pytest_cache",
    ".ruff_cache",
}

SUPPORTED_EXTENSIONS = {
    ".js",
    ".jsx",
    ".json",
    ".md",
    ".py",
    ".sol",
    ".toml",
    ".ts",
    ".tsx",
    ".yaml",
    ".yml",
}


@dataclass(frozen=True)
class ExcludedFile:
    """A file excluded from analysis with an auditable reason."""

    path: str
    reason: str


@dataclass(frozen=True)
class ScanResult:
    """Supported files and recorded safety exclusions."""

    files: tuple[Path, ...]
    exclusions: tuple[ExcludedFile, ...]


def scan_repository(repository: Path) -> ScanResult:
    """Scan a repository and record safety-related exclusions."""
    repository = repository.resolve()

    if not repository.is_dir():
        raise ValueError(f"Repository does not exist: {repository}")

    files: list[Path] = []
    exclusions: list[ExcludedFile] = []

    for path in repository.rglob("*"):
        if not path.is_file():
            continue

        relative_path = path.relative_to(repository)

        if any(part in IGNORED_DIRECTORIES for part in relative_path.parts):
            continue

        if path.name.lower() in SENSITIVE_FILE_NAMES:
            exclusions.append(
                ExcludedFile(relative_path.as_posix(), "sensitive_path")
            )
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        if path.stat().st_size > MAX_FILE_SIZE_BYTES:
            exclusions.append(
                ExcludedFile(relative_path.as_posix(), "file_too_large")
            )
            continue

        files.append(path)

    return ScanResult(
        files=tuple(sorted(files)),
        exclusions=tuple(sorted(exclusions, key=lambda item: item.path)),
    )


def discover_source_files(repository: Path) -> list[Path]:
    """Return supported repository files in deterministic order."""
    return list(scan_repository(repository).files)
