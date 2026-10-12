"""Build a safe, traceable chunk corpus from a repository."""

from dataclasses import dataclass
from pathlib import Path

from cohere_workshops.chunks import RepositoryChunk, chunk_document
from cohere_workshops.documents import load_document
from cohere_workshops.safety import detect_secret_types
from cohere_workshops.scanner import ExcludedFile, scan_repository


@dataclass(frozen=True)
class CorpusError:
    """A reproducible failure encountered while building a corpus."""

    path: str
    stage: str
    reason: str


@dataclass(frozen=True)
class RepositoryCorpus:
    """Repository chunks plus all exclusions and processing errors."""

    root: str
    chunks: tuple[RepositoryChunk, ...]
    exclusions: tuple[ExcludedFile, ...]
    errors: tuple[CorpusError, ...]


def build_repository_corpus(repository: Path) -> RepositoryCorpus:
    """Build a safe chunk corpus without making external API calls."""
    repository = repository.resolve()
    scan_result = scan_repository(repository)
    chunks: list[RepositoryChunk] = []
    exclusions = list(scan_result.exclusions)
    errors: list[CorpusError] = []

    for path in scan_result.files:
        relative_path = path.relative_to(repository).as_posix()

        try:
            document = load_document(path, repository)
        except UnicodeDecodeError:
            errors.append(
                CorpusError(relative_path, "load", "invalid_utf8")
            )
            continue

        if detect_secret_types(document.content):
            exclusions.append(
                ExcludedFile(document.path, "suspected_secret")
            )
            continue

        try:
            chunks.extend(chunk_document(document))
        except SyntaxError:
            errors.append(
                CorpusError(
                    document.path,
                    "chunk",
                    "invalid_python_syntax",
                )
            )

    return RepositoryCorpus(
        root=repository.name,
        chunks=tuple(chunks),
        exclusions=tuple(
            sorted(exclusions, key=lambda item: item.path)
        ),
        errors=tuple(errors),
    )
