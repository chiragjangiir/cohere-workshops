"""Build a deterministic structural map of a repository."""

from dataclasses import dataclass
from pathlib import Path

from cohere_workshops.documents import load_document
from cohere_workshops.scanner import ExcludedFile, scan_repository
from cohere_workshops.structure import CodeSymbol, analyze_python


@dataclass(frozen=True)
class MappedFile:
    """Metadata and structural evidence for one repository file."""

    path: str
    language: str
    content_hash: str
    symbols: tuple[CodeSymbol, ...]
    imports: tuple[str, ...]


@dataclass(frozen=True)
class ProcessingError:
    """A reproducible file-processing failure."""

    path: str
    stage: str
    reason: str


@dataclass(frozen=True)
class RepositoryMap:
    """A deterministic map of the supported files in a repository."""

    root: str
    files: tuple[MappedFile, ...]
    exclusions: tuple[ExcludedFile, ...]
    errors: tuple[ProcessingError, ...]


def map_repository(repository: Path) -> RepositoryMap:
    """Scan and structurally map a repository without using AI."""
    repository = repository.resolve()
    scan_result = scan_repository(repository)
    mapped_files: list[MappedFile] = []
    errors: list[ProcessingError] = []

    for path in scan_result.files:
        relative_path = path.relative_to(repository).as_posix()

        try:
            document = load_document(path, repository)
        except UnicodeDecodeError:
            errors.append(
                ProcessingError(relative_path, "load", "invalid_utf8")
            )
            continue

        symbols: tuple[CodeSymbol, ...] = ()
        imports: tuple[str, ...] = ()

        if document.language == "python":
            try:
                structure = analyze_python(document)
            except SyntaxError:
                errors.append(
                    ProcessingError(
                        document.path,
                        "structure",
                        "invalid_python_syntax",
                    )
                )
            else:
                symbols = structure.symbols
                imports = structure.imports

        mapped_files.append(
            MappedFile(
                path=document.path,
                language=document.language,
                content_hash=document.content_hash,
                symbols=symbols,
                imports=imports,
            )
        )

    return RepositoryMap(
        root=repository.name,
        files=tuple(mapped_files),
        exclusions=scan_result.exclusions,
        errors=tuple(errors),
    )
