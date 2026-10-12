"""Create stable repository chunks for semantic retrieval."""

from dataclasses import dataclass
from hashlib import sha256

from cohere_workshops.documents import RepositoryDocument
from cohere_workshops.safety import detect_secret_types
from cohere_workshops.structure import analyze_python


@dataclass(frozen=True)
class RepositoryChunk:
    """A traceable unit of repository content."""

    chunk_id: str
    path: str
    language: str
    kind: str
    symbol: str | None
    start_line: int
    end_line: int
    content: str
    content_hash: str
    source_hash: str


def _create_chunk(
    document: RepositoryDocument,
    *,
    kind: str,
    symbol: str | None,
    start_line: int,
    end_line: int,
    content: str,
) -> RepositoryChunk:
    content_hash = sha256(content.encode("utf-8")).hexdigest()
    identity = "|".join(
        (
            document.path,
            document.content_hash,
            kind,
            symbol or "",
            str(start_line),
            str(end_line),
            content_hash,
        )
    )
    chunk_id = sha256(identity.encode("utf-8")).hexdigest()

    return RepositoryChunk(
        chunk_id=chunk_id,
        path=document.path,
        language=document.language,
        kind=kind,
        symbol=symbol,
        start_line=start_line,
        end_line=end_line,
        content=content,
        content_hash=content_hash,
        source_hash=document.content_hash,
    )


def chunk_document(document: RepositoryDocument) -> tuple[RepositoryChunk, ...]:
    """Create deterministic file and symbol chunks from one document."""
    if detect_secret_types(document.content):
        raise ValueError(f"Refusing to chunk suspected secret: {document.path}")

    lines = document.content.splitlines()
    file_end_line = max(len(lines), 1)
    chunks = [
        _create_chunk(
            document,
            kind="file",
            symbol=None,
            start_line=1,
            end_line=file_end_line,
            content=document.content,
        )
    ]

    if document.language != "python":
        return tuple(chunks)

    structure = analyze_python(document)

    for symbol in structure.symbols:
        content = "\n".join(lines[symbol.line - 1 : symbol.end_line])
        chunks.append(
            _create_chunk(
                document,
                kind=symbol.kind,
                symbol=symbol.qualified_name,
                start_line=symbol.line,
                end_line=symbol.end_line,
                content=content,
            )
        )

    return tuple(chunks)
