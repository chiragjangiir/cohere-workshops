"""Create stable repository chunks for semantic retrieval."""

from dataclasses import dataclass
from hashlib import sha256

from cohere_workshops.documents import RepositoryDocument
from cohere_workshops.safety import detect_secret_types
from cohere_workshops.structure import analyze_python

MAX_CHUNK_CHARACTERS = 12_000


@dataclass(frozen=True)
class RepositoryChunk:
    """A traceable, size-bounded unit of repository content."""

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
    part_index: int = 1
    part_count: int = 1


def _split_content(
    content: str,
    *,
    start_line: int,
    max_characters: int,
) -> tuple[tuple[int, int, str], ...]:
    """Split content while preserving every character and source line."""
    if max_characters < 1:
        raise ValueError("max_characters must be at least 1")

    if not content:
        return ((start_line, start_line, ""),)

    source_lines = content.splitlines(keepends=True)
    segments: list[tuple[int, int, str]] = []
    current: list[str] = []
    current_length = 0
    current_start = start_line
    current_end = start_line

    def flush() -> None:
        nonlocal current, current_length

        if current:
            segments.append(
                (current_start, current_end, "".join(current))
            )
            current = []
            current_length = 0

    for offset, source_line in enumerate(source_lines):
        line_number = start_line + offset
        remaining = source_line

        while len(remaining) > max_characters:
            flush()
            segments.append(
                (
                    line_number,
                    line_number,
                    remaining[:max_characters],
                )
            )
            remaining = remaining[max_characters:]

        if (
            current
            and current_length + len(remaining) > max_characters
        ):
            flush()

        if remaining:
            if not current:
                current_start = line_number
            current.append(remaining)
            current_length += len(remaining)
            current_end = line_number

    flush()
    return tuple(segments)


def _create_chunk(
    document: RepositoryDocument,
    *,
    kind: str,
    symbol: str | None,
    start_line: int,
    end_line: int,
    content: str,
    part_index: int,
    part_count: int,
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
            str(part_index),
            str(part_count),
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
        part_index=part_index,
        part_count=part_count,
    )


def _create_bounded_chunks(
    document: RepositoryDocument,
    *,
    kind: str,
    symbol: str | None,
    start_line: int,
    content: str,
    max_characters: int,
) -> tuple[RepositoryChunk, ...]:
    segments = _split_content(
        content,
        start_line=start_line,
        max_characters=max_characters,
    )
    part_count = len(segments)

    return tuple(
        _create_chunk(
            document,
            kind=kind,
            symbol=symbol,
            start_line=segment_start,
            end_line=segment_end,
            content=segment_content,
            part_index=index,
            part_count=part_count,
        )
        for index, (
            segment_start,
            segment_end,
            segment_content,
        ) in enumerate(segments, start=1)
    )


def chunk_document(
    document: RepositoryDocument,
    *,
    max_characters: int = MAX_CHUNK_CHARACTERS,
) -> tuple[RepositoryChunk, ...]:
    """Create deterministic, size-bounded file and symbol chunks."""
    if detect_secret_types(document.content):
        raise ValueError(f"Refusing to chunk suspected secret: {document.path}")

    source_lines = document.content.splitlines(keepends=True)
    chunks = list(
        _create_bounded_chunks(
            document,
            kind="file",
            symbol=None,
            start_line=1,
            content=document.content,
            max_characters=max_characters,
        )
    )

    if document.language != "python":
        return tuple(chunks)

    structure = analyze_python(document)

    for symbol in structure.symbols:
        content = "".join(
            source_lines[symbol.line - 1 : symbol.end_line]
        )
        chunks.extend(
            _create_bounded_chunks(
                document,
                kind=symbol.kind,
                symbol=symbol.qualified_name,
                start_line=symbol.line,
                content=content,
                max_characters=max_characters,
            )
        )

    return tuple(chunks)
