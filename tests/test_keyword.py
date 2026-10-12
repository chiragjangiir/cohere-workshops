from cohere_workshops.chunks import RepositoryChunk
from cohere_workshops.keyword import keyword_retrieve, tokenize


def make_chunk(name: str, content: str) -> RepositoryChunk:
    return RepositoryChunk(
        chunk_id=name,
        path=f"src/{name}.py",
        language="python",
        kind="file",
        symbol=None,
        start_line=1,
        end_line=1,
        content=content,
        content_hash=f"{name}-content",
        source_hash=f"{name}-source",
    )


def test_tokenizes_snake_case_and_camel_case_identifiers() -> None:
    assert tokenize("scan_repository fileScanner") == (
        "scan",
        "repository",
        "file",
        "scanner",
    )


def test_keyword_baseline_ranks_lexical_matches() -> None:
    chunks = (
        make_chunk("billing", "calculate invoice totals"),
        make_chunk(
            "scanner",
            "scan repository files and apply git ignore rules",
        ),
        make_chunk("chat", "send a conversational message"),
    )

    results = keyword_retrieve(
        "respect gitignore rules during repository scanning",
        chunks,
        top_k=2,
    )

    assert results[0].chunk.chunk_id == "scanner"
    assert results[0].score > 0
    assert all(result.chunk.chunk_id != "billing" for result in results)


def test_keyword_baseline_returns_no_unrelated_results() -> None:
    chunks = (make_chunk("billing", "calculate invoice totals"),)

    assert keyword_retrieve("repository scanner", chunks) == ()
