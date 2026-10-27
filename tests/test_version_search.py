import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from cohere_workshops.chunks import RepositoryChunk
from cohere_workshops.version_search import (
    append_records,
    count_unchanged,
    search_versions,
    select_level,
    serialize_query_result,
)
from cohere_workshops.versions import (
    CURRENT,
    REMOVED,
    SUPERSEDED,
    RevisionSnapshot,
    VersionedChunk,
    VersionedCorpus,
)

OLD_COMMIT = "a" * 40
NEW_COMMIT = "b" * 40


@dataclass
class FakeEmbeddings:
    float: list[list[float]]


@dataclass
class FakeResponse:
    embeddings: FakeEmbeddings


# The first marker found in a text decides its vector.
DOCUMENT_VECTORS = (
    ("whole file", (0.5, 0.5, 0.5)),
    ("flat", (1.0, 0.0, 0.0)),
    ("tiered", (0.9, 0.1, 0.0)),
    ("validate", (0.0, 1.0, 0.0)),
    ("receipt", (0.0, 0.0, 1.0)),
)
QUERY_VECTORS = (("fee", (1.0, 0.0, 0.0)), ("phone", (0.0, 1.0, 0.0)))


class KeywordClient:
    """Fake Embed client with fixed vectors, so no API credits are spent."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def _vector(self, text: str) -> list[float]:
        for query, vector in QUERY_VECTORS:
            if text == query:
                return list(vector)
        for marker, vector in DOCUMENT_VECTORS:
            if marker in text:
                return list(vector)
        return [0.1, 0.1, 0.1]

    def embed(self, **kwargs: object) -> FakeResponse:
        self.calls.append(kwargs)
        texts = kwargs["texts"]
        assert isinstance(texts, list)
        return FakeResponse(FakeEmbeddings([self._vector(t) for t in texts]))


def make_item(
    chunk_id: str,
    content: str,
    status: str,
    *,
    kind: str = "function",
    symbol: str | None = "run",
) -> VersionedChunk:
    revision, commit = ("v2", NEW_COMMIT) if status == CURRENT else ("v1", OLD_COMMIT)
    chunk = RepositoryChunk(
        chunk_id=chunk_id,
        path="payments.py",
        language="python",
        kind=kind,
        symbol=symbol,
        start_line=1,
        end_line=3,
        content=content,
        content_hash=f"hash-{chunk_id}",
        source_hash=f"source-{revision}",
    )
    return VersionedChunk(chunk, revision, commit, status)


def make_corpus(items: tuple[VersionedChunk, ...]) -> VersionedCorpus:
    return VersionedCorpus(
        previous=RevisionSnapshot("v1", OLD_COMMIT, (), (), ()),
        current=RevisionSnapshot("v2", NEW_COMMIT, (), (), ()),
        chunks=items,
        unchanged=(),
    )


@pytest.fixture
def corpus() -> VersionedCorpus:
    return make_corpus(
        (
            make_item("new-fee", "def fee(): tiered", CURRENT, symbol="fee"),
            make_item("receipt", "def receipt(): pass", CURRENT, symbol="receipt"),
            make_item("old-fee", "def fee(): flat", SUPERSEDED, symbol="fee"),
            make_item("old-phone", "def validate(): pass", REMOVED, symbol="validate"),
            make_item("file", "whole file", CURRENT, kind="file", symbol=None),
        )
    )


def test_reports_when_outdated_code_ranks_first(corpus: VersionedCorpus) -> None:
    [result] = search_versions(KeywordClient(), corpus, ["fee"], top_k=3)

    assert [hit.item.chunk.chunk_id for hit in result.hits][:2] == [
        "old-fee",
        "new-fee",
    ]
    assert result.summary.top_result_is_current is False
    assert result.summary.outdated_in_top_k >= 1
    assert result.summary.score_gap is not None
    assert result.summary.score_gap < 0


def test_embeds_chunks_once_and_each_query_once(corpus: VersionedCorpus) -> None:
    client = KeywordClient()

    search_versions(client, corpus, ["fee", "phone"])

    input_types = [call["input_type"] for call in client.calls]
    assert input_types == ["search_document", "search_query", "search_query"]


def test_symbol_level_excludes_file_chunks(corpus: VersionedCorpus) -> None:
    [result] = search_versions(KeywordClient(), corpus, ["fee"], top_k=10)

    assert all(hit.item.chunk.kind != "file" for hit in result.hits)
    assert len(select_level(corpus.chunks, "file")) == 1


def test_rejects_unknown_level(corpus: VersionedCorpus) -> None:
    with pytest.raises(ValueError, match="level must be one of"):
        search_versions(KeywordClient(), corpus, ["fee"], level="line")


def test_returns_nothing_for_empty_level() -> None:
    corpus = make_corpus((make_item("one", "fee", CURRENT),))

    assert search_versions(KeywordClient(), corpus, ["fee"], level="file") == ()


def test_records_traceable_results_without_source(
    corpus: VersionedCorpus,
    tmp_path: Path,
) -> None:
    [result] = search_versions(KeywordClient(), corpus, ["fee"], top_k=2)
    record = serialize_query_result(
        result,
        corpus,
        run_id="run-1",
        level="symbol",
        model="embed-v4.0",
    )
    output = tmp_path / "results" / "runs.jsonl"

    append_records(output, [record])
    append_records(output, [record])

    lines = output.read_text(encoding="utf-8").splitlines()
    saved = json.loads(lines[0])
    assert len(lines) == 2
    assert saved["previous"]["commit"] == OLD_COMMIT
    assert saved["results"][0]["status"] == SUPERSEDED
    assert "content" not in saved["results"][0]
    assert "def fee" not in lines[0]


def test_counts_unchanged_chunks_per_level() -> None:
    unchanged_file = make_item("old-file", "whole file", SUPERSEDED, kind="file").chunk
    unchanged_symbol = make_item("old-receipt", "receipt", SUPERSEDED).chunk
    corpus = VersionedCorpus(
        previous=RevisionSnapshot("v1", OLD_COMMIT, (), (), ()),
        current=RevisionSnapshot("v2", NEW_COMMIT, (), (), ()),
        chunks=(),
        unchanged=(unchanged_file, unchanged_symbol),
    )

    assert corpus.unchanged_count == 2
    assert count_unchanged(corpus, "symbol") == 1
    assert count_unchanged(corpus, "file") == 1
