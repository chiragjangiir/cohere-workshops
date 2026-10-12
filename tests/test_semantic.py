from dataclasses import dataclass
from pathlib import Path

from cohere_workshops.semantic import semantic_search_repository


@dataclass
class FakeEmbeddings:
    float: list[list[float]]


@dataclass
class FakeEmbedResponse:
    embeddings: FakeEmbeddings


@dataclass
class FakeRerankItem:
    index: int
    relevance_score: float


@dataclass
class FakeRerankResponse:
    results: list[FakeRerankItem]


class FakeClient:
    def __init__(self) -> None:
        self.embed_calls: list[dict[str, object]] = []
        self.rerank_calls: list[dict[str, object]] = []

    def embed(self, **kwargs: object) -> object:
        self.embed_calls.append(kwargs)
        texts = kwargs["texts"]
        assert isinstance(texts, list)
        return FakeEmbedResponse(
            FakeEmbeddings([[1.0, 0.0] for _ in texts])
        )

    def rerank(self, **kwargs: object) -> object:
        self.rerank_calls.append(kwargs)
        top_n = kwargs["top_n"]
        assert isinstance(top_n, int)
        return FakeRerankResponse(
            [
                FakeRerankItem(index=index, relevance_score=1.0 - index / 10)
                for index in range(top_n)
            ]
        )


def test_runs_complete_semantic_retrieval_pipeline(
    tmp_path: Path,
) -> None:
    (tmp_path / "scanner.py").write_text(
        "def scan():\n    return []\n",
        encoding="utf-8",
    )
    client = FakeClient()

    result = semantic_search_repository(
        client,
        tmp_path,
        "Change repository file scanning",
        retrieval_k=2,
        rerank_k=2,
    )

    assert result.root == tmp_path.name
    assert result.chunk_count == 2
    assert len(result.results) == 2
    assert len(client.embed_calls) == 2
    assert client.embed_calls[0]["input_type"] == "search_document"
    assert client.embed_calls[1]["input_type"] == "search_query"
    assert len(client.rerank_calls) == 1
    assert client.rerank_calls[0]["top_n"] == 2
