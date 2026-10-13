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
    assert result.structural_expansion is False
    assert len(result.results) == 2
    assert len(client.embed_calls) == 2
    assert client.embed_calls[0]["input_type"] == "search_document"
    assert client.embed_calls[1]["input_type"] == "search_query"
    assert len(client.rerank_calls) == 1
    assert client.rerank_calls[0]["top_n"] == 2



class StructuralFakeClient(FakeClient):
    def embed(self, **kwargs: object) -> object:
        self.embed_calls.append(kwargs)
        texts = kwargs["texts"]
        assert isinstance(texts, list)

        if kwargs["input_type"] == "search_query":
            vectors = [[1.0, 0.0]]
        else:
            vectors = [
                [1.0, 0.0]
                if "path: src/example/reader.py" in text
                else [0.0, 1.0]
                for text in texts
            ]

        return FakeEmbedResponse(FakeEmbeddings(vectors))


def test_expands_semantic_candidates_with_repository_structure(
    tmp_path: Path,
) -> None:
    package = tmp_path / "src" / "example"
    package.mkdir(parents=True)
    (package / "reader.py").write_text(
        "import example.writer\n\ndef read():\n    pass\n",
        encoding="utf-8",
    )
    (package / "writer.py").write_text(
        "def write():\n    pass\n",
        encoding="utf-8",
    )
    client = StructuralFakeClient()

    result = semantic_search_repository(
        client,
        tmp_path,
        "Update reader behavior",
        retrieval_k=1,
        rerank_k=3,
        structural_expansion=True,
        max_structural_expansions=2,
    )

    assert result.structural_expansion is True
    assert len(result.results) == 3
    assert client.rerank_calls[0]["top_n"] == 3
