import json
from dataclasses import dataclass

import pytest

from cohere_workshops.chunks import RepositoryChunk
from cohere_workshops.embeddings import EmbeddedChunk
from cohere_workshops.prediction import (
    COMMAND_MODEL,
    predict_change_impact,
)
from cohere_workshops.reranking import RerankedChunk
from cohere_workshops.retrieval import RetrievedChunk


@dataclass
class FakeText:
    text: str


@dataclass
class FakeThinking:
    thinking: str = "internal reasoning"


@dataclass
class FakeMessage:
    content: list[object]


@dataclass
class FakeResponse:
    message: FakeMessage


class FakeClient:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls: list[dict[str, object]] = []

    def chat(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return FakeResponse(
            FakeMessage(
                [
                    FakeThinking(),
                    FakeText(json.dumps(self.payload)),
                ]
            )
        )


def make_evidence() -> tuple[RerankedChunk, ...]:
    chunk = RepositoryChunk(
        chunk_id="scanner-chunk",
        path="src/cohere_workshops/scanner.py",
        language="python",
        kind="function",
        symbol="scan_repository",
        start_line=59,
        end_line=98,
        content="def scan_repository(repository):\n    pass",
        content_hash="chunk-hash",
        source_hash="source-hash",
    )
    embedded = EmbeddedChunk(
        chunk=chunk,
        vector=(1.0, 0.0),
        model="embed-v4.0",
    )
    retrieved = RetrievedChunk(
        embedded_chunk=embedded,
        score=0.8,
    )
    return (
        RerankedChunk(
            retrieved_chunk=retrieved,
            rerank_score=0.9,
            model="rerank-v4.0-pro",
            original_rank=1,
        ),
    )


def test_generates_validated_grounded_prediction() -> None:
    client = FakeClient(
        {
            "abstain": False,
            "abstention_reason": "",
            "predictions": [
                {
                    "path": "src/cohere_workshops/scanner.py",
                    "symbols": ["scan_repository"],
                    "evidence_chunk_ids": ["scanner-chunk"],
                    "confidence": 0.9,
                    "reason": "The scanner owns repository discovery.",
                }
            ],
            "predicted_tests": [],
            "unresolved_questions": [],
        }
    )

    result = predict_change_impact(
        client,
        "Respect Git ignore rules",
        make_evidence(),
    )

    assert result.abstain is False
    assert result.model == COMMAND_MODEL
    assert result.predictions[0].path == (
        "src/cohere_workshops/scanner.py"
    )
    assert result.predictions[0].symbols == ("scan_repository",)
    call = client.calls[0]
    assert call["model"] == COMMAND_MODEL
    assert call["temperature"] == 0
    assert call["response_format"]["type"] == "json_object"


def test_rejects_hallucinated_file_path() -> None:
    client = FakeClient(
        {
            "abstain": False,
            "abstention_reason": "",
            "predictions": [
                {
                    "path": "src/invented.py",
                    "symbols": [],
                    "evidence_chunk_ids": ["scanner-chunk"],
                    "confidence": 0.7,
                    "reason": "Invented path.",
                }
            ],
            "predicted_tests": [],
            "unresolved_questions": [],
        }
    )

    with pytest.raises(ValueError, match="unsupported file path"):
        predict_change_impact(
            client,
            "Respect Git ignore rules",
            make_evidence(),
        )


def test_abstains_without_repository_evidence() -> None:
    client = FakeClient({})

    result = predict_change_impact(
        client,
        "Respect Git ignore rules",
        (),
    )

    assert result.abstain is True
    assert result.predictions == ()
    assert client.calls == []
