import json
from dataclasses import dataclass

import pytest

from cohere_workshops.prediction import COMMAND_MODEL
from cohere_workshops.ungrounded import predict_ungrounded_change_impact


@dataclass
class FakeText:
    text: str


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
            FakeMessage([FakeText(json.dumps(self.payload))])
        )


def test_generates_path_only_ungrounded_prediction() -> None:
    client = FakeClient(
        {
            "abstain": False,
            "abstention_reason": "",
            "predictions": [
                {
                    "path": "src/cohere_workshops/scanner.py",
                    "confidence": 0.6,
                    "reason": "The filename suggests scanner ownership.",
                }
            ],
            "predicted_tests": ["tests/test_scanner.py"],
            "unresolved_questions": [],
        }
    )
    paths = (
        "src/cohere_workshops/scanner.py",
        "tests/test_scanner.py",
    )

    result = predict_ungrounded_change_impact(
        client,
        "Respect Git ignore rules",
        paths,
    )

    assert result.abstain is False
    assert result.model == COMMAND_MODEL
    assert result.predictions[0].path == paths[0]
    assert result.predictions[0].symbols == ()
    assert result.predictions[0].evidence_chunk_ids == ()
    assert result.predicted_tests == (paths[1],)

    call = client.calls[0]
    assert call["temperature"] == 0
    prompt = call["messages"][0]["content"]
    assert paths[0] in prompt
    assert "def scan_repository" not in prompt
    schema = call["response_format"]["schema"]
    prediction_path = schema["properties"]["predictions"][
        "items"
    ]["properties"]["path"]
    assert prediction_path["enum"] == list(paths)


def test_rejects_ungrounded_prediction_outside_repository_paths() -> None:
    client = FakeClient(
        {
            "abstain": False,
            "abstention_reason": "",
            "predictions": [
                {
                    "path": "src/invented.py",
                    "confidence": 0.5,
                    "reason": "Unsupported guess.",
                }
            ],
            "predicted_tests": [],
            "unresolved_questions": [],
        }
    )

    with pytest.raises(ValueError, match="unsupported file path"):
        predict_ungrounded_change_impact(
            client,
            "Respect Git ignore rules",
            ("src/cohere_workshops/scanner.py",),
        )


def test_ungrounded_prediction_abstains_without_paths() -> None:
    client = FakeClient({})

    result = predict_ungrounded_change_impact(
        client,
        "Respect Git ignore rules",
        (),
    )

    assert result.abstain is True
    assert result.predictions == ()
    assert client.calls == []
