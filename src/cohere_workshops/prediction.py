"""Generate and validate evidence-grounded change-impact predictions."""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from cohere_workshops.reranking import RerankedChunk

COMMAND_MODEL = "command-a-plus-05-2026"

PREDICTION_SCHEMA = {
    "type": "object",
    "properties": {
        "abstain": {"type": "boolean"},
        "abstention_reason": {"type": "string"},
        "predictions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "symbols": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "evidence_chunk_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "confidence": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": [
                    "path",
                    "symbols",
                    "evidence_chunk_ids",
                    "confidence",
                    "reason",
                ],
            },
        },
        "predicted_tests": {
            "type": "array",
            "items": {"type": "string"},
        },
        "unresolved_questions": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "abstain",
        "abstention_reason",
        "predictions",
        "predicted_tests",
        "unresolved_questions",
    ],
}


class ChatClient(Protocol):
    """Minimal client interface required by the prediction layer."""

    def chat(self, **kwargs: object) -> object:
        """Return a chat response."""


@dataclass(frozen=True)
class FilePrediction:
    """One evidence-grounded predicted file change."""

    path: str
    symbols: tuple[str, ...]
    evidence_chunk_ids: tuple[str, ...]
    confidence: float
    reason: str


@dataclass(frozen=True)
class ChangeImpactPrediction:
    """Validated structured prediction or explicit abstention."""

    abstain: bool
    abstention_reason: str
    predictions: tuple[FilePrediction, ...]
    predicted_tests: tuple[str, ...]
    unresolved_questions: tuple[str, ...]
    model: str


def _chunk_from_result(result: RerankedChunk):
    return result.retrieved_chunk.embedded_chunk.chunk


def build_prediction_prompt(
    query: str,
    evidence: Sequence[RerankedChunk],
) -> str:
    """Build a prompt containing only retrieved repository evidence."""
    sections = []

    for result in evidence:
        chunk = _chunk_from_result(result)
        sections.append(
            "\n".join(
                (
                    f"chunk_id: {chunk.chunk_id}",
                    f"path: {chunk.path}",
                    f"kind: {chunk.kind}",
                    f"symbol: {chunk.symbol or 'not_applicable'}",
                    f"lines: {chunk.start_line}-{chunk.end_line}",
                    f"rerank_score: {result.rerank_score}",
                    "content:",
                    chunk.content,
                )
            )
        )

    joined_evidence = "\n\n---\n\n".join(sections)
    return f"""Generate a JSON change-impact prediction.

Proposed change:
{query}

Rules:
1. Use only the repository evidence below.
2. Predict only paths that appear in the evidence.
3. Every prediction must cite at least one listed chunk_id.
4. Include only symbols explicitly shown in the evidence.
5. Confidence must be between 0 and 1.
6. If the evidence is insufficient, abstain and explain why.
7. Do not invent files, symbols, tests, or dependencies.

Repository evidence:
{joined_evidence}
"""


def _require_string_list(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) for item in value
    ):
        raise ValueError(f"{field} must be a list of strings")
    return tuple(value)


def _parse_prediction(
    payload: object,
    evidence: Sequence[RerankedChunk],
    model: str,
) -> ChangeImpactPrediction:
    if not isinstance(payload, dict):
        raise TypeError("prediction must be a JSON object")

    required_fields = {
        "abstain",
        "abstention_reason",
        "predictions",
        "predicted_tests",
        "unresolved_questions",
    }
    if not required_fields.issubset(payload):
        raise ValueError("prediction is missing required fields")

    abstain = payload["abstain"]
    abstention_reason = payload["abstention_reason"]
    raw_predictions = payload["predictions"]

    if not isinstance(abstain, bool):
        raise TypeError("abstain must be a boolean")
    if not isinstance(abstention_reason, str):
        raise TypeError("abstention_reason must be a string")
    if not isinstance(raw_predictions, list):
        raise TypeError("predictions must be a list")

    chunks = [_chunk_from_result(result) for result in evidence]
    allowed_paths = {chunk.path for chunk in chunks}
    allowed_chunk_ids = {chunk.chunk_id for chunk in chunks}
    allowed_symbols = {
        chunk.symbol
        for chunk in chunks
        if chunk.symbol is not None
    }

    predictions: list[FilePrediction] = []
    prediction_positions: dict[str, int] = {}

    for raw in raw_predictions:
        if not isinstance(raw, dict):
            raise TypeError("each prediction must be an object")

        path = raw.get("path")
        confidence = raw.get("confidence")
        reason = raw.get("reason")

        if not isinstance(path, str) or path not in allowed_paths:
            raise ValueError("prediction contains an unsupported file path")
        if not isinstance(confidence, (int, float)) or isinstance(
            confidence,
            bool,
        ):
            raise TypeError("confidence must be numeric")
        if not 0 <= float(confidence) <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("prediction reason must not be empty")

        symbols = _require_string_list(raw.get("symbols"), "symbols")
        evidence_ids = _require_string_list(
            raw.get("evidence_chunk_ids"),
            "evidence_chunk_ids",
        )

        if any(symbol not in allowed_symbols for symbol in symbols):
            raise ValueError("prediction contains an unsupported symbol")
        if not evidence_ids:
            raise ValueError("each prediction requires repository evidence")
        if any(
            chunk_id not in allowed_chunk_ids
            for chunk_id in evidence_ids
        ):
            raise ValueError("prediction cites unsupported evidence")

        prediction = FilePrediction(
            path=path,
            symbols=symbols,
            evidence_chunk_ids=evidence_ids,
            confidence=float(confidence),
            reason=reason,
        )

        if path not in prediction_positions:
            prediction_positions[path] = len(predictions)
            predictions.append(prediction)
        else:
            position = prediction_positions[path]
            existing = predictions[position]
            combined_reason = existing.reason

            if reason != existing.reason:
                combined_reason = f"{existing.reason} {reason}"

            predictions[position] = FilePrediction(
                path=path,
                symbols=tuple(
                    dict.fromkeys((*existing.symbols, *symbols))
                ),
                evidence_chunk_ids=tuple(
                    dict.fromkeys(
                        (
                            *existing.evidence_chunk_ids,
                            *evidence_ids,
                        )
                    )
                ),
                confidence=max(existing.confidence, float(confidence)),
                reason=combined_reason,
            )

    predicted_tests = _require_string_list(
        payload["predicted_tests"],
        "predicted_tests",
    )
    unresolved_questions = _require_string_list(
        payload["unresolved_questions"],
        "unresolved_questions",
    )

    if any(path not in allowed_paths for path in predicted_tests):
        raise ValueError("prediction contains an unsupported test path")

    if abstain:
        if predictions:
            raise ValueError("an abstention must not contain predictions")
        if not abstention_reason.strip():
            raise ValueError("an abstention requires a reason")
    else:
        if not predictions:
            raise ValueError("a non-abstention requires predictions")
        if abstention_reason.strip():
            raise ValueError(
                "a non-abstention must have an empty abstention reason"
            )

    return ChangeImpactPrediction(
        abstain=abstain,
        abstention_reason=abstention_reason,
        predictions=tuple(predictions),
        predicted_tests=predicted_tests,
        unresolved_questions=unresolved_questions,
        model=model,
    )


def predict_change_impact(
    client: ChatClient,
    query: str,
    evidence: Sequence[RerankedChunk],
    *,
    model: str = COMMAND_MODEL,
) -> ChangeImpactPrediction:
    """Ask Cohere Command for a schema-constrained grounded prediction."""
    if not query.strip():
        raise ValueError("query must not be empty")
    if not evidence:
        return ChangeImpactPrediction(
            abstain=True,
            abstention_reason="No repository evidence was retrieved.",
            predictions=(),
            predicted_tests=(),
            unresolved_questions=(),
            model=model,
        )

    response = client.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": build_prediction_prompt(query, evidence),
            }
        ],
        response_format={
            "type": "json_object",
            "schema": PREDICTION_SCHEMA,
        },
        temperature=0,
    )
    content = response.message.content
    if not content:
        raise ValueError("Cohere returned an empty prediction response")

    response_text = None

    for item in content:
        try:
            candidate_text = item.text
        except AttributeError:
            continue

        if isinstance(candidate_text, str):
            response_text = candidate_text
            break

    if response_text is None:
        raise ValueError(
            "Cohere prediction response did not contain a text block"
        )

    payload = json.loads(response_text)
    return _parse_prediction(payload, evidence, model)
