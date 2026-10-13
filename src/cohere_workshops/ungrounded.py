"""Generate a path-only ungrounded change-impact baseline."""

import json
from collections.abc import Sequence
from copy import deepcopy

from cohere_workshops.prediction import (
    COMMAND_MODEL,
    PREDICTION_SCHEMA,
    ChangeImpactPrediction,
    ChatClient,
    FilePrediction,
)


def _schema_for_paths(paths: Sequence[str]) -> dict[str, object]:
    """Constrain the baseline to paths that exist in the repository."""
    schema = deepcopy(PREDICTION_SCHEMA)
    properties = schema["properties"]
    prediction_properties = properties["predictions"]["items"][
        "properties"
    ]
    prediction_properties.pop("symbols")
    prediction_properties.pop("evidence_chunk_ids")
    required = properties["predictions"]["items"]["required"]
    required.remove("symbols")
    required.remove("evidence_chunk_ids")
    prediction_properties["path"]["enum"] = list(paths)
    properties["predicted_tests"]["items"]["enum"] = list(paths)
    return schema


def build_ungrounded_prompt(query: str, paths: Sequence[str]) -> str:
    """Build a baseline prompt containing paths but no repository content."""
    path_list = "\n".join(f"- {path}" for path in paths)
    return f"""Generate a JSON change-impact prediction.

Proposed change:
{query}

Repository file paths:
{path_list}

Rules:
1. Use only the file paths listed above.
2. Do not assume access to file contents, symbols, imports, or retrieved evidence.
3. Predict files that are likely to change based only on the request and paths.
4. Confidence must be between 0 and 1.
5. If the available information is insufficient, abstain and explain why.
6. Do not invent files.
"""


def _string_list(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) for item in value
    ):
        raise ValueError(f"{field} must be a list of strings")
    return tuple(value)


def _response_text(response: object) -> str:
    content = response.message.content
    if not content:
        raise ValueError("Cohere returned an empty prediction response")

    for item in content:
        try:
            text = item.text
        except AttributeError:
            continue
        if isinstance(text, str):
            return text

    raise ValueError(
        "Cohere prediction response did not contain a text block"
    )


def _parse_ungrounded_prediction(
    payload: object,
    paths: Sequence[str],
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

    allowed_paths = set(paths)
    predictions: list[FilePrediction] = []
    positions: dict[str, int] = {}

    for raw in raw_predictions:
        if not isinstance(raw, dict):
            raise TypeError("each prediction must be an object")

        path = raw.get("path")
        confidence = raw.get("confidence")
        reason = raw.get("reason")

        if not isinstance(path, str) or path not in allowed_paths:
            raise ValueError("prediction contains an unsupported file path")
        if not isinstance(confidence, (int, float)) or isinstance(
            confidence, bool
        ):
            raise TypeError("confidence must be numeric")
        if not 0 <= float(confidence) <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("prediction reason must not be empty")

        prediction = FilePrediction(
            path=path,
            symbols=(),
            evidence_chunk_ids=(),
            confidence=float(confidence),
            reason=reason,
        )

        if path not in positions:
            positions[path] = len(predictions)
            predictions.append(prediction)
        else:
            position = positions[path]
            existing = predictions[position]
            combined_reason = existing.reason
            if reason != existing.reason:
                combined_reason = f"{existing.reason} {reason}"
            predictions[position] = FilePrediction(
                path=path,
                symbols=(),
                evidence_chunk_ids=(),
                confidence=max(existing.confidence, float(confidence)),
                reason=combined_reason,
            )

    predicted_tests = _string_list(
        payload["predicted_tests"], "predicted_tests"
    )
    unresolved_questions = _string_list(
        payload["unresolved_questions"], "unresolved_questions"
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


def predict_ungrounded_change_impact(
    client: ChatClient,
    query: str,
    paths: Sequence[str],
    *,
    model: str = COMMAND_MODEL,
) -> ChangeImpactPrediction:
    """Predict from a change request and file paths without source evidence."""
    if not query.strip():
        raise ValueError("query must not be empty")

    unique_paths = tuple(dict.fromkeys(paths))
    if not unique_paths:
        return ChangeImpactPrediction(
            abstain=True,
            abstention_reason="No repository file paths were provided.",
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
                "content": build_ungrounded_prompt(query, unique_paths),
            }
        ],
        response_format={
            "type": "json_object",
            "schema": _schema_for_paths(unique_paths),
        },
        temperature=0,
    )
    payload = json.loads(_response_text(response))
    return _parse_ungrounded_prediction(payload, unique_paths, model)
