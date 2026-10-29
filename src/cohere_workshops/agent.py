"""A read-only, version aware code agent built on Cohere Command."""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from cohere_workshops.agent_tools import (
    SEARCH_TOOL,
    CodeSearchTool,
    decision_records,
    evidence_document,
    stale_references,
)
from cohere_workshops.prediction import COMMAND_MODEL

AGENT_PROMPT = """You are a read-only code assistant for one Git repository.
Always call search_code before answering. You may call it again with a better query.
The tool only returns code that is valid at the target commit.
Answer using only that evidence. If it does not answer the question, say so."""

ANSWER_PROMPT = """Answer the question using only the evidence below.
Every piece of evidence is valid at the target commit.
Cite the evidence IDs you used. If the evidence does not answer the question,
set declined to true, explain why in decline_reason and leave answer empty.

Question:
{question}

Evidence:
{evidence}
"""


class ChatClient(Protocol):
    def chat(self, **kwargs: object) -> object: ...


@dataclass
class AgentResult:
    """The final answer plus everything needed to audit it."""

    question: str
    answer: str
    declined: bool
    decline_reason: str
    evidence_ids: list[str]
    references: list[dict[str, object]]
    tool_plans: list[str] = field(default_factory=list)
    forced_search: bool = False
    stale: list[str] = field(default_factory=list)


def answer_schema(evidence_ids: Sequence[str]) -> dict[str, object]:
    """JSON Schema that only allows citing evidence that passed the Git check."""
    return {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "declined": {"type": "boolean"},
            "decline_reason": {"type": "string"},
            "evidence_ids": {
                "type": "array",
                "items": {"type": "string", "enum": list(evidence_ids)},
            },
        },
        "required": ["answer", "declined", "decline_reason", "evidence_ids"],
    }


def message_text(message: object) -> str | None:
    """Return the first text block, skipping thinking blocks."""
    for item in getattr(message, "content", None) or []:
        text = getattr(item, "text", None)
        if isinstance(text, str):
            return text
    return None


def _tool_content(documents: list[dict[str, object]]) -> list[dict[str, object]]:
    if not documents:
        documents = [{"id": "none", "result": "No code valid at the target commit."}]
    return [
        {"type": "document", "document": {"data": doc, "id": str(doc["id"])}}
        for doc in documents
    ]


def gather_evidence(
    client: ChatClient,
    tool: CodeSearchTool,
    question: str,
    *,
    model: str = COMMAND_MODEL,
    max_tool_calls: int = 3,
) -> tuple[list[str], bool]:
    """Let Command decide what to search for. Our code runs every search."""
    messages: list[object] = [
        {"role": "system", "content": AGENT_PROMPT},
        {"role": "user", "content": question},
    ]
    plans: list[str] = []
    calls_made = 0

    for _ in range(max_tool_calls + 1):
        response = client.chat(
            model=model, messages=messages, tools=[SEARCH_TOOL], temperature=0
        )
        message = response.message
        tool_calls = getattr(message, "tool_calls", None) or []
        if not tool_calls:
            break

        plan = getattr(message, "tool_plan", None)
        if plan:
            plans.append(plan)
        messages.append(message)

        for tool_call in tool_calls:
            documents: list[dict[str, object]] = []
            if tool_call.function.name == "search_code" and calls_made < max_tool_calls:
                arguments = json.loads(tool_call.function.arguments or "{}")
                query = str(arguments.get("query", "")).strip() or question
                documents = tool.search(query)
                calls_made += 1
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": _tool_content(documents),
                }
            )

    forced = calls_made == 0
    if forced:
        tool.search(question)
    return plans, forced


def run_agent(
    client: ChatClient,
    tool: CodeSearchTool,
    question: str,
    *,
    model: str = COMMAND_MODEL,
    max_tool_calls: int = 3,
) -> AgentResult:
    """Search, validate with Git, then answer only from validated evidence."""
    if not question.strip():
        raise ValueError("question must not be empty")

    plans, forced = gather_evidence(
        client, tool, question, model=model, max_tool_calls=max_tool_calls
    )
    evidence = tool.valid_evidence

    if not evidence:
        return AgentResult(
            question=question,
            answer="",
            declined=True,
            decline_reason="No code valid at the target commit was found.",
            evidence_ids=[],
            references=[],
            tool_plans=plans,
            forced_search=forced,
        )

    documents = [evidence_document(item, tool.validator) for item in evidence.values()]
    response = client.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": ANSWER_PROMPT.format(
                    question=question,
                    evidence=json.dumps(documents, indent=2),
                ),
            }
        ],
        response_format={"type": "json_object", "schema": answer_schema(list(evidence))},
        temperature=0,
    )
    text = message_text(response.message)
    if text is None:
        raise ValueError("Command returned no text answer")

    payload = json.loads(text)
    cited = [i for i in payload.get("evidence_ids", []) if i in evidence]
    answer = str(payload.get("answer", ""))
    declined = bool(payload.get("declined", False))

    return AgentResult(
        question=question,
        answer=answer,
        declined=declined,
        decline_reason=str(payload.get("decline_reason", "")),
        evidence_ids=cited,
        references=[
            {
                "evidence_id": i,
                "path": evidence[i].chunk.path,
                "symbol": evidence[i].chunk.symbol,
                "lines": f"{evidence[i].chunk.start_line}-{evidence[i].chunk.end_line}",
                "valid_at_commit": tool.validator.commit,
            }
            for i in cited
        ],
        tool_plans=plans,
        forced_search=forced,
        stale=stale_references(answer, tool),
    )


def agent_record(
    result: AgentResult,
    tool: CodeSearchTool,
    *,
    run_id: str,
    models: dict[str, str],
) -> dict[str, object]:
    """Full trace: tool calls, validation decisions and the final answer."""
    return {
        "run_id": run_id,
        "question": result.question,
        "target": {"revision": tool.validator.revision, "commit": tool.validator.commit},
        "models": models,
        "tool_plans": result.tool_plans,
        "forced_search": result.forced_search,
        "tool_calls": decision_records(tool.calls),
        "answer": result.answer,
        "declined": result.declined,
        "decline_reason": result.decline_reason,
        "references": result.references,
        "stale_references": result.stale,
    }
