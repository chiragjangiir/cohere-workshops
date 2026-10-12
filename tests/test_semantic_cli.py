from cohere_workshops.corpus import CorpusError
from cohere_workshops.scanner import ExcludedFile
from cohere_workshops.semantic import SemanticSearchResult
from cohere_workshops.semantic_cli import serialize_search_result


def test_serializes_metadata_without_vectors_or_source_content() -> None:
    result = SemanticSearchResult(
        root="example",
        chunk_count=0,
        exclusions=(ExcludedFile(".env", "sensitive_path"),),
        errors=(CorpusError("broken.py", "chunk", "invalid_python_syntax"),),
        results=(),
    )

    payload = serialize_search_result(result, "Update scanning")

    assert payload["root"] == "example"
    assert payload["query"] == "Update scanning"
    assert payload["chunk_count"] == 0
    assert payload["exclusions"] == [
        {"path": ".env", "reason": "sensitive_path"}
    ]
    assert payload["errors"] == [
        {
            "path": "broken.py",
            "stage": "chunk",
            "reason": "invalid_python_syntax",
        }
    ]
    assert payload["evidence"] == []
    assert "vector" not in payload
    assert "content" not in payload
