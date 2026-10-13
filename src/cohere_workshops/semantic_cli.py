"""Command-line interface for Cohere semantic repository retrieval."""

import argparse
import json
import os
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import cohere
from dotenv import load_dotenv

from cohere_workshops.embeddings import EMBED_MODEL
from cohere_workshops.prediction import predict_change_impact
from cohere_workshops.reranking import RERANK_MODEL
from cohere_workshops.semantic import (
    SemanticSearchResult,
    semantic_search_repository,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the semantic-search command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve repository evidence for a proposed code change "
            "using Cohere Embed and Rerank."
        )
    )
    parser.add_argument("repository", help="Repository directory to search.")
    parser.add_argument("query", help="Issue or proposed change to investigate.")
    parser.add_argument(
        "--retrieval-k",
        type=int,
        default=20,
        help="Number of semantic candidates to retrieve.",
    )
    parser.add_argument(
        "--rerank-k",
        type=int,
        default=10,
        help="Number of candidates to return after reranking.",
    )
    parser.add_argument(
        "--structural",
        action="store_true",
        help="Expand semantic candidates using local import relationships.",
    )
    parser.add_argument(
        "--max-structural-expansions",
        type=int,
        default=10,
        help="Maximum number of structurally related chunks to add.",
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Generate a grounded Command change-impact prediction.",
    )
    return parser


def serialize_search_result(
    result: SemanticSearchResult,
    query: str,
) -> dict[str, object]:
    """Create auditable JSON without vectors or complete source content."""
    evidence = []

    for rank, reranked in enumerate(result.results, start=1):
        retrieved = reranked.retrieved_chunk
        chunk = retrieved.embedded_chunk.chunk
        evidence.append(
            {
                "rank": rank,
                "path": chunk.path,
                "kind": chunk.kind,
                "symbol": chunk.symbol,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "chunk_id": chunk.chunk_id,
                "content_hash": chunk.content_hash,
                "source_hash": chunk.source_hash,
                "semantic_score": retrieved.score,
                "rerank_score": reranked.rerank_score,
                "original_retrieval_rank": reranked.original_rank,
            }
        )

    return {
        "root": result.root,
        "query": query,
        "embed_model": EMBED_MODEL,
        "rerank_model": RERANK_MODEL,
        "chunk_count": result.chunk_count,
        "structural_expansion": result.structural_expansion,
        "exclusions": [asdict(item) for item in result.exclusions],
        "errors": [asdict(item) for item in result.errors],
        "evidence": evidence,
    }


def main(arguments: Sequence[str] | None = None) -> None:
    """Run semantic repository retrieval and print traceable JSON."""
    options = build_parser().parse_args(arguments)
    load_dotenv()

    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise RuntimeError("COHERE_API_KEY is missing from .env")

    client = cohere.ClientV2(
        api_key=api_key,
        log_warning_experimental_features=False,
    )
    result = semantic_search_repository(
        client,
        Path(options.repository),
        options.query,
        retrieval_k=options.retrieval_k,
        rerank_k=options.rerank_k,
        structural_expansion=options.structural,
        max_structural_expansions=options.max_structural_expansions,
    )
    payload = serialize_search_result(result, options.query)

    if options.predict:
        prediction = predict_change_impact(
            client,
            options.query,
            result.results,
        )
        payload["prediction"] = asdict(prediction)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
