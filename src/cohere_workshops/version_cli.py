"""Command-line interface for searching code across Git versions."""

import argparse
import hashlib
import os
import re
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv

from cohere_workshops.embeddings import EMBED_MODEL
from cohere_workshops.version_search import (
    LEVELS,
    QueryResult,
    append_records,
    count_unchanged,
    search_versions,
    select_level,
    serialize_query_result,
)
from cohere_workshops.versions import VersionedCorpus, build_versioned_corpus

DEMO_QUERIES = (
    "How is the transfer fee calculated?",
    "Check that a phone number is valid before sending money",
    "Send a mobile money payment",
)
OFFLINE_MODEL = "offline-word-hashing"
OFFLINE_DIMENSIONS = 512


class OfflineEmbedClient:
    """Practice mode without an API key. Counts words, understands nothing."""

    def embed(self, **kwargs: object) -> object:
        texts = kwargs["texts"]
        assert isinstance(texts, list)
        vectors = []

        for text in texts:
            vector = [0.0] * OFFLINE_DIMENSIONS
            vector[0] = 1.0  # avoids an all-zero vector for empty text
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                digest = hashlib.sha256(word.encode("utf-8")).digest()
                slot = 1 + int.from_bytes(digest[:4], "big") % (OFFLINE_DIMENSIONS - 1)
                vector[slot] += 1.0
            vectors.append(vector)

        return SimpleNamespace(embeddings=SimpleNamespace(float=vectors))


def build_parser() -> argparse.ArgumentParser:
    """Create the version-search command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Search code from two Git revisions with Cohere Embed and "
            "report whether current or outdated code ranks first."
        )
    )
    parser.add_argument("repository", help="Local Git repository.")
    parser.add_argument("--old", required=True, help="Older revision, e.g. v1.")
    parser.add_argument("--new", required=True, help="Current revision, e.g. v2.")
    parser.add_argument(
        "--query",
        action="append",
        help="Question to search for. Repeat for several. Defaults to demo questions.",
    )
    parser.add_argument("--level", choices=LEVELS, default="symbol")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Practice without an API key. Not semantic search.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/version-runs.jsonl"),
        help="JSON Lines file that results are appended to.",
    )
    return parser


def _create_client(offline: bool) -> object:
    if offline:
        return OfflineEmbedClient()

    import cohere

    load_dotenv()
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise RuntimeError("COHERE_API_KEY is missing from .env")
    return cohere.ClientV2(api_key=api_key, log_warning_experimental_features=False)


def print_corpus_summary(corpus: VersionedCorpus, level: str) -> None:
    counts = Counter(item.status for item in select_level(corpus.chunks, level))
    print(
        f"{corpus.previous.revision} ({corpus.previous.commit[:7]}) -> "
        f"{corpus.current.revision} ({corpus.current.commit[:7]})"
    )
    print(
        f"Indexed at {level} level: {counts['current']} current, "
        f"{counts['superseded']} superseded, {counts['removed']} removed"
    )
    print(f"Skipped {count_unchanged(corpus, level)} unchanged older chunks")

    for snapshot in (corpus.previous, corpus.current):
        for item in snapshot.exclusions:
            print(f"Excluded in {snapshot.revision}: {item.path} ({item.reason})")
        for error in snapshot.errors:
            print(f"Error in {snapshot.revision}: {error.path} ({error.reason})")


def print_query_result(result: QueryResult) -> None:
    print(f"\nQuery: {result.query}")
    print(f"{'Rank':<5} {'Score':<7} {'Version':<11} {'Rev':<6} Location")

    for hit in result.hits:
        chunk = hit.item.chunk
        location = f"{chunk.path}::{chunk.symbol or '<file>'}"
        print(
            f"{hit.rank:<5} {hit.score:<7.4f} {hit.item.status:<11} "
            f"{hit.item.revision:<6} {location}"
        )

    summary = result.summary
    print(f"Top result is current: {'yes' if summary.top_result_is_current else 'NO'}")
    print(f"Outdated code in top {len(result.hits)}: {summary.outdated_in_top_k}")
    if summary.score_gap is not None:
        print(f"Score gap, best current minus best outdated: {summary.score_gap:+.4f}")


def main(arguments: Sequence[str] | None = None) -> None:
    """Search two revisions and print and record which version wins."""
    options = build_parser().parse_args(arguments)
    if options.top_k < 1:
        raise ValueError("--top-k must be at least 1")

    queries = tuple(options.query or DEMO_QUERIES)
    model = OFFLINE_MODEL if options.offline else EMBED_MODEL
    client = _create_client(options.offline)

    corpus = build_versioned_corpus(
        Path(options.repository),
        options.old,
        options.new,
    )
    print_corpus_summary(corpus, options.level)

    results = search_versions(
        client,
        corpus,
        queries,
        level=options.level,
        top_k=options.top_k,
    )
    if not results:
        print("No chunks to search at this level.")
        return

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    records = []

    for result in results:
        print_query_result(result)
        records.append(
            serialize_query_result(
                result,
                corpus,
                run_id=run_id,
                level=options.level,
                model=model,
            )
        )

    append_records(options.output, records)
    print(f"\nSaved {len(records)} records to {options.output}")


if __name__ == "__main__":
    main()
