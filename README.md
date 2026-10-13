# Cohere Workshops

Dunia Hub's four-part Python workshop and open research project on evidence-grounded change-impact analysis for open-source repositories.

The prototype investigates whether deterministic repository analysis combined with Cohere Embed, Rerank, and Command can predict which files and symbols are likely to change for a proposed issue or feature request.

## Current capabilities

- Safe repository scanning with file-size limits and Git ignore rules
- Sensitive filename and secret-pattern detection
- UTF-8 loading with SHA-256 source hashes
- Python AST symbols, qualified names, imports, and line ranges
- Deterministic file and symbol chunks capped at 12,000 characters
- Cohere `embed-v4.0` document and query embeddings
- Local cosine retrieval and BM25 keyword baseline
- Cohere `rerank-v4.0-pro` reranking
- Optional deterministic import-relationship expansion
- Cohere `command-a-plus-05-2026` structured predictions
- Local validation of paths, symbols, evidence IDs, tests, and confidence
- Explicit abstention when repository evidence is unavailable
- File-level Recall@K, Precision@K, reciprocal rank, and average precision
- Auditable exclusions, errors, and exploratory experiment records

## Research question

Can an evidence-grounded AI system predict affected parts of a Python repository more accurately than keyword retrieval, embedding retrieval, reranking, or a language model without repository grounding?

Formal evaluation will use historical merged pull requests and only the repository snapshot from before each change.

## Installation

Requirements: Git, Python 3.11 or newer, and a Cohere API key for live requests.

```bash
git clone https://github.com/dunia-hub/cohere-workshops.git
cd cohere-workshops
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
cp .env.example .env
```

Add the private key to `.env`:

```env
COHERE_API_KEY=your_private_key
```

Never commit `.env`.

## Verification

```bash
ruff check src tests
pytest
```

Tests use fake Cohere clients and do not spend API credits.

## Commands

Build a deterministic JSON repository map:

```bash
cohere-workshops .
cohere-workshops /path/to/repository
```

Run semantic retrieval and reranking:

```bash
cohere-search . "Respect Git ignore rules during scanning"
```

Add structural import relationships:

```bash
cohere-search . "Update repository loading" --structural
```

Generate a grounded change-impact prediction:

```bash
cohere-search . "Update repository loading" --structural --predict
```

Live search commands use Cohere API credits. JSON output excludes API keys, embedding vectors, and complete source content.

## Safety boundary

Before external processing, the prototype applies directory exclusions, Git ignore rules, sensitive filename checks, file-size limits, and high-confidence secret detection. Every exclusion is recorded. Command output is accepted only when its identifiers are supported by retrieved repository evidence.

## Research configurations

1. BM25 keyword retrieval
2. Cohere Embed retrieval
3. Cohere Embed with Cohere Rerank
4. Structural and semantic retrieval
5. Evidence-grounded Command prediction with abstention
6. Ungrounded language-model baseline (planned)

Development cases remain separate from the final evaluation dataset.

## Documentation

### Research

- [Research protocol](docs/research/protocol.md)
- [Evaluation plan](docs/research/evaluation-plan.md)
- [Data and ethics](docs/research/data-and-ethics.md)
- [Experiment log](docs/research/experiment-log.md)

### Engineering

- [System architecture](docs/engineering/architecture.md)
- [Engineering decisions](docs/engineering/decisions.md)
- [Reproducibility guide](docs/engineering/reproducibility.md)

### Experiments

- [Exploratory records](experiments/exploratory)

## Current limitations

- Structural parsing supports Python only.
- Formal pull-request dataset construction is not implemented.
- Embeddings are not cached between runs.
- The ungrounded Command baseline and aggregate benchmark runner are not implemented.
- Dependency locking and continuous integration are not configured.
- Current experiment records are exploratory, not final results.

Workshop materials will be based on the tested implementation and measured results.
