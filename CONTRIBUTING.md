# Contributing to Cohere Workshops

Thank you for helping build **Right Code, Wrong Version**, our open research project on whether AI code search can tell current code from old code.

You do not need to have attended the workshops. If you are new, start with an issue labelled `good first issue`.

## Ground rules

1. **Be kind.** We are a learning community. Ask questions, and answer them patiently.
2. **One change per pull request.** Small pull requests get reviewed faster.
3. **Open an issue first** for anything bigger than a small fix, so we can agree on the approach before you write code.
4. **Never commit secrets.** Your Cohere API key lives in `.env`, which Git ignores. Do not paste keys into issues, pull requests or code.

## Set up

You need Python 3.11 or newer, Git and a Cohere API key for live runs.

macOS or Linux:

```bash
git clone https://github.com/dunia-hub/cohere-workshops.git
cd cohere-workshops
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Windows PowerShell:

```powershell
git clone https://github.com/dunia-hub/cohere-workshops.git
cd cohere-workshops
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
Copy-Item .env.example .env
```

## Before you open a pull request

Both of these must pass:

```bash
ruff check src tests
pytest -q
```

## Code rules

These rules keep the research trustworthy. Reviews check them first.

1. **Tests never call Cohere.** Use a fake client, like the ones in `tests/`. Tests must be free, fast and work offline.
2. **Every change has a test.** New behavior needs a new test. A bug fix needs a test that fails before the fix.
3. **Git decides, the model does not.** File lists, versions, validity, ground truth and metrics are worked out with plain code. Cohere models may search, rank and write, but never decide what is true about the repository.
4. **The agent stays read only.** No code may write to, commit to or delete from the repository being analysed.
5. **Safety checks stay on.** Do not weaken the secret detection, sensitive file rules or size limits in `scanner.py` and `safety.py`.
6. **Records never store source code, vectors or API keys.** Store paths, commit hashes, chunk IDs and scores instead.
7. **Keep it readable.** Small functions, clear names, a short docstring on every public function. Workshop participants read this code.

## Research rules

1. **Exploratory runs are not results.** Anything you try while building a feature goes under `experiments/exploratory/` and cannot be reported as a finding.
2. **Record what you ran.** For any live experiment, note the app commit, the target repository and commits, model names, the exact command and the date. See `docs/engineering/reproducibility.md`.
3. **Keep failed runs.** A run that went wrong is useful data. Write down what failed instead of deleting it.
4. **Do not commit `results/` or `demo_repo/`.** They are generated on your machine.

## Commits and pull requests

- Write commit messages in the imperative, like the existing history: `Add rerank format option`, `Fix commit date lookup`.
- Fill in the pull request template.
- Link the issue your pull request closes, for example `Closes #12`.
- A maintainer reviews every pull request. Expect friendly questions, not judgement.

## Reporting a bug

Open an issue with:

- the exact command you ran
- what you expected
- what happened, including the full error
- your operating system and Python version (`python --version`)

Remove any API keys or private paths before posting.

## Questions

Ask in the issue you are working on, or reach the Dunia Hub team at fatuma@duniahub.xyz.
