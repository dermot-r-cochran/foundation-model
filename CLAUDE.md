# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An epistemic, experience-driven foundation model: a small causal-transformer language model (PyTorch, the only runtime dependency) wrapped in machinery for uncertainty estimation, continual learning, and decentralised training — federated, peer-to-peer, and expertise-weighted voting. Designed to run anywhere from a laptop (CPU) to a GPU cloud VM via `FoundationModelConfig` scale presets (`tiny`/`small`/`base`/`large`; `tiny` is the one meant for tests and CI).

## Commands

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU wheel, what CI installs
pip install -e ".[dev]"          # package + pytest/pytest-cov
pytest                           # full suite (testpaths = ["tests"] in pyproject.toml)
pytest tests/test_federated.py   # one file
pytest tests/test_voting.py -k expertise   # one test by keyword
pytest --cov=foundation_model --cov-fail-under=88   # exactly what CI runs
```

There is no linter or type-checker configured — CI (`.github/workflows/ci.yml`, Python 3.12, CPU torch) runs only the pytest command above. The `--cov-fail-under=88` is a **ratchet** at the measured baseline: raise it with tests that earn it, never lower it. Testing mechanics — what each test file guards, known gaps, how to extend the suite — are in `TestingStrategy.md`; keep its file-per-capability table true by giving any new capability its own `tests/test_<capability>.py`.

Docker is the deployment story, not the dev loop: a multi-stage `Dockerfile` with `cpu` and `gpu` targets (GPU differs only by `--build-arg TORCH_EXTRA="--index-url https://download.pytorch.org/whl/cu121"`), both running the demo entry point `python -m foundation_model`. `docker compose up foundation-model-{cpu,federated,p2p,gpu}` selects a demo via the `DEMO_MODE` env var (`single`/`federated`/`p2p`/`all`), which `foundation_model/__main__.py` reads.

## Architecture

Everything hangs off one core idea: the model quantifies its own uncertainty, and every distributed mechanism weights participants by demonstrated competence rather than headcount or dataset size.

- **`model.py`** — decoder-only transformer (weight-tied embeddings/LM head) whose every `Dropout` is `MCDropout`, a subclass that stays active at inference. `predict_with_uncertainty()` runs N stochastic forward passes and returns mean logits plus per-token epistemic uncertainty. **`uncertainty.py`** decomposes total predictive entropy into epistemic (mutual information — high for out-of-distribution input) and aleatoric (mean per-sample entropy — irreducible noise) via `compute_uncertainty()`.
- **`sampling.py`** — four representative sub-samplers (stratified quantile buckets, greedy k-centers diversity, top-k uncertainty, and a combined `RepresentativeSampler`). This is shared infrastructure: the experience buffer, federated clients, and any large/skewed dataset use it to pick a covering subset instead of training on everything.
- **`experience.py` + `trainer.py`** — continual learning. `Trainer.train_step()` stores each batch in an `ExperienceBuffer` (ring buffer) with the training loss as its priority; `replay_step()` re-trains on a stratified sample of past experiences so easy and hard cases are both replayed, guarding against catastrophic forgetting.
- **`voting.py`** — the expertise layer the distributed modules build on. `expertise_weight = reputation / validation_loss`, tracked per contributor by `ReputationTracker`. `ExpertiseWeightedAggregator` merges state dicts by that weight (with optional Byzantine-robust tail trimming); `DiscreteVote` applies the same weighting to any categorical decision, so one calibrated expert can legitimately outvote a novice majority.
- **`federated.py`** — server/client federated learning. `FederatedServer.aggregate()` supports `"fedavg"` (weight by dataset size) or `"expertise"` (delegates to the voting module's aggregator, needs per-client validation losses). Clients wrap a private `Trainer` and sub-sample locally via `StratifiedSampler` when `fit(max_samples=...)` is set.
- **`p2p.py`** — serverless alternative: gossip averaging where each peer blends parameters with a random neighbour per round; `expertise_weighted=True` biases each blend toward the lower-validation-loss peer, so knowledge flows from experts outward instead of being diluted.

The public API is re-exported flat from `foundation_model/__init__.py`.

## Conventions

- Numerical test assertions use deliberate tolerances (`pytest.approx` with a stated rationale), never exact float equality — a BLAS change can break the latter.
- `README.md` carries worked usage examples for every capability; keep its capability table and `TestingStrategy.md`'s test table in sync with any new module.
