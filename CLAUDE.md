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
pytest --cov=foundation_model --cov-fail-under=97   # exactly what CI runs
```

There is no linter or type-checker configured — CI (`.github/workflows/ci.yml`, Python 3.12, CPU torch) runs the pytest command above and, in a `docs` job, the stdlib doc check `python .github/scripts/check_docs.py` (README test citations exist, links resolve, one front-matter block at most, the README's counts and tables match the code). Every README row or bullet that claims a capability names the test that proves it, or says "no test yet" or "not yet implemented" (Dermot, 10 October 2026). The `--cov-fail-under=97` is a **ratchet** at the measured baseline: raise it with tests that earn it, never lower it. The comparison is exact to two decimals (`[tool.coverage.report] precision = 2` in `pyproject.toml`); at pytest-cov's default precision a total just under the ratchet rounds up and passes while the summary prints FAIL, which is how 87.62% sat green behind a ratchet of 88 until 2026-10-19. Testing mechanics — what each test file guards, known gaps, how to extend the suite — are in `TestingStrategy.md`; keep its file-per-capability table true by giving any new capability its own `tests/test_<capability>.py`.

Docker is the deployment story, not the dev loop (a `docker` job in CI builds the `cpu` target with the CPU torch index and runs it once with `DEMO_MODE=single`, on push to `main` only, never on pull requests; Dermot, 30 October 2026): a multi-stage `Dockerfile` with `cpu` and `gpu` targets (GPU differs only by `--build-arg TORCH_EXTRA="--index-url https://download.pytorch.org/whl/cu121"`), both running the demo entry point `python -m foundation_model`. `docker compose up foundation-model-{cpu,federated,p2p,gpu}` selects a demo via the `DEMO_MODE` env var (`single`/`federated`/`p2p`/`all`), which `foundation_model/__main__.py` reads.

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

## Which model does which work

Dermot's direction, 2026-10-11: the account-wide split recorded in `star-rangers`' `CLAUDE.md`, applied here. Opus is the default. Fable is kept for narrative prose and image prompts and has no work in this repository, with one exception: a whole-repo next-improvement read.

- **Fable: whole-repo next-improvement reads.** Fable may read the whole repository and propose its next improvement (the `next-improvement` question); that is reading and proposing, and it widens no one's write reservations, so the protected parts stay Opus's (Dermot, 30 October 2026, "Accept all four").
- **Haiku: reading and trivial fixes.** Read-only searches, summarising a failing CI log, and `check_docs.py` drift. The coverage gate and the doc check catch its mistakes.
- **Sonnet: code and tests.** Logic changes and new tests, including tests that raise the coverage ratchet. A test that merely passes is not necessarily one that checks the right thing, which is why this is not Haiku's. Numerical assertions keep a stated tolerance.
- **Opus: the protected parts.** The uncertainty decomposition in `uncertainty.py` and the expertise weighting in `voting.py`, which every distributed module builds on. No model below Opus edits these, even where a gate would catch the mistake.
- **Opus reviews and approves** a Haiku or Sonnet subagent's diff, and it merges when CI is green. Opus escalates where a change needs a judgement it should not make alone: to Fable where one is available, otherwise to Dermot directly as choices (Dermot, 2026-10-11). Haiku may also work under a Sonnet subagent, its searches and summaries feeding Sonnet's job as input; approval still comes from Opus, never from Sonnet.
- **Chrome extension:** Opus for anything that signs in, submits or posts; Sonnet for read-only browsing; never Haiku. The extension acts in Dermot's own Chrome with his sign-ins, and a web page can carry text written to steer the model.

A subagent runs on its parent session's model unless the call names another, so name the model on every delegation.

## Related repositories

The map of Dermot's public repositories and what crosses between them is
`RELATED-REPOSITORIES.md` in `dermot-r-cochran/star-rangers`; this section
names only this repository's own neighbours (added 2026-09-29 at his
direction). Nothing below shares code or data with this repository; what is
shared is stated exactly.

- **`dermot-r-cochran/Voting`** is a resemblance, not a relationship: the
  expertise-weighted voting in `voting.py` and that Rust proportional
  allocation crate share a word and nothing else. Don't look there for this
  module's semantics or vice versa.
- **`dermot-r-cochran/swarm`** (EPISTEME) and **`dermot-r-cochran/careful-memory`**
  are the account's other repositories about belief and uncertainty as
  primitives; this one quantifies its own uncertainty inside a model rather
  than keeping beliefs as objects. Neighbouring ideas, independent code.
- **Siblings by convention:** the account's engineering repositories share a
  way of working, not code: a `TestingStrategy.md` that keeps testing
  mechanics apart from the repository's rules, a CI coverage ratchet at the
  measured baseline, and architecture decision records guarded by named tests
  where a repository keeps them. Which repositories follow which, and how far
  each one's tests reach its decisions, is listed once, in the map's *Siblings
  by convention* section, and not restated here: counts copied into several
  files drift (this paragraph said `swarm` guarded "three of its four" ADRs
  after its fifth guard had landed). When a convention here needs changing,
  those are the reference for how it is done in the account, and a change to
  the convention itself is worth landing in all of them or in none.
