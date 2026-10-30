# Testing Strategy

How this repository is tested, what each suite guards, and the honest state
of enforcement.

## The suite (`tests/`, pytest — 98 tests across 9 files)

The test files map one-to-one onto the model's architectural claims, so the
suite doubles as a checklist of what the README promises:

| File | Guards |
| --- | --- |
| `test_model.py` | the core model's construction and forward behaviour |
| `test_sampling.py` | sampling correctness and determinism controls |
| `test_uncertainty.py` | the epistemic layer — the decomposition's values at both extremes (identical passes: all aleatoric; confident disagreeing passes: all epistemic, log(S)), and a model trained on a narrow stream being at least ten times as uncertain off it |
| `test_experience.py` | experience-driven updates |
| `test_voting.py` | ensemble/voting aggregation |
| `test_trainer.py` | the training loop |
| `test_federated.py` | federated training behaviour |
| `test_p2p.py` | peer-to-peer coordination |
| `test_main.py` | the demo entry point `python -m foundation_model` that the Docker targets run — each `DEMO_MODE` in-process on CPU, and `DEVICE=cuda` falling back to CPU |

Run: `pytest` (config in `pyproject.toml`: `testpaths = ["tests"]`), with
`torch>=2.0` installed — the one heavy dependency. All 87 tests verified
passing locally on 2026-08-24 (an earlier revision of this document recorded
the suite as not-yet-executed; torch was installed the same day and the run
confirmed green, at 88% coverage). The five `test_main.py` tests were added
on 2026-10-19 and the measured total rose to 97.91%. Four value tests for
the uncertainty decomposition were added on 2026-10-30; they cover lines
the shape tests already ran, so the total did not move.

## The coverage ratchet, exactly

CI runs `pytest --cov=foundation_model --cov-fail-under=97`. pytest-cov
compares the total against that figure at coverage's report precision,
which defaults to 0 decimals: a total of 87.62% rounded to 88 and passed the
old ratchet of 88 for the whole of September and early October 2026, while
the same run printed `FAIL Required test coverage of 88% not reached`. The
CI job was green and its log said FAIL. `[tool.coverage.report] precision =
2` in `pyproject.toml` makes the comparison exact (97.91 against 97.92 now
fails, verified locally), and the ratchet is set at the floor of the measured
total, so it can only be passed by coverage that is really there.

## The doc check (`.github/scripts/check_docs.py`)

Added 2026-10-10 with the README-proof convention: every capability row and
bullet in the README names the test that proves it, or says "no test yet".
The script is standard library only and runs as its own `docs` job in CI, so
it needs no torch. It fails on a README citation (`tests/<file>.py::<name>`)
whose file or name does not exist, a relative link in the README or under
`docs/` that resolves to nothing, a front-matter block below the top of any
Markdown file, and a README count or table that disagrees with the code: the
capability count against the overview table, the samplers against the
`*Sampler` classes in `sampling.py`, the scale presets against
`config.py`. What it cannot check is that a cited test proves the claim
beside it; that stays a reviewer's question, and the README says "no test
yet" wherever a test only runs the code without checking the claim. The
out-of-distribution behaviour of epistemic uncertainty and the value of the
aleatoric term have their tests in `test_uncertainty.py`; the robustness of
trimmed aggregation, the last such place, has its tests in `test_voting.py`
since the evening of 2026-10-30 (see the decision below).

## Decision: trimmed aggregation trims by weight (2026-10-30)

`ExpertiseWeightedAggregator(trim_fraction=...)` drops the lowest- and
highest-*weighted* contributors before averaging; it does not look at
parameter values (it is not a coordinate-wise trimmed mean). Dermot kept
that behaviour on 30 October 2026 ("Accept all four", recommendation (a):
keep trimming by weight and test it). Two tests in `test_voting.py` hold it:
`test_trim_excludes_inflated_reputation_and_worst_loss` proves what it buys
(an inflated reputation and a very poor validation loss are both dropped,
leaving the honest update, where without trimming the inflated contributor
wins), and `test_trim_is_by_weight_not_by_parameter_value` records its
limit (a poisoned update at middling weight survives). Changing to
value-based trimming would be a new decision, and the second test is the one
that would change with it.

## Known gaps (candidates for next)

- ~~No CI~~ — **closed 2026-08-24**: `.github/workflows/ci.yml` installs the
  CPU torch wheel with pip caching and runs the suite on Python 3.12 with a
  coverage ratchet at the measured 88% baseline, on every PR and push to
  `main`.
- **Determinism deserves explicit property tests.** Seeded sampling and
  voting aggregation both claim reproducibility; a test that runs each twice
  and asserts identical output pins it cheaply.
- ~~The Docker path is unexercised~~ — **partly closed 2026-10-30**: a
  `docker` job in CI builds the `cpu` target and runs it once with
  `DEMO_MODE=single`, on push to `main` only, not on pull requests (Dermot,
  30 October 2026, "Accept all four"), so a build that rots shows as a red
  run on `main` after the merge that broke it, not on the PR. It builds with
  the CPU torch index (`TORCH_EXTRA`) to keep the runner's disk and time
  down. What the image *runs* is covered in-process since 2026-10-19
  (`test_main.py`); still untested are the `gpu` target and the `cuda`
  branch on a GPU host.
- ~~Coverage is unmeasured~~ — the CI ratchet above holds it at ≥97%,
  compared exactly (see "The coverage ratchet, exactly").

## Extending

- A new architectural capability gets its own `test_<capability>.py`, keeping
  the file-per-claim mapping intact — the suite's legibility is that the
  table above stays true.
- Numerical assertions should use tolerances deliberately
  (`pytest.approx` with a stated rationale), not exact float equality that a
  BLAS change can break.
