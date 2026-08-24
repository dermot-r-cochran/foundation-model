# Testing Strategy

How this repository is tested, what each suite guards, and the honest state
of enforcement.

## The suite (`tests/`, pytest — 77 tests across 8 files)

The test files map one-to-one onto the model's architectural claims, so the
suite doubles as a checklist of what the README promises:

| File | Guards |
| --- | --- |
| `test_model.py` | the core model's construction and forward behaviour |
| `test_sampling.py` | sampling correctness and determinism controls |
| `test_uncertainty.py` | the epistemic layer — uncertainty estimates behaving as claimed |
| `test_experience.py` | experience-driven updates |
| `test_voting.py` | ensemble/voting aggregation |
| `test_federated.py` | federated training behaviour |
| `test_p2p.py` | peer-to-peer coordination |

Run: `pytest` (config in `pyproject.toml`: `testpaths = ["tests"]`), with
`torch>=2.0` installed — the one heavy dependency, and the reason this suite
was **not** executed during the 2026-08-24 documentation pass (recorded here
so nobody mistakes "documented" for "verified"; every other repo in this
account's set had its suite run before its strategy was written).

## Known gaps (candidates for next)

- **No CI.** The torch dependency makes the workflow heavier than the
  sibling repos', not infeasible: install the CPU wheel
  (`pip install torch --index-url https://download.pytorch.org/whl/cpu`) with
  pip caching, then `pytest`. A suite guarding uncertainty and federated
  behaviour is exactly the kind that regresses quietly without enforcement.
- **Determinism deserves explicit property tests.** Seeded sampling and
  voting aggregation both claim reproducibility; a test that runs each twice
  and asserts identical output pins it cheaply.
- The Docker path (`Dockerfile`, `docker-compose.yml`) is unexercised by any
  test; a CI job that at least builds the image would catch dependency rot.
- Coverage is unmeasured; once CI exists, add `--cov` with a
  `--cov-fail-under` ratchet at the measured baseline (the pattern in the
  `swarm` repo).

## Extending

- A new architectural capability gets its own `test_<capability>.py`, keeping
  the file-per-claim mapping intact — the suite's legibility is that the
  table above stays true.
- Numerical assertions should use tolerances deliberately
  (`pytest.approx` with a stated rationale), not exact float equality that a
  BLAS change can break.
