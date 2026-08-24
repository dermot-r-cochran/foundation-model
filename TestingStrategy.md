# Testing Strategy

How this repository is tested, what each suite guards, and the honest state
of enforcement.

## The suite (`tests/`, pytest — 87 tests across 8 files)

The test files map one-to-one onto the model's architectural claims, so the
suite doubles as a checklist of what the README promises:

| File | Guards |
| --- | --- |
| `test_model.py` | the core model's construction and forward behaviour |
| `test_sampling.py` | sampling correctness and determinism controls |
| `test_uncertainty.py` | the epistemic layer — uncertainty estimates behaving as claimed |
| `test_experience.py` | experience-driven updates |
| `test_voting.py` | ensemble/voting aggregation |
| `test_trainer.py` | the training loop |
| `test_federated.py` | federated training behaviour |
| `test_p2p.py` | peer-to-peer coordination |

Run: `pytest` (config in `pyproject.toml`: `testpaths = ["tests"]`), with
`torch>=2.0` installed — the one heavy dependency. All 87 tests verified
passing locally on 2026-08-24 (an earlier revision of this document recorded
the suite as not-yet-executed; torch was installed the same day and the run
confirmed green, at 88% coverage).

## Known gaps (candidates for next)

- ~~No CI~~ — **closed 2026-08-24**: `.github/workflows/ci.yml` installs the
  CPU torch wheel with pip caching and runs the suite on Python 3.12 with a
  coverage ratchet at the measured 88% baseline, on every PR and push to
  `main`.
- **Determinism deserves explicit property tests.** Seeded sampling and
  voting aggregation both claim reproducibility; a test that runs each twice
  and asserts identical output pins it cheaply.
- The Docker path (`Dockerfile`, `docker-compose.yml`) is unexercised by any
  test; a CI job that at least builds the image would catch dependency rot.
- ~~Coverage is unmeasured~~ — the CI ratchet above holds it at ≥88%.

## Extending

- A new architectural capability gets its own `test_<capability>.py`, keeping
  the file-per-claim mapping intact — the suite's legibility is that the
  table above stays true.
- Numerical assertions should use tolerances deliberately
  (`pytest.approx` with a stated rationale), not exact float equality that a
  BLAS change can break.
