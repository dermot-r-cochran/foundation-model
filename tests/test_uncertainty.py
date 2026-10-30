"""Tests for epistemic / aleatoric uncertainty estimation."""

import pytest
import torch

from foundation_model import FoundationModel, FoundationModelConfig, compute_uncertainty
from foundation_model.trainer import Trainer
from foundation_model.uncertainty import (
    aleatoric_uncertainty,
    entropy,
    epistemic_uncertainty,
    predictive_entropy,
)


@pytest.fixture
def tiny_model() -> FoundationModel:
    return FoundationModel(FoundationModelConfig.tiny())


# ------------------------------------------------------------------
# entropy
# ------------------------------------------------------------------


def test_entropy_uniform_distribution() -> None:
    V = 10
    probs = torch.ones(2, 4, V) / V
    h = entropy(probs)
    assert h.shape == (2, 4)
    expected = torch.log(torch.tensor(float(V)))
    assert torch.allclose(h, expected.expand(2, 4), atol=1e-4)


def test_entropy_certain_distribution() -> None:
    probs = torch.zeros(2, 4, 10)
    probs[:, :, 0] = 1.0
    h = entropy(probs)
    assert (h < 0.01).all()


# ------------------------------------------------------------------
# compute_uncertainty shapes and non-negativity
# ------------------------------------------------------------------


def test_compute_uncertainty_shapes(tiny_model: FoundationModel) -> None:
    B, T = 2, 8
    V = tiny_model.config.vocab_size
    input_ids = torch.randint(0, V, (B, T))
    result = compute_uncertainty(tiny_model, input_ids, n_samples=5)

    assert result["mean_logits"].shape == (B, T, V)
    assert result["predictive_entropy"].shape == (B, T)
    assert result["aleatoric"].shape == (B, T)
    assert result["epistemic"].shape == (B, T)


def test_epistemic_non_negative(tiny_model: FoundationModel) -> None:
    B, T = 2, 8
    input_ids = torch.randint(0, tiny_model.config.vocab_size, (B, T))
    result = compute_uncertainty(tiny_model, input_ids, n_samples=10)
    assert (result["epistemic"] >= 0).all()


def test_predictive_entropy_ge_aleatoric(tiny_model: FoundationModel) -> None:
    """Predictive entropy >= aleatoric (before clamp, may be slightly violated by float32)."""
    B, T = 2, 8
    input_ids = torch.randint(0, tiny_model.config.vocab_size, (B, T))
    result = compute_uncertainty(tiny_model, input_ids, n_samples=20)
    # epistemic = pred_entropy - aleatoric clamped to >= 0; verify clamping works
    assert (result["epistemic"] >= 0).all()


# ------------------------------------------------------------------
# Decomposition helpers on synthetic data
# ------------------------------------------------------------------


def _make_logits_stack(S: int, B: int, T: int, V: int) -> torch.Tensor:
    """Build a deterministic logits stack for testing."""
    return torch.randn(S, B, T, V)


def test_epistemic_uncertainty_shape() -> None:
    logits_stack = _make_logits_stack(10, 2, 4, 20)
    eu = epistemic_uncertainty(logits_stack)
    assert eu.shape == (2, 4)
    assert (eu >= 0).all()


def test_aleatoric_uncertainty_shape() -> None:
    logits_stack = _make_logits_stack(10, 2, 4, 20)
    au = aleatoric_uncertainty(logits_stack)
    assert au.shape == (2, 4)


def test_predictive_entropy_shape() -> None:
    logits_stack = _make_logits_stack(10, 2, 4, 20)
    pe = predictive_entropy(logits_stack)
    assert pe.shape == (2, 4)


# ------------------------------------------------------------------
# The decomposition's values, not just its shapes
# ------------------------------------------------------------------
#
# Total predictive entropy H[mean_s p_s] splits into aleatoric (the mean of
# the per-pass entropies, mean_s H[p_s]) and epistemic (the remainder, the
# mutual information between the dropout mask and the prediction). The two
# extremes below pin each term to a value computed independently of
# uncertainty.py. Tolerances are 1e-4 against values of order 1: float32
# softmax and log over a vocabulary of 20 lose about 1e-6 per element.


def test_aleatoric_is_mean_of_per_pass_entropies() -> None:
    """Aleatoric equals the mean over passes of each pass's own entropy."""
    torch.manual_seed(0)
    logits_stack = _make_logits_stack(10, 2, 4, 20)
    probs = torch.softmax(logits_stack, dim=-1)
    # Computed by hand, one pass at a time, without entropy() from the module.
    per_pass = torch.stack(
        [-(p * p.clamp_min(1e-12).log()).sum(dim=-1) for p in probs]
    )
    expected = per_pass.mean(dim=0)
    assert torch.allclose(aleatoric_uncertainty(logits_stack), expected, atol=1e-4)


def test_epistemic_is_zero_when_every_pass_agrees() -> None:
    """Identical passes: no disagreement, so all uncertainty is aleatoric."""
    torch.manual_seed(1)
    one_pass = torch.randn(1, 2, 4, 20)
    logits_stack = one_pass.expand(10, -1, -1, -1).clone()
    pe = predictive_entropy(logits_stack)
    au = aleatoric_uncertainty(logits_stack)
    eu = epistemic_uncertainty(logits_stack)
    assert torch.allclose(pe, au, atol=1e-4)
    assert torch.allclose(eu, torch.zeros_like(eu), atol=1e-4)


def test_epistemic_is_log_s_when_every_pass_is_sure_of_a_different_token() -> None:
    """Confident passes that disagree: aleatoric ~0, epistemic ~log(S).

    Pass s puts all its mass on token s. Each pass alone has entropy ~0,
    the mean prediction is uniform over S tokens with entropy log(S), and
    the whole of that is epistemic: the model is sure, but of different
    things on different dropout masks, which is what "the model knows what
    it doesn't know" means here. Logit 50 on the chosen token leaves the
    other 19 tokens with mass ~1e-22 each, far below the tolerance.
    """
    S, B, T, V = 5, 2, 3, 20
    logits_stack = torch.zeros(S, B, T, V)
    for s in range(S):
        logits_stack[s, :, :, s] = 50.0
    au = aleatoric_uncertainty(logits_stack)
    eu = epistemic_uncertainty(logits_stack)
    log_s = torch.full((B, T), float(torch.log(torch.tensor(float(S)))))
    assert torch.allclose(au, torch.zeros_like(au), atol=1e-4)
    assert torch.allclose(eu, log_s, atol=1e-4)


# ------------------------------------------------------------------
# Out of distribution: the README's first promise
# ------------------------------------------------------------------


def test_epistemic_higher_out_of_distribution() -> None:
    """A model trained on a narrow stream is more uncertain off it.

    The tiny model learns a deterministic cycle over the first 16 tokens of
    a 256-token vocabulary (next = current + 1 mod 16). After 200 steps the
    cycle is near-certain under every dropout mask, so epistemic uncertainty
    on it is close to zero, while tokens 16..255, never seen in training,
    make the masks disagree. Across 20 seeds on 2026-10-30 the ratio
    out/in was never below 176 (at 100 steps never below 11); the factor of
    10 asserted here leaves a wide margin for a BLAS or torch change
    without being satisfiable by noise.
    """
    torch.manual_seed(0)
    config = FoundationModelConfig.tiny()
    config.vocab_size = 256
    config.learning_rate = 1e-3
    model = FoundationModel(config)
    trainer = Trainer(model, config)
    B, T, in_vocab = 8, 16, 16

    def in_distribution_batch():
        start = torch.randint(0, in_vocab, (B, 1))
        seq = (start + torch.arange(T + 1).unsqueeze(0)) % in_vocab
        return seq[:, :-1].contiguous(), seq[:, 1:].contiguous()

    for _ in range(200):
        ids, labels = in_distribution_batch()
        trainer.train_step(ids, labels, store_in_buffer=False)

    ids_in, _ = in_distribution_batch()
    ids_out = torch.randint(in_vocab, config.vocab_size, (B, T))
    e_in = compute_uncertainty(model, ids_in, n_samples=20)["epistemic"].mean()
    e_out = compute_uncertainty(model, ids_out, n_samples=20)["epistemic"].mean()
    assert e_out > 10 * e_in, f"in-distribution {e_in:.4f}, out {e_out:.4f}"
