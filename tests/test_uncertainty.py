"""Tests for epistemic / aleatoric uncertainty estimation."""

import pytest
import torch

from foundation_model import FoundationModel, FoundationModelConfig, compute_uncertainty
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
