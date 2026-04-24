"""Tests for FoundationModel."""

import pytest
import torch

from foundation_model import FoundationModel, FoundationModelConfig


@pytest.fixture
def tiny_config() -> FoundationModelConfig:
    return FoundationModelConfig.tiny()


@pytest.fixture
def tiny_model(tiny_config: FoundationModelConfig) -> FoundationModel:
    return FoundationModel(tiny_config)


def test_forward_shape(tiny_model: FoundationModel, tiny_config: FoundationModelConfig) -> None:
    B, T = 2, 8
    input_ids = torch.randint(0, tiny_config.vocab_size, (B, T))
    logits = tiny_model(input_ids)
    assert logits.shape == (B, T, tiny_config.vocab_size)


def test_predict_with_uncertainty_shapes(tiny_model: FoundationModel, tiny_config: FoundationModelConfig) -> None:
    B, T = 2, 8
    input_ids = torch.randint(0, tiny_config.vocab_size, (B, T))
    mean_logits, uncertainty = tiny_model.predict_with_uncertainty(input_ids, n_samples=5)
    assert mean_logits.shape == (B, T, tiny_config.vocab_size)
    assert uncertainty.shape == (B, T)
    assert (uncertainty >= 0).all()


def test_n_parameters_positive(tiny_model: FoundationModel) -> None:
    assert tiny_model.n_parameters > 0


@pytest.mark.parametrize("preset", ["tiny", "small", "base", "large"])
def test_all_presets_instantiate(preset: str) -> None:
    cfg = getattr(FoundationModelConfig, preset)()
    model = FoundationModel(cfg)
    assert model.n_parameters > 0


def test_weight_tying(tiny_model: FoundationModel) -> None:
    assert tiny_model.head.weight is tiny_model.token_embed.weight
