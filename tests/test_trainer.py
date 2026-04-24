"""Tests for Trainer and experience replay."""

import pytest
import torch

from foundation_model import FoundationModel, FoundationModelConfig
from foundation_model.trainer import Trainer


@pytest.fixture
def trainer() -> Trainer:
    config = FoundationModelConfig.tiny()
    model = FoundationModel(config)
    return Trainer(model, config)


def _batch(config: FoundationModelConfig, B: int = 2, T: int = 8):
    ids = torch.randint(0, config.vocab_size, (B, T))
    labels = torch.randint(0, config.vocab_size, (B, T))
    return ids, labels


def test_train_step_returns_loss(trainer: Trainer) -> None:
    ids, labels = _batch(trainer.config)
    result = trainer.train_step(ids, labels)
    assert "loss" in result
    assert result["loss"] > 0
    assert result["step"] == 1


def test_train_step_increments_step(trainer: Trainer) -> None:
    ids, labels = _batch(trainer.config)
    for i in range(3):
        trainer.train_step(ids, labels)
    assert trainer.step == 3


def test_train_step_populates_buffer(trainer: Trainer) -> None:
    ids, labels = _batch(trainer.config)
    trainer.train_step(ids, labels)
    assert len(trainer.experience_buffer) == 1


def test_replay_step_none_when_buffer_empty(trainer: Trainer) -> None:
    result = trainer.replay_step(batch_size=4)
    assert result is None


@pytest.mark.parametrize("strategy", ["priority", "stratified", "uniform"])
def test_replay_step_after_training(strategy: str) -> None:
    config = FoundationModelConfig.tiny()
    model = FoundationModel(config)
    t = Trainer(model, config)
    for _ in range(8):
        ids, labels = _batch(config)
        t.train_step(ids, labels)
    result = t.replay_step(batch_size=4, strategy=strategy)
    assert result is not None
    assert "loss" in result


def test_loss_decreases_with_training() -> None:
    """Loss should trend downward when trained repeatedly on the same batch."""
    config = FoundationModelConfig.tiny()
    torch.manual_seed(0)
    model = FoundationModel(config)
    t = Trainer(model, config)
    ids, labels = _batch(config, B=4)

    losses = [t.train_step(ids, labels, store_in_buffer=False)["loss"] for _ in range(20)]
    assert losses[-1] < losses[0], "Loss did not decrease during training"
