"""Tests for Trainer and experience replay."""

import pytest
import torch
import torch.nn.functional as F

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


# ------------------------------------------------------------------
# Catastrophic forgetting: the README's section-2 promise
# ------------------------------------------------------------------


def _cycle_batch(lo: int, width: int, B: int = 8, T: int = 16):
    """A batch from the task "next = current + 1, cycling over tokens lo..lo+width-1"."""
    start = torch.randint(lo, lo + width, (B, 1))
    seq = ((start - lo + torch.arange(T + 1).unsqueeze(0)) % width) + lo
    return seq[:, :-1].contiguous(), seq[:, 1:].contiguous()


def _mean_loss(model: FoundationModel, lo: int, width: int, n_batches: int = 8) -> float:
    total = 0.0
    with torch.no_grad():
        for _ in range(n_batches):
            ids, labels = _cycle_batch(lo, width)
            logits = model(ids)
            total += F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1)).item()
    return total / n_batches


def test_replay_keeps_an_earlier_task_after_training_on_another() -> None:
    """Replay guards against catastrophic forgetting, as the README says it does.

    Task A is a deterministic cycle over tokens 0..15 (next = current + 1
    mod 16); task B is the same rule over tokens 16..31, which share nothing
    with A. From one seed, the tiny model trains on A for 100 steps, which
    fills the buffer with A, then on B for 100 steps: once with a stratified
    replay step after every B step, once without. Without replay the loss on
    A climbs from about 0.1 back toward log(16) = 2.77, the loss of guessing;
    with replay it stays below where it was when A finished. Across 20 seeds
    on 2026-11-01 the ratio (A-loss without replay / A-loss with replay) was
    never below 38; the factor of 5 asserted here leaves room for a BLAS or
    torch change without being satisfiable by noise. B must be learned in the
    replay run too, so the test cannot pass by replay merely crowding out the
    new task.
    """
    in_vocab = 16

    def run(replay: bool) -> tuple[float, float]:
        torch.manual_seed(0)
        config = FoundationModelConfig.tiny()
        config.vocab_size = 256
        config.learning_rate = 1e-3
        model = FoundationModel(config)
        trainer = Trainer(model, config)
        for _ in range(100):
            trainer.train_step(*_cycle_batch(0, in_vocab))
        for _ in range(100):
            trainer.train_step(*_cycle_batch(in_vocab, in_vocab))
            if replay:
                trainer.replay_step(batch_size=4, strategy="stratified")
        return _mean_loss(model, 0, in_vocab), _mean_loss(model, in_vocab, in_vocab)

    a_without, _ = run(replay=False)
    a_with, b_with = run(replay=True)
    assert a_without > 5 * a_with, f"A-loss without replay {a_without:.3f}, with {a_with:.3f}"
    assert b_with < 0.5, f"B was not learned alongside replay: {b_with:.3f}"
