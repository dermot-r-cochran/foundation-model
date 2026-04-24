"""Tests for ExperienceBuffer and its sampling strategies."""

import pytest
import torch

from foundation_model import ExperienceBuffer


@pytest.fixture
def populated_buffer() -> ExperienceBuffer:
    buf = ExperienceBuffer(capacity=50, n_strata=4)
    for i in range(20):
        ids = torch.randint(0, 100, (4, 8))
        labels = torch.randint(0, 100, (4, 8))
        buf.add(ids, labels, priority=float(i + 1))
    return buf


# ------------------------------------------------------------------
# Basic add / len
# ------------------------------------------------------------------


def test_add_increments_len() -> None:
    buf = ExperienceBuffer(capacity=100)
    ids = torch.randint(0, 100, (4, 8))
    labels = torch.randint(0, 100, (4, 8))
    buf.add(ids, labels, priority=1.0)
    assert len(buf) == 1
    assert buf.total_added == 1


def test_capacity_acts_as_ring_buffer() -> None:
    buf = ExperienceBuffer(capacity=5)
    for _ in range(10):
        buf.add(torch.zeros(1, 4, dtype=torch.long), torch.zeros(1, 4, dtype=torch.long))
    assert len(buf) == 5
    assert buf.total_added == 10


def test_is_ready() -> None:
    buf = ExperienceBuffer(capacity=10)
    assert not buf.is_ready(1)
    buf.add(torch.zeros(1, 4, dtype=torch.long), torch.zeros(1, 4, dtype=torch.long))
    assert buf.is_ready(1)
    assert not buf.is_ready(2)


# ------------------------------------------------------------------
# Sampling strategies
# ------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["priority", "stratified", "uniform"])
def test_sample_returns_correct_count(populated_buffer: ExperienceBuffer, strategy: str) -> None:
    samples = populated_buffer.sample(6, strategy=strategy)
    assert len(samples) == 6


@pytest.mark.parametrize("strategy", ["priority", "stratified", "uniform"])
def test_sample_batch_shapes(populated_buffer: ExperienceBuffer, strategy: str) -> None:
    batch = populated_buffer.sample_batch(4, strategy=strategy)
    assert batch is not None
    input_ids, labels = batch
    assert input_ids.shape[0] == 4
    assert labels.shape[0] == 4


def test_sample_batch_returns_none_when_empty() -> None:
    buf = ExperienceBuffer(capacity=10)
    assert buf.sample_batch(4) is None


def test_sample_does_not_exceed_buffer_size() -> None:
    buf = ExperienceBuffer(capacity=10)
    for _ in range(3):
        buf.add(torch.zeros(1, 4, dtype=torch.long), torch.zeros(1, 4, dtype=torch.long))
    samples = buf.sample(100, strategy="uniform")
    assert len(samples) == 3


# ------------------------------------------------------------------
# Stratified sampling covers all strata
# ------------------------------------------------------------------


def test_stratified_sampling_covers_priority_range() -> None:
    """Stratified sampling should select from both low- and high-priority ends."""
    buf = ExperienceBuffer(capacity=200, n_strata=4)
    low_priority_ids, high_priority_ids = set(), set()

    for i in range(100):
        ids = torch.full((1, 4), i, dtype=torch.long)
        labels = torch.zeros(1, 4, dtype=torch.long)
        priority = 0.01 if i < 50 else 100.0  # half low, half high priority
        buf.add(ids, labels, priority=priority)
        if i < 50:
            low_priority_ids.add(i)
        else:
            high_priority_ids.add(i)

    # Sample 20 items with stratified strategy
    g = torch.Generator()
    g.manual_seed(42)
    batch = buf.sample(20, strategy="stratified", generator=g)
    sampled_vals = {int(e.input_ids[0, 0].item()) for e in batch}

    # Stratified sampling must include at least one item from each priority tier
    assert sampled_vals & low_priority_ids, "No low-priority samples selected"
    assert sampled_vals & high_priority_ids, "No high-priority samples selected"
