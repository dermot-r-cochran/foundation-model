"""Tests for ExperienceBuffer and its sampling strategies."""

import random

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


# ------------------------------------------------------------------
# Determinism: which seed governs which strategy
# ------------------------------------------------------------------


def _indexed_buffer() -> ExperienceBuffer:
    """Twenty entries whose input_ids carry their own index, priority i + 1."""
    buf = ExperienceBuffer(capacity=50, n_strata=4)
    for i in range(20):
        buf.add(
            torch.full((1, 4), i, dtype=torch.long),
            torch.zeros(1, 4, dtype=torch.long),
            priority=float(i + 1),
        )
    return buf


def test_stratified_replay_sample_reproduces_under_a_generator() -> None:
    """A stratified replay sample is reproducible from the generator alone.

    Twenty entries carry their index in input_ids. Two calls of
    `sample(6, strategy="stratified", generator=g)` use fresh generators with
    the same seed, while the global torch RNG and Python's `random` are seeded
    differently before each call. The index lists are compared exactly (ints).
    It turns red if ExperienceBuffer.sample stops passing `generator` on to
    the stratified sampler (the draws would then follow the global torch RNG,
    which differs between the calls).
    """
    buf = _indexed_buffer()
    runs = []
    for global_seed in (1, 2):
        torch.manual_seed(global_seed)
        random.seed(global_seed)
        g = torch.Generator()
        g.manual_seed(0)
        batch = buf.sample(6, strategy="stratified", generator=g)
        runs.append([int(e.input_ids[0, 0].item()) for e in batch])
    assert len(runs[0]) == 6
    assert runs[0] == runs[1]


@pytest.mark.parametrize("strategy", ["priority", "uniform"])
def test_priority_and_uniform_samples_reproduce_under_random_seed(strategy: str) -> None:
    """Priority and uniform replay follow Python's `random`, so `random.seed` reproduces them.

    These two strategies draw with `random.choices` and `random.sample`; the
    `generator` argument governs the stratified path only. Seeding `random`
    with 5 before each of two calls, with no generator, gives the same index
    list (exact, ints). It turns red if either strategy moves to a source of
    randomness that `random.seed` does not control, such as the global torch
    RNG. (It does not assert that the generator is ignored: that is a
    behaviour this repository may change.)
    """
    buf = _indexed_buffer()
    runs = []
    for _ in range(2):
        random.seed(5)
        batch = buf.sample(6, strategy=strategy)
        runs.append([int(e.input_ids[0, 0].item()) for e in batch])
    assert len(runs[0]) == 6
    assert runs[0] == runs[1]
