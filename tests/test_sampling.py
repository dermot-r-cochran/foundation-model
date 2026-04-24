"""Tests for representative sub-sampling strategies."""

import pytest
import torch

from foundation_model import (
    DiversitySampler,
    RepresentativeSampler,
    StratifiedSampler,
    UncertaintySampler,
)


# ------------------------------------------------------------------
# StratifiedSampler
# ------------------------------------------------------------------


def test_stratified_returns_correct_count() -> None:
    sampler = StratifiedSampler(n_strata=4)
    indices = list(range(40))
    priorities = [float(i) for i in range(40)]
    chosen = sampler.sample(indices, priorities, k=16)
    assert len(chosen) == 16


def test_stratified_selects_from_all_strata() -> None:
    """Every priority quartile should contribute at least one sample."""
    sampler = StratifiedSampler(n_strata=4)
    indices = list(range(40))
    # four clear tiers: 0-9, 10-19, 20-29, 30-39
    priorities = [float(i) for i in range(40)]
    g = torch.Generator()
    g.manual_seed(0)
    chosen = sampler.sample(indices, priorities, k=16, generator=g)
    assert any(i < 10 for i in chosen), "Lowest stratum not represented"
    assert any(i >= 30 for i in chosen), "Highest stratum not represented"


def test_stratified_k_larger_than_pool() -> None:
    sampler = StratifiedSampler(n_strata=4)
    indices = list(range(5))
    chosen = sampler.sample(indices, [1.0] * 5, k=100)
    assert len(chosen) == 5


def test_stratified_empty_pool() -> None:
    sampler = StratifiedSampler(n_strata=4)
    assert sampler.sample([], [], k=10) == []


def test_stratified_uniform_priorities() -> None:
    """Uniform priorities should still return the requested k samples."""
    sampler = StratifiedSampler(n_strata=4)
    indices = list(range(20))
    chosen = sampler.sample(indices, [1.0] * 20, k=8)
    assert len(chosen) == 8


# ------------------------------------------------------------------
# DiversitySampler
# ------------------------------------------------------------------


def test_diversity_returns_correct_count() -> None:
    sampler = DiversitySampler()
    emb = torch.randn(20, 8)
    chosen = sampler.sample(emb, k=5)
    assert len(chosen) == 5


def test_diversity_no_duplicates() -> None:
    sampler = DiversitySampler()
    emb = torch.randn(20, 8)
    chosen = sampler.sample(emb, k=10)
    assert len(set(chosen)) == 10


def test_diversity_k_larger_than_pool() -> None:
    sampler = DiversitySampler()
    emb = torch.randn(5, 8)
    chosen = sampler.sample(emb, k=100)
    assert len(chosen) == 5


def test_diversity_maximises_spread() -> None:
    """Selected points should be spread further than a random sample."""
    torch.manual_seed(0)
    sampler = DiversitySampler()
    # Points clustered in two groups: first 10 near 0, next 10 near 10
    emb = torch.cat([torch.randn(10, 2) * 0.01, torch.randn(10, 2) * 0.01 + 10.0])
    chosen = sampler.sample(emb, k=4)
    # Should select points from both clusters
    from_low = sum(1 for i in chosen if i < 10)
    from_high = sum(1 for i in chosen if i >= 10)
    assert from_low > 0 and from_high > 0, "Diversity sampler failed to span clusters"


# ------------------------------------------------------------------
# UncertaintySampler
# ------------------------------------------------------------------


def test_uncertainty_returns_top_k() -> None:
    sampler = UncertaintySampler()
    uncertainties = [0.1, 0.9, 0.3, 0.8, 0.5]
    indices = list(range(5))
    chosen = sampler.sample(uncertainties, indices, k=2)
    assert set(chosen) == {1, 3}  # indices of 0.9 and 0.8


def test_uncertainty_k_larger_than_pool() -> None:
    sampler = UncertaintySampler()
    chosen = sampler.sample([1.0, 2.0], [0, 1], k=100)
    assert len(chosen) == 2


# ------------------------------------------------------------------
# RepresentativeSampler
# ------------------------------------------------------------------


def test_representative_returns_correct_count() -> None:
    sampler = RepresentativeSampler(diversity_fraction=0.5)
    emb = torch.randn(30, 8)
    uncertainties = [float(i) / 30 for i in range(30)]
    indices = list(range(30))
    chosen = sampler.sample(emb, uncertainties, indices, k=10)
    assert len(chosen) == 10


def test_representative_no_duplicates() -> None:
    sampler = RepresentativeSampler(diversity_fraction=0.5)
    emb = torch.randn(30, 8)
    uncertainties = [float(i) / 30 for i in range(30)]
    indices = list(range(30))
    chosen = sampler.sample(emb, uncertainties, indices, k=10)
    assert len(set(chosen)) == len(chosen)


def test_representative_diversity_fraction_zero() -> None:
    """All budget goes to uncertainty sampling when fraction=0."""
    sampler = RepresentativeSampler(diversity_fraction=0.0)
    emb = torch.randn(20, 4)
    # Top-3 highest uncertainty indices are 19, 18, 17
    uncertainties = list(range(20))
    indices = list(range(20))
    chosen = sampler.sample(emb, uncertainties, indices, k=3)
    assert set(chosen) == {17, 18, 19}


def test_representative_invalid_fraction() -> None:
    with pytest.raises(ValueError):
        RepresentativeSampler(diversity_fraction=1.5)
