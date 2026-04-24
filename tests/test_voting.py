"""Tests for expertise-weighted voting and aggregation."""

import pytest
import torch

from foundation_model import (
    ContributorStats,
    DiscreteVote,
    ExpertiseWeightedAggregator,
    FoundationModel,
    FoundationModelConfig,
    ReputationTracker,
    expertise_weight,
)


@pytest.fixture
def tiny_base() -> dict:
    config = FoundationModelConfig.tiny()
    model = FoundationModel(config)
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


# ------------------------------------------------------------------
# expertise_weight
# ------------------------------------------------------------------


def test_lower_loss_gives_higher_weight() -> None:
    expert = ContributorStats(0, validation_loss=0.1, reputation=1.0)
    novice = ContributorStats(1, validation_loss=5.0, reputation=1.0)
    assert expertise_weight(expert) > expertise_weight(novice)


def test_higher_reputation_gives_higher_weight() -> None:
    a = ContributorStats(0, validation_loss=1.0, reputation=2.0)
    b = ContributorStats(1, validation_loss=1.0, reputation=0.5)
    assert expertise_weight(a) > expertise_weight(b)


def test_data_size_alone_does_not_determine_weight() -> None:
    """A contributor with more data but worse loss should weigh less."""
    expert_few = ContributorStats(0, n_samples=10, validation_loss=0.1, reputation=1.0)
    novice_many = ContributorStats(1, n_samples=10_000, validation_loss=5.0, reputation=1.0)
    assert expertise_weight(expert_few) > expertise_weight(novice_many)


# ------------------------------------------------------------------
# ReputationTracker
# ------------------------------------------------------------------


def test_positive_contribution_boosts_reputation() -> None:
    tracker = ReputationTracker(decay=1.0, boost=1.1, penalty=0.9)
    tracker.register(0)
    tracker.update(0, n_samples=10, validation_loss=1.0, contribution_score=0.5)
    assert tracker.get_stats(0).reputation > 1.0


def test_negative_contribution_penalises_reputation() -> None:
    tracker = ReputationTracker(decay=1.0, boost=1.1, penalty=0.9)
    tracker.register(0)
    tracker.update(0, n_samples=10, validation_loss=1.0, contribution_score=-0.5)
    assert tracker.get_stats(0).reputation < 1.0


def test_reputation_floor_enforced() -> None:
    tracker = ReputationTracker(decay=0.01, boost=1.0, penalty=0.01)
    tracker.register(0)
    for _ in range(100):
        tracker.update(0, n_samples=1, validation_loss=1.0, contribution_score=-1.0)
    assert tracker.get_stats(0).reputation >= 1e-4


# ------------------------------------------------------------------
# ExpertiseWeightedAggregator
# ------------------------------------------------------------------


def test_aggregate_closer_to_expert(tiny_base: dict) -> None:
    """Result should be closer to the expert update than to the novice update."""
    expert_state = {k: v + 0.1 for k, v in tiny_base.items()}
    novice_state = {k: v - 0.5 for k, v in tiny_base.items()}

    agg = ExpertiseWeightedAggregator()
    result = agg.aggregate(
        tiny_base,
        [
            (0, expert_state, 100, 0.1),   # expert: low loss
            (1, novice_state, 1000, 5.0),  # novice: high loss, more data
        ],
    )

    first_key = next(iter(tiny_base))
    dist_to_expert = (result[first_key].float() - expert_state[first_key].float()).abs().mean()
    dist_to_novice = (result[first_key].float() - novice_state[first_key].float()).abs().mean()
    assert dist_to_expert < dist_to_novice


def test_aggregate_empty_returns_base(tiny_base: dict) -> None:
    agg = ExpertiseWeightedAggregator()
    result = agg.aggregate(tiny_base, [])
    assert set(result.keys()) == set(tiny_base.keys())


def test_aggregate_with_trim_fraction(tiny_base: dict) -> None:
    agg = ExpertiseWeightedAggregator(trim_fraction=0.2)
    updates = [
        (i, {k: v + float(i) * 0.01 for k, v in tiny_base.items()}, 10, 1.0 + i * 0.1)
        for i in range(5)
    ]
    result = agg.aggregate(tiny_base, updates)
    assert set(result.keys()) == set(tiny_base.keys())


def test_reputation_updated_after_aggregation(tiny_base: dict) -> None:
    tracker = ReputationTracker()
    agg = ExpertiseWeightedAggregator(reputation_tracker=tracker)
    state = {k: v.clone() for k, v in tiny_base.items()}
    agg.aggregate(tiny_base, [(0, state, 10, 1.0)])
    assert tracker.get_stats(0) is not None
    assert tracker.get_stats(0).n_rounds == 1


# ------------------------------------------------------------------
# DiscreteVote
# ------------------------------------------------------------------


def test_expert_outvotes_majority() -> None:
    """One expert should beat two novices in expertise-weighted voting."""
    tracker = ReputationTracker()
    s0 = tracker.register(0)
    s0.validation_loss = 0.1
    s0.reputation = 10.0
    s1 = tracker.register(1)
    s1.validation_loss = 5.0
    s1.reputation = 0.5
    s2 = tracker.register(2)
    s2.validation_loss = 5.0
    s2.reputation = 0.5

    vote = DiscreteVote(tracker)
    winner, scores = vote.tally([
        (0, "option_a"),  # expert
        (1, "option_b"),  # novice
        (2, "option_b"),  # novice
    ])
    assert winner == "option_a", "Expert's option should win despite being outvoted 2:1"


def test_majority_wins_with_equal_expertise() -> None:
    """When all voters have equal expertise, the majority option wins."""
    vote = DiscreteVote()
    winner, _ = vote.tally([
        (0, "option_a"),
        (1, "option_a"),
        (2, "option_b"),
    ])
    assert winner == "option_a"


def test_unknown_contributor_gets_default_weight() -> None:
    """A contributor not in the tracker should get a default weight of 1."""
    vote = DiscreteVote()
    winner, scores = vote.tally([(99, "option_x")])
    assert winner == "option_x"
    assert scores["option_x"] == pytest.approx(1.0)
