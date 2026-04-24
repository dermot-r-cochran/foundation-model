"""Expertise-weighted voting and model aggregation.

Rather than democratic majority rule — which can be dominated by many
low-quality participants — votes and model updates are weighted by each
contributor's *demonstrated expertise*: their validation loss and their
accumulated reputation score.

This avoids the "tyranny of the uneducated majority": a single
highly-calibrated contributor can out-weigh many poorly-performing ones.

Classes
-------
ContributorStats
    Mutable record of a contributor's quality metrics.
ReputationTracker
    Tracks and updates per-contributor reputation across rounds.
ExpertiseWeightedAggregator
    Aggregates model state-dicts weighted by expertise rather than by
    data-set size.  Optionally trims both tails (Byzantine robustness).
DiscreteVote
    Expertise-weighted tally for any categorical decision
    (e.g. hyperparameter selection, architecture variant choice).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Contributor statistics and reputation
# ---------------------------------------------------------------------------


@dataclass
class ContributorStats:
    """Quality metrics for a single contributor."""

    contributor_id: int
    n_samples: int = 0
    # Lower = better; 10.0 signals "not yet evaluated"
    validation_loss: float = 10.0
    # Accumulated expertise score; starts at 1.0 (neutral)
    reputation: float = 1.0
    n_rounds: int = 0


def expertise_weight(stats: ContributorStats, eps: float = 1e-8) -> float:
    """Scalar expertise weight for a contributor.

    ``weight = reputation / (validation_loss + eps)``

    This gives higher weight to contributors with low validation loss
    *and* a strong track record, regardless of their dataset size.
    """
    return stats.reputation / (stats.validation_loss + eps)


class ReputationTracker:
    """Tracks contributor reputations across federated / P2P rounds.

    After each aggregation round the tracker boosts a contributor's
    reputation when their update was aligned with the consensus
    (positive cosine similarity) and penalises it when misaligned.
    A per-round decay factor keeps the scores recency-sensitive.

    Parameters
    ----------
    decay:
        Multiplicative decay applied every round (encourages recency).
    boost:
        Additional multiplier applied on top of *decay* for a good update.
    penalty:
        Multiplier applied on top of *decay* for a bad update.
    """

    def __init__(
        self,
        decay: float = 0.9,
        boost: float = 1.1,
        penalty: float = 0.9,
    ) -> None:
        self.decay = decay
        self.boost = boost
        self.penalty = penalty
        self._stats: Dict[int, ContributorStats] = {}

    def register(self, contributor_id: int) -> ContributorStats:
        """Ensure *contributor_id* is present; return their stats."""
        if contributor_id not in self._stats:
            self._stats[contributor_id] = ContributorStats(contributor_id)
        return self._stats[contributor_id]

    def update(
        self,
        contributor_id: int,
        n_samples: int,
        validation_loss: float,
        contribution_score: float,
    ) -> None:
        """Update statistics for *contributor_id*.

        Parameters
        ----------
        n_samples:
            Number of local training samples used this round.
        validation_loss:
            Validation loss after local training (lower = better).
        contribution_score:
            Cosine similarity of this contributor's update with the
            round's aggregate update.  Positive means aligned (good),
            negative means opposing (suspicious / Byzantine).
        """
        stats = self.register(contributor_id)
        stats.n_samples = n_samples
        stats.validation_loss = validation_loss
        stats.n_rounds += 1
        multiplier = self.boost if contribution_score >= 0 else self.penalty
        stats.reputation = max(stats.reputation * self.decay * multiplier, 1e-4)

    def get_stats(self, contributor_id: int) -> Optional[ContributorStats]:
        return self._stats.get(contributor_id)

    def all_stats(self) -> Dict[int, ContributorStats]:
        return dict(self._stats)


# ---------------------------------------------------------------------------
# Internal: cosine similarity of model updates
# ---------------------------------------------------------------------------


def _cosine_similarity_of_updates(
    base: Dict[str, torch.Tensor],
    update: Dict[str, torch.Tensor],
    aggregate: Dict[str, torch.Tensor],
) -> float:
    """Cosine similarity between an individual delta and the aggregate delta."""

    def _delta_vec(state: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat(
            [(state[k].float() - base[k].float()).view(-1) for k in base]
        )

    d_u = _delta_vec(update)
    d_a = _delta_vec(aggregate)
    n_u, n_a = d_u.norm(), d_a.norm()
    if n_u < 1e-12 or n_a < 1e-12:
        return 0.0
    return (d_u @ d_a / (n_u * n_a)).item()


# ---------------------------------------------------------------------------
# Expertise-weighted model aggregation
# ---------------------------------------------------------------------------


class ExpertiseWeightedAggregator:
    """Aggregates model updates weighted by contributor expertise.

    Unlike plain FedAvg (which weights by dataset size), this aggregator
    weights contributions by ``reputation / validation_loss``.  A single
    highly-accurate contributor therefore has more influence than many
    poorly-performing ones.

    Optional Byzantine robustness: when ``trim_fraction > 0`` the
    lowest and highest-weighted contributors are excluded before
    averaging, guarding against both low-quality participants *and*
    manipulation by artificially inflated reputation.

    Parameters
    ----------
    reputation_tracker:
        Shared :class:`ReputationTracker`; a fresh one is created if
        ``None``.
    trim_fraction:
        Fraction of contributors to drop from each tail
        (0 → no trimming; 0.1 → drop lowest 10 % and highest 10 %).
    """

    def __init__(
        self,
        reputation_tracker: Optional[ReputationTracker] = None,
        trim_fraction: float = 0.0,
    ) -> None:
        self.reputation_tracker = reputation_tracker or ReputationTracker()
        self.trim_fraction = trim_fraction

    def aggregate(
        self,
        base_params: Dict[str, torch.Tensor],
        updates: List[Tuple[int, Dict[str, torch.Tensor], int, float]],
    ) -> Dict[str, torch.Tensor]:
        """Compute an expertise-weighted average of model state dicts.

        Parameters
        ----------
        base_params:
            Global model parameters *before* this round's updates.
        updates:
            List of ``(contributor_id, state_dict, n_samples, val_loss)``
            tuples — one per participating client.

        Returns
        -------
        Aggregated state dict.
        """
        if not updates:
            return base_params

        # Seed / refresh contributor stats from this round's data
        for cid, _, n, loss in updates:
            stats = self.reputation_tracker.register(cid)
            stats.n_samples = n
            stats.validation_loss = loss

        # Expertise weights
        weights = [
            expertise_weight(self.reputation_tracker.get_stats(cid))
            for cid, *_ in updates
        ]

        # Optional tail-trimming for Byzantine robustness
        if self.trim_fraction > 0.0:
            k = max(1, int(len(updates) * self.trim_fraction))
            order = sorted(range(len(weights)), key=lambda i: weights[i])
            keep = order[k : len(order) - k] if len(order) - 2 * k > 0 else order
            updates = [updates[i] for i in keep]
            weights = [weights[i] for i in keep]

        if not updates:
            return base_params

        total = sum(weights)
        normalised = [w / total for w in weights]

        aggregated: Dict[str, torch.Tensor] = {
            key: torch.stack(
                [normalised[i] * updates[i][1][key].float() for i in range(len(updates))]
            ).sum(dim=0)
            for key in base_params
        }

        # Update reputations using cosine similarity vs the consensus
        for cid, state, n, loss in updates:
            score = _cosine_similarity_of_updates(base_params, state, aggregated)
            self.reputation_tracker.update(cid, n, loss, score)

        return aggregated


# ---------------------------------------------------------------------------
# Expertise-weighted discrete voting
# ---------------------------------------------------------------------------


class DiscreteVote:
    """Expertise-weighted tally for categorical decisions.

    Each voter casts one option.  The winning option is the one whose
    voters' expertise weights sum to the most — *not* the option with
    the most votes by count.  A single expert can therefore out-vote a
    majority of poorly-performing participants.

    Parameters
    ----------
    reputation_tracker:
        Shared :class:`ReputationTracker`; a fresh one is created if
        ``None``.
    """

    def __init__(
        self, reputation_tracker: Optional[ReputationTracker] = None
    ) -> None:
        self.reputation_tracker = reputation_tracker or ReputationTracker()

    def tally(
        self,
        votes: List[Tuple[int, str]],
    ) -> Tuple[str, Dict[str, float]]:
        """Tally votes and return the winning option.

        Parameters
        ----------
        votes:
            List of ``(contributor_id, option)`` pairs.  Contributors
            must have been registered in the :class:`ReputationTracker`
            beforehand (unknown contributors receive a default weight of 1).

        Returns
        -------
        winner:
            The option with the highest total expertise weight.
        scores:
            Mapping from option → total expertise weight.
        """
        scores: Dict[str, float] = {}
        for cid, option in votes:
            stats = self.reputation_tracker.get_stats(cid)
            w = expertise_weight(stats) if stats is not None else 1.0
            scores[option] = scores.get(option, 0.0) + w
        winner = max(scores, key=scores.__getitem__)
        return winner, scores
