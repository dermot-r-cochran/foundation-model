"""Representative sub-sampling strategies for large datasets.

When a dataset is too large to process in full, these samplers select
a representative subset that preserves the statistical properties of
the whole dataset better than uniform random sampling.

Strategies
----------
StratifiedSampler
    Buckets samples by priority/loss quartile and draws equally from each
    bucket, ensuring coverage of both easy and hard examples.

DiversitySampler
    Greedy k-centers coreset: selects k points that maximise the minimum
    distance between any two selected embeddings.  Guarantees that the
    chosen subset covers the full feature space as uniformly as possible.

UncertaintySampler
    Selects the k samples with the highest epistemic uncertainty, focusing
    the training budget on the examples the model finds most informative.

RepresentativeSampler
    Combines diversity and uncertainty into a single balanced selection
    controlled by ``diversity_fraction``.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _bucket_by_quantile(values: List[float], n_buckets: int) -> List[int]:
    """Assign each value to a bucket index (0 … n_buckets-1) by quantile."""
    n = len(values)
    if n == 0:
        return []
    sorted_idx = sorted(range(n), key=lambda i: values[i])
    labels = [0] * n
    bucket_size = n / n_buckets
    for rank, original_idx in enumerate(sorted_idx):
        labels[original_idx] = min(int(rank / bucket_size), n_buckets - 1)
    return labels


# ---------------------------------------------------------------------------
# Public samplers
# ---------------------------------------------------------------------------


class StratifiedSampler:
    """Stratified sub-sampling over priority/loss strata.

    The pool is divided into ``n_strata`` equally-sized quantile buckets
    (from lowest to highest priority).  ``k`` samples are then drawn with
    equal quota from every bucket, so rare high-priority examples are
    represented at the same rate as the more abundant low-priority ones.

    Parameters
    ----------
    n_strata:
        Number of priority strata / quantile buckets.
    """

    def __init__(self, n_strata: int = 4) -> None:
        if n_strata < 1:
            raise ValueError("n_strata must be >= 1")
        self.n_strata = n_strata

    def sample(
        self,
        indices: Sequence[int],
        priorities: Sequence[float],
        k: int,
        generator: Optional[torch.Generator] = None,
    ) -> List[int]:
        """Return *k* indices selected by stratified sampling.

        Parameters
        ----------
        indices:
            Pool of candidate indices.
        priorities:
            Priority score for each index (e.g. training loss).
        k:
            Number of samples to draw.
        generator:
            Optional :class:`torch.Generator` for reproducibility.
        """
        idx_list = list(indices)
        pri_list = list(priorities)
        k = min(k, len(idx_list))
        if k == 0:
            return []

        labels = _bucket_by_quantile(pri_list, self.n_strata)

        # Group indices by stratum
        buckets: dict[int, List[int]] = {}
        for orig_idx, stratum in zip(idx_list, labels):
            buckets.setdefault(stratum, []).append(orig_idx)

        n_present = len(buckets)
        base_quota = k // n_present
        remainder = k - base_quota * n_present

        selected: List[int] = []
        for i, (_, bucket) in enumerate(sorted(buckets.items())):
            quota = min(base_quota + (1 if i < remainder else 0), len(bucket))
            perm = torch.randperm(len(bucket), generator=generator).tolist()
            selected.extend(bucket[j] for j in perm[:quota])

        # Top-up if integer division left us short
        if len(selected) < k:
            remaining = list(set(idx_list) - set(selected))
            extra = min(k - len(selected), len(remaining))
            perm = torch.randperm(len(remaining), generator=generator).tolist()
            selected.extend(remaining[j] for j in perm[:extra])

        return selected[:k]


class DiversitySampler:
    """Greedy k-centers coreset selection.

    Selects *k* samples whose embeddings maximise the minimum pairwise
    distance, guaranteeing that the chosen subset covers the full
    feature space as uniformly as possible.

    Reference: Sener & Savarese, "Active Learning for Convolutional Neural
    Networks: A Core-Set Approach", ICLR 2018.
    """

    def sample(
        self,
        embeddings: torch.Tensor,
        k: int,
        seed_idx: int = 0,
    ) -> List[int]:
        """Return *k* diverse indices from *embeddings*.

        Parameters
        ----------
        embeddings:
            ``[N, D]`` float tensor of feature vectors.
        k:
            Number of samples to select.
        seed_idx:
            Index of the first anchor point (default: 0).
        """
        N = embeddings.shape[0]
        k = min(k, N)
        if k == 0:
            return []

        emb = embeddings.float()
        selected = [seed_idx]

        # Distance of every point to its nearest already-selected centre
        dists = torch.cdist(emb, emb[seed_idx].unsqueeze(0)).squeeze(1)

        for _ in range(k - 1):
            next_idx = int(dists.argmax().item())
            selected.append(next_idx)
            new_d = torch.cdist(emb, emb[next_idx].unsqueeze(0)).squeeze(1)
            dists = torch.minimum(dists, new_d)

        return selected


class UncertaintySampler:
    """Uncertainty-driven sampling.

    Returns the *k* samples with the highest epistemic uncertainty,
    making them the most informative candidates for the next training step.
    """

    def sample(
        self,
        uncertainties: Sequence[float],
        indices: Sequence[int],
        k: int,
    ) -> List[int]:
        """Return the *k* highest-uncertainty indices.

        Parameters
        ----------
        uncertainties:
            Scalar epistemic uncertainty score per candidate.
        indices:
            Candidate indices corresponding to *uncertainties*.
        k:
            Number of samples to select.
        """
        k = min(k, len(indices))
        order = sorted(range(len(uncertainties)), key=lambda i: -uncertainties[i])
        return [indices[i] for i in order[:k]]


class RepresentativeSampler:
    """Combined representative sub-sampling.

    Splits the budget *k* between two complementary strategies:

    * **Diversity**   – greedy k-centers on sample embeddings ensures
      broad coverage of the feature space.
    * **Uncertainty** – highest-epistemic-uncertainty samples focus the
      budget on regions where the model is least confident.

    The split is controlled by ``diversity_fraction`` (default 0.5).

    Parameters
    ----------
    diversity_fraction:
        Fraction of *k* allocated to diversity sampling ``[0, 1]``.
    """

    def __init__(self, diversity_fraction: float = 0.5) -> None:
        if not 0.0 <= diversity_fraction <= 1.0:
            raise ValueError("diversity_fraction must be in [0, 1]")
        self.diversity_fraction = diversity_fraction
        self._diversity = DiversitySampler()
        self._uncertainty = UncertaintySampler()

    def sample(
        self,
        embeddings: torch.Tensor,
        uncertainties: Sequence[float],
        indices: Sequence[int],
        k: int,
        seed_idx: int = 0,
    ) -> List[int]:
        """Select *k* representative indices.

        Parameters
        ----------
        embeddings:
            ``[N, D]`` feature tensor for diversity sampling.
        uncertainties:
            Per-sample epistemic uncertainty scores.
        indices:
            Original dataset indices for the *N* candidates.
        k:
            Total number of samples to select.
        seed_idx:
            Seed anchor for the diversity sampler.
        """
        k = min(k, len(indices))
        if k == 0:
            return []

        # When diversity_fraction == 0 skip the diversity step entirely so
        # that the full budget goes to uncertainty sampling.
        k_div = round(k * self.diversity_fraction)
        k_unc = k - k_div

        if k_div > 0:
            diverse_pool = self._diversity.sample(embeddings, k_div, seed_idx)
            diverse_set: set[int] = {indices[i] for i in diverse_pool}
        else:
            diverse_set = set()

        # Fill the uncertainty quota from candidates not already selected
        all_uncertain = self._uncertainty.sample(
            list(uncertainties), list(indices), k_unc + k_div
        )
        uncertain_set = [i for i in all_uncertain if i not in diverse_set]

        combined = list(diverse_set) + uncertain_set
        return combined[:k]
