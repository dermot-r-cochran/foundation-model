"""Experience replay buffer with representative sub-sampling support."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import torch

from .sampling import StratifiedSampler


@dataclass
class Experience:
    """A single stored experience."""

    input_ids: torch.Tensor
    labels: torch.Tensor
    priority: float = 1.0
    metadata: Dict = field(default_factory=dict)


SamplingStrategy = Literal["priority", "stratified", "uniform"]


class ExperienceBuffer:
    """Fixed-capacity experience replay buffer for continual learning.

    Supports three sampling strategies:

    ``"priority"``
        Weighted random sampling — high-priority (high-loss) experiences
        are drawn more frequently.  Good when the overall distribution is
        reasonably covered and you want to focus on hard cases.

    ``"stratified"``
        Stratified sub-sampling via :class:`~foundation_model.sampling.StratifiedSampler`.
        Divides the buffer into priority quartile buckets and samples equally
        from each, guaranteeing that both easy and hard experiences are
        represented regardless of how many of each are stored.  Recommended
        for large buffers with skewed priority distributions.

    ``"uniform"``
        Plain uniform random sampling without replacement.

    Parameters
    ----------
    capacity:
        Maximum number of experiences to keep (ring buffer — oldest entries
        are evicted when the buffer is full).
    n_strata:
        Number of quantile strata used by the ``"stratified"`` strategy.
    """

    def __init__(self, capacity: int = 10_000, n_strata: int = 4) -> None:
        self.capacity = capacity
        self.n_strata = n_strata
        self._buffer: deque[Experience] = deque(maxlen=capacity)
        self._total_added: int = 0
        self._stratified_sampler = StratifiedSampler(n_strata=n_strata)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        priority: float = 1.0,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Store a new experience."""
        self._buffer.append(
            Experience(
                input_ids=input_ids.detach().cpu(),
                labels=labels.detach().cpu(),
                priority=max(priority, 1e-6),
                metadata=metadata or {},
            )
        )
        self._total_added += 1

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        batch_size: int,
        strategy: SamplingStrategy = "priority",
        generator: Optional[torch.Generator] = None,
    ) -> List[Experience]:
        """Sample a list of :class:`Experience` objects from the buffer.

        Parameters
        ----------
        batch_size:
            Number of experiences to sample.
        strategy:
            One of ``"priority"``, ``"stratified"``, or ``"uniform"``.
        generator:
            Optional :class:`torch.Generator` used by the stratified sampler.
        """
        n = len(self._buffer)
        if n == 0:
            return []
        batch_size = min(batch_size, n)

        buf = list(self._buffer)

        if strategy == "stratified":
            indices = self._stratified_sampler.sample(
                indices=list(range(n)),
                priorities=[e.priority for e in buf],
                k=batch_size,
                generator=generator,
            )
            return [buf[i] for i in indices]

        if strategy == "priority":
            priorities = [e.priority for e in buf]
            total = sum(priorities)
            weights = [p / total for p in priorities]
            return random.choices(buf, weights=weights, k=batch_size)

        # uniform
        return random.sample(buf, batch_size)

    def sample_batch(
        self,
        batch_size: int,
        strategy: SamplingStrategy = "priority",
        generator: Optional[torch.Generator] = None,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Sample and return ``(input_ids, labels)`` stacked tensors.

        Returns ``None`` when the buffer is empty.

        Parameters
        ----------
        batch_size:
            Number of experiences to stack.
        strategy:
            Sampling strategy — see :meth:`sample`.
        generator:
            Optional :class:`torch.Generator` for stratified sampling.
        """
        experiences = self.sample(batch_size, strategy=strategy, generator=generator)
        if not experiences:
            return None
        input_ids = torch.stack([e.input_ids for e in experiences])
        labels = torch.stack([e.labels for e in experiences])
        return input_ids, labels

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def is_ready(self, min_samples: int = 1) -> bool:
        """Return ``True`` when the buffer holds at least *min_samples* entries."""
        return len(self._buffer) >= min_samples

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def total_added(self) -> int:
        """Cumulative number of experiences ever added (ignores evictions)."""
        return self._total_added
