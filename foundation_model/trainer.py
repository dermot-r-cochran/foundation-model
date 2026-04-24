"""Training loop with experience replay and continual learning."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from .config import FoundationModelConfig
from .experience import ExperienceBuffer, SamplingStrategy
from .model import FoundationModel


class Trainer:
    """Trains a :class:`~foundation_model.model.FoundationModel` and manages
    an :class:`~foundation_model.experience.ExperienceBuffer` for continual
    learning via experience replay.

    Parameters
    ----------
    model:
        The model to train.
    config:
        Hyper-parameters (learning rate, weight decay, buffer capacity …).
    experience_buffer:
        An existing buffer to reuse; a fresh one is created if ``None``.
    """

    def __init__(
        self,
        model: FoundationModel,
        config: FoundationModelConfig,
        experience_buffer: Optional[ExperienceBuffer] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.experience_buffer = experience_buffer or ExperienceBuffer(
            config.buffer_capacity
        )
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self._step: int = 0

    # ------------------------------------------------------------------
    # Core training
    # ------------------------------------------------------------------

    def train_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        store_in_buffer: bool = True,
    ) -> Dict[str, float]:
        """One forward–backward–optimise step.

        Parameters
        ----------
        input_ids, labels:
            ``[B, T]`` integer tensors on any device.
        store_in_buffer:
            When ``True`` the batch is stored in the experience buffer with
            the training loss as its priority.

        Returns
        -------
        dict with ``"loss"`` and ``"step"`` keys.
        """
        self.model.train()
        self.optimizer.zero_grad()

        logits = self.model(input_ids)          # [B, T, V]
        B, T, V = logits.shape
        loss = self.criterion(logits.view(B * T, V), labels.view(B * T))
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self._step += 1

        if store_in_buffer:
            self.experience_buffer.add(input_ids, labels, priority=loss.item())

        return {"loss": loss.item(), "step": self._step}

    # ------------------------------------------------------------------
    # Experience replay
    # ------------------------------------------------------------------

    def replay_step(
        self,
        batch_size: int = 8,
        strategy: SamplingStrategy = "stratified",
    ) -> Optional[Dict[str, float]]:
        """Train on a batch replayed from the experience buffer.

        Uses *stratified* sampling by default so that both easy and hard
        past experiences are replayed in equal proportion, preventing
        catastrophic forgetting of rare but important patterns.

        Parameters
        ----------
        batch_size:
            Number of experiences to replay.
        strategy:
            Sampling strategy — ``"stratified"``, ``"priority"``, or
            ``"uniform"``.  See :class:`~foundation_model.experience.ExperienceBuffer`.

        Returns
        -------
        Training metrics dict, or ``None`` if the buffer is not ready.
        """
        if not self.experience_buffer.is_ready(batch_size):
            return None

        batch = self.experience_buffer.sample_batch(batch_size, strategy=strategy)
        if batch is None:
            return None

        device = next(self.model.parameters()).device
        input_ids, labels = batch
        # Stored experiences may have shape [B, T]; after stacking k of them we
        # get [k, B, T].  Flatten the leading two dims so the model always sees
        # a 2-D [N, T] tensor regardless of how the experiences were stored.
        T = input_ids.shape[-1]
        return self.train_step(
            input_ids.view(-1, T).to(device),
            labels.view(-1, T).to(device),
            store_in_buffer=False,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def step(self) -> int:
        """Total number of optimiser steps taken."""
        return self._step
