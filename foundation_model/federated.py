"""Federated learning: FedAvg and expertise-weighted aggregation.

Topology
--------
Server  ──► broadcasts global model to selected clients
Clients ──► train locally (optionally on a representative sub-sample)
Clients ──► return updated parameters + sample count to the server
Server  ──► aggregates via FedAvg *or* ExpertiseWeightedAggregator

Representative sub-sampling
---------------------------
When a client has a large local dataset it can call :meth:`FederatedClient.fit`
with ``max_samples`` set.  The client then uses :class:`StratifiedSampler`
to select a representative subset that covers both easy and hard examples
proportionally, rather than feeding redundant or skewed batches to the
global model.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn

from .config import FoundationModelConfig
from .model import FoundationModel
from .sampling import StratifiedSampler
from .trainer import Trainer
from .voting import ExpertiseWeightedAggregator, ReputationTracker


# ---------------------------------------------------------------------------
# Plain FedAvg helper (data-size weighted)
# ---------------------------------------------------------------------------


def fedavg_aggregate(
    updates: List[Dict[str, torch.Tensor]],
    weights: List[float],
) -> Dict[str, torch.Tensor]:
    """Weighted FedAvg aggregation (weight by dataset size).

    Parameters
    ----------
    updates:
        List of client state dicts.
    weights:
        Non-negative weight for each client (typically ``n_samples``).

    Returns
    -------
    Aggregated state dict.
    """
    assert len(updates) == len(weights) and len(updates) > 0
    total = sum(weights)
    normalised = [w / total for w in weights]
    return {
        key: torch.stack(
            [normalised[i] * updates[i][key].float() for i in range(len(updates))]
        ).sum(dim=0)
        for key in updates[0]
    }


# ---------------------------------------------------------------------------
# Federated client
# ---------------------------------------------------------------------------

AggregationStrategy = Literal["fedavg", "expertise"]


class FederatedClient:
    """A federated learning client.

    Holds a private local model, trains on private data, and returns
    model updates to the server.  When the local dataset is large,
    :meth:`fit` can be told to sub-sample a representative subset
    before training.

    Parameters
    ----------
    client_id:
        Unique integer identifier.
    model_config:
        Architecture / training hyper-parameters.
    """

    def __init__(
        self,
        client_id: int,
        model_config: FoundationModelConfig,
    ) -> None:
        self.client_id = client_id
        self.model_config = model_config
        self.model = FoundationModel(model_config)
        self.trainer = Trainer(self.model, model_config)
        self._stratified_sampler = StratifiedSampler()

    # ------------------------------------------------------------------
    # Parameter exchange
    # ------------------------------------------------------------------

    def set_parameters(self, state: Dict[str, torch.Tensor]) -> None:
        """Overwrite local model weights with the server's global model."""
        self.model.load_state_dict(state, strict=True)

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Return a detached copy of the current local model weights."""
        return {k: v.detach().clone() for k, v in self.model.state_dict().items()}

    # ------------------------------------------------------------------
    # Local training
    # ------------------------------------------------------------------

    def _select_representative_subset(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        max_samples: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Down-sample *input_ids* / *labels* to *max_samples* rows using
        stratified sampling.

        A quick per-row proxy loss (mean absolute token index) is used as
        the priority signal so that more "extreme" (unusual) sequences are
        as likely to be selected as common ones.
        """
        N = input_ids.shape[0]
        # Proxy priority: variance of token IDs per row (higher = more unusual)
        priorities = input_ids.float().var(dim=-1).tolist()
        chosen = self._stratified_sampler.sample(
            indices=list(range(N)),
            priorities=priorities,
            k=max_samples,
        )
        return input_ids[chosen], labels[chosen]

    def fit(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        epochs: int = 1,
        max_samples: Optional[int] = None,
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        """Train locally and return updated parameters + number of samples.

        Parameters
        ----------
        input_ids, labels:
            Full local training tensors ``[N, T]``.
        epochs:
            Number of local passes over the (sub-sampled) data.
        max_samples:
            When set and ``N > max_samples``, a representative sub-sample
            of size *max_samples* is selected via stratified sampling before
            training.  This prevents large but skewed local datasets from
            dominating the federated update.

        Returns
        -------
        updated_params:
            State dict after local training.
        n_samples:
            Number of training rows actually used.
        """
        N = input_ids.shape[0]
        if max_samples is not None and N > max_samples:
            input_ids, labels = self._select_representative_subset(
                input_ids, labels, max_samples
            )

        for _ in range(epochs):
            self.trainer.train_step(input_ids, labels, store_in_buffer=True)

        return self.get_parameters(), input_ids.shape[0]


# ---------------------------------------------------------------------------
# Federated server
# ---------------------------------------------------------------------------


class FederatedConfig:
    """Configuration knobs for the federated learning coordinator.

    Parameters
    ----------
    n_clients:
        Total number of clients in the federation.
    local_epochs:
        Default number of local training epochs per round.
    strategy:
        ``"fedavg"``    – plain data-size weighted FedAvg.
        ``"expertise"`` – expertise-weighted aggregation; requires a
        :class:`~foundation_model.voting.ReputationTracker`.
    trim_fraction:
        Fraction of contributors to exclude from each tail when
        ``strategy="expertise"`` (Byzantine robustness).
    """

    def __init__(
        self,
        n_clients: int = 10,
        local_epochs: int = 1,
        strategy: AggregationStrategy = "fedavg",
        trim_fraction: float = 0.0,
    ) -> None:
        self.n_clients = n_clients
        self.local_epochs = local_epochs
        self.strategy = strategy
        self.trim_fraction = trim_fraction


class FederatedServer:
    """Coordinates federated training across multiple clients.

    Supports two aggregation strategies (controlled via
    :class:`FederatedConfig`):

    ``"fedavg"``
        Classic FedAvg — weight each client's update by the number of
        local training samples.

    ``"expertise"``
        Expertise-weighted aggregation via
        :class:`~foundation_model.voting.ExpertiseWeightedAggregator` —
        clients with lower validation loss and stronger reputation
        contribute more to the global model, regardless of dataset size.
        This prevents a large coalition of poorly-performing clients from
        degrading the global model.

    Parameters
    ----------
    model_config:
        Architecture configuration for the global model.
    fed_config:
        Federated learning settings.
    reputation_tracker:
        Shared reputation store; auto-created when ``None``.
    """

    def __init__(
        self,
        model_config: FoundationModelConfig,
        fed_config: Optional[FederatedConfig] = None,
        reputation_tracker: Optional[ReputationTracker] = None,
    ) -> None:
        self.model_config = model_config
        self.fed_config = fed_config or FederatedConfig()
        self.global_model = FoundationModel(model_config)
        self._round: int = 0

        tracker = reputation_tracker or ReputationTracker()
        self._expertise_agg = ExpertiseWeightedAggregator(
            reputation_tracker=tracker,
            trim_fraction=self.fed_config.trim_fraction,
        )

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Return a detached copy of the current global model weights."""
        return {k: v.detach().clone() for k, v in self.global_model.state_dict().items()}

    def aggregate(
        self,
        client_updates: List[Tuple[Dict[str, torch.Tensor], int]],
        val_losses: Optional[List[float]] = None,
    ) -> None:
        """Aggregate client updates into the global model.

        Parameters
        ----------
        client_updates:
            List of ``(state_dict, n_samples)`` from each participating client.
        val_losses:
            Per-client validation loss, required when
            ``fed_config.strategy == "expertise"``.  Ignored for ``"fedavg"``.
        """
        if not client_updates:
            return

        base = self.get_parameters()

        if self.fed_config.strategy == "expertise":
            if val_losses is None:
                val_losses = [10.0] * len(client_updates)
            updates_with_meta = [
                (i, state, n, loss)
                for i, ((state, n), loss) in enumerate(
                    zip(client_updates, val_losses)
                )
            ]
            aggregated = self._expertise_agg.aggregate(base, updates_with_meta)
        else:
            states = [u[0] for u in client_updates]
            weights = [float(u[1]) for u in client_updates]
            aggregated = fedavg_aggregate(states, weights)

        self.global_model.load_state_dict(aggregated, strict=True)
        self._round += 1

    @property
    def round(self) -> int:
        """Current federation round number."""
        return self._round
