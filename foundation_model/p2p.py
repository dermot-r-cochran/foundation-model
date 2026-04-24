"""Peer-to-peer distributed learning via gossip-based model averaging.

Each peer maintains a local model and periodically exchanges parameters
with a randomly selected neighbour.  Over many rounds all peers converge
toward the global mean without any central coordinator.

Expertise-weighted gossip
-------------------------
When peers have measured their local validation loss, the gossip step can
be weighted by expertise: a peer with *lower* validation loss has more
influence over its neighbour during exchange.  This prevents many novice
peers from diluting the knowledge of a well-trained minority.

Reference: Boyd et al., "Randomized Gossip Algorithms", IEEE Trans. Inf.
Theory 52(6), 2006.
"""

from __future__ import annotations

import random
import threading
from typing import Dict, List, Optional

import torch

from .config import FoundationModelConfig
from .model import FoundationModel
from .trainer import Trainer


# ---------------------------------------------------------------------------
# Internal blend helper
# ---------------------------------------------------------------------------


def _blend(
    state_a: Dict[str, torch.Tensor],
    state_b: Dict[str, torch.Tensor],
    alpha: float,
) -> Dict[str, torch.Tensor]:
    """Return ``alpha * state_a + (1 - alpha) * state_b``."""
    return {
        k: alpha * state_a[k].float() + (1.0 - alpha) * state_b[k].float()
        for k in state_a
    }


# ---------------------------------------------------------------------------
# Peer
# ---------------------------------------------------------------------------


class Peer:
    """A single node in a P2P learning network.

    Each peer:

    * Maintains a local :class:`~foundation_model.model.FoundationModel`.
    * Trains on private local data via its own :class:`~foundation_model.trainer.Trainer`.
    * Exchanges model parameters with neighbours through gossip averaging.

    Parameters
    ----------
    peer_id:
        Unique integer identifier.
    model_config:
        Architecture / training hyper-parameters.
    gossip_alpha:
        Default mixing coefficient used when ``val_loss`` is unavailable.
        ``alpha = 0.5`` → equal blend.
    """

    def __init__(
        self,
        peer_id: int,
        model_config: FoundationModelConfig,
        gossip_alpha: float = 0.5,
    ) -> None:
        self.peer_id = peer_id
        self.gossip_alpha = gossip_alpha
        self.val_loss: Optional[float] = None  # set after local training
        self.model = FoundationModel(model_config)
        self.trainer = Trainer(self.model, model_config)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Parameter exchange
    # ------------------------------------------------------------------

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        with self._lock:
            return {k: v.detach().clone() for k, v in self.model.state_dict().items()}

    def set_parameters(self, state: Dict[str, torch.Tensor]) -> None:
        with self._lock:
            self.model.load_state_dict(state, strict=True)

    # ------------------------------------------------------------------
    # Local training
    # ------------------------------------------------------------------

    def train_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, float]:
        """One local training step; updates ``val_loss`` from the batch loss."""
        with self._lock:
            result = self.trainer.train_step(input_ids, labels)
        # Use training loss as a proxy for val_loss (updated each step)
        self.val_loss = result["loss"]
        return result

    # ------------------------------------------------------------------
    # Gossip
    # ------------------------------------------------------------------

    def gossip_with(
        self,
        other: "Peer",
        expertise_weighted: bool = False,
    ) -> None:
        """Exchange and blend parameters with *other*.

        Standard gossip (``expertise_weighted=False``)
            Both peers take an equal mix (``alpha = gossip_alpha``).

        Expertise-weighted gossip (``expertise_weighted=True``)
            The mixing coefficients are derived from each peer's validation
            loss.  The peer with *lower* loss (more expertise) keeps more
            of its own parameters and donates more to the other:

            ``alpha_self  = exp_self  / (exp_self + exp_other)``
            ``alpha_other = exp_other / (exp_self + exp_other)``

            where ``exp = 1 / (val_loss + ε)``.

            Effect:
            * An expert peer barely changes (its knowledge is preserved).
            * A novice peer moves strongly toward the expert
              (it learns a lot from a single exchange).

        Both peers acquire the appropriate blended state atomically.
        """
        params_self = self.get_parameters()
        params_other = other.get_parameters()

        if (
            expertise_weighted
            and self.val_loss is not None
            and other.val_loss is not None
        ):
            eps = 1e-8
            exp_self = 1.0 / (self.val_loss + eps)
            exp_other = 1.0 / (other.val_loss + eps)
            total = exp_self + exp_other
            alpha_self = exp_self / total    # expert keeps more of itself
            alpha_other = exp_other / total  # novice keeps less of itself
        else:
            alpha_self = self.gossip_alpha
            alpha_other = other.gossip_alpha

        new_self = _blend(params_self, params_other, alpha=alpha_self)
        new_other = _blend(params_other, params_self, alpha=alpha_other)
        self.set_parameters(new_self)
        other.set_parameters(new_other)


# ---------------------------------------------------------------------------
# P2P network
# ---------------------------------------------------------------------------


class P2PNetwork:
    """A P2P network of :class:`Peer` nodes.

    Provides helpers for running gossip rounds and broadcasting updates
    across the entire network.

    Parameters
    ----------
    peers:
        The participating peers (at least two required).
    """

    def __init__(self, peers: List[Peer]) -> None:
        if len(peers) < 2:
            raise ValueError("A P2P network requires at least two peers.")
        self.peers = peers

    @classmethod
    def create(
        cls,
        n_peers: int,
        model_config: FoundationModelConfig,
        gossip_alpha: float = 0.5,
    ) -> "P2PNetwork":
        """Create *n_peers* fresh peers sharing the same model config."""
        peers = [Peer(i, model_config, gossip_alpha) for i in range(n_peers)]
        return cls(peers)

    # ------------------------------------------------------------------
    # Gossip
    # ------------------------------------------------------------------

    def gossip_round(
        self,
        n_pairs: Optional[int] = None,
        expertise_weighted: bool = False,
    ) -> None:
        """Run one round of random pairwise gossip.

        Parameters
        ----------
        n_pairs:
            Number of peer pairs to form.  Defaults to ``len(peers) // 2``.
        expertise_weighted:
            When ``True``, each gossip exchange is weighted by the peers'
            validation losses.  See :meth:`Peer.gossip_with`.
        """
        n_pairs = n_pairs or max(1, len(self.peers) // 2)
        shuffled = list(self.peers)
        random.shuffle(shuffled)
        pairs = list(zip(shuffled[::2], shuffled[1::2]))[:n_pairs]
        for a, b in pairs:
            a.gossip_with(b, expertise_weighted=expertise_weighted)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def broadcast(self, source: Peer) -> None:
        """Push *source*'s parameters to every other peer."""
        state = source.get_parameters()
        for peer in self.peers:
            if peer is not source:
                peer.set_parameters(state)

    def average_parameters(self) -> Dict[str, torch.Tensor]:
        """Compute the arithmetic mean of all peers' parameters."""
        all_states = [p.get_parameters() for p in self.peers]
        return {
            key: torch.stack([s[key].float() for s in all_states]).mean(dim=0)
            for key in all_states[0]
        }
