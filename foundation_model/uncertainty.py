"""Epistemic and aleatoric uncertainty estimation via MC-Dropout."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from .model import FoundationModel


def entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Shannon entropy of a probability distribution over the last dimension.

    Parameters
    ----------
    probs:
        ``[..., V]`` probability tensor (must sum to 1 over the last dim).
    eps:
        Small constant for numerical stability.

    Returns
    -------
    torch.Tensor
        Entropy values with the last dimension removed.
    """
    return -(probs * (probs + eps).log()).sum(dim=-1)


def predictive_entropy(
    logits_samples: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Total predictive entropy averaged over MC-Dropout samples.

    Parameters
    ----------
    logits_samples:
        ``[S, B, T, V]`` – logits from *S* stochastic forward passes.

    Returns
    -------
    torch.Tensor  ``[B, T]``
    """
    probs = F.softmax(logits_samples, dim=-1)   # [S, B, T, V]
    mean_probs = probs.mean(dim=0)              # [B, T, V]
    return entropy(mean_probs, eps)


def aleatoric_uncertainty(
    logits_samples: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Aleatoric (data) uncertainty: mean entropy of individual predictions.

    Parameters
    ----------
    logits_samples:
        ``[S, B, T, V]``

    Returns
    -------
    torch.Tensor  ``[B, T]``
    """
    probs = F.softmax(logits_samples, dim=-1)   # [S, B, T, V]
    per_sample_h = entropy(probs, eps)           # [S, B, T]
    return per_sample_h.mean(dim=0)              # [B, T]


def epistemic_uncertainty(
    logits_samples: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Epistemic (model) uncertainty: mutual information between weights and predictions.

    Computed as: predictive entropy − aleatoric uncertainty.

    Parameters
    ----------
    logits_samples:
        ``[S, B, T, V]``

    Returns
    -------
    torch.Tensor  ``[B, T]``  (non-negative)
    """
    return (predictive_entropy(logits_samples, eps) - aleatoric_uncertainty(logits_samples, eps)).clamp(min=0.0)


def compute_uncertainty(
    model: "FoundationModel",
    input_ids: torch.Tensor,
    n_samples: int = 20,
) -> Dict[str, torch.Tensor]:
    """Run a full uncertainty decomposition on *input_ids*.

    Activates MC-Dropout by putting the model in training mode, then runs
    *n_samples* stochastic forward passes and decomposes total predictive
    uncertainty into epistemic and aleatoric components.

    Parameters
    ----------
    model:
        A :class:`~foundation_model.model.FoundationModel` instance.
    input_ids:
        ``[B, T]`` integer token tensor.
    n_samples:
        Number of MC-Dropout forward passes (more = more accurate estimates).

    Returns
    -------
    dict with keys:
        ``"mean_logits"``        – ``[B, T, V]``
        ``"predictive_entropy"`` – ``[B, T]``
        ``"aleatoric"``          – ``[B, T]``
        ``"epistemic"``          – ``[B, T]``
    """
    model.train()  # activate MCDropout
    logits_list = []
    with torch.no_grad():
        for _ in range(n_samples):
            logits_list.append(model(input_ids))

    logits_stack = torch.stack(logits_list, dim=0)  # [S, B, T, V]

    return {
        "mean_logits": logits_stack.mean(dim=0),
        "predictive_entropy": predictive_entropy(logits_stack),
        "aleatoric": aleatoric_uncertainty(logits_stack),
        "epistemic": epistemic_uncertainty(logits_stack),
    }
