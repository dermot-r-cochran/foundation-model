"""foundation_model – public API."""

from .config import FoundationModelConfig
from .experience import ExperienceBuffer
from .federated import (
    FederatedClient,
    FederatedConfig,
    FederatedServer,
    fedavg_aggregate,
)
from .model import FoundationModel
from .p2p import P2PNetwork, Peer
from .sampling import (
    DiversitySampler,
    RepresentativeSampler,
    StratifiedSampler,
    UncertaintySampler,
)
from .trainer import Trainer
from .uncertainty import compute_uncertainty
from .voting import (
    ContributorStats,
    DiscreteVote,
    ExpertiseWeightedAggregator,
    ReputationTracker,
    expertise_weight,
)

__all__ = [
    # Core model
    "FoundationModelConfig",
    "FoundationModel",
    # Experience
    "ExperienceBuffer",
    # Sampling
    "StratifiedSampler",
    "DiversitySampler",
    "UncertaintySampler",
    "RepresentativeSampler",
    # Uncertainty
    "compute_uncertainty",
    # Training
    "Trainer",
    # Federated learning
    "FederatedConfig",
    "FederatedClient",
    "FederatedServer",
    "fedavg_aggregate",
    # P2P
    "Peer",
    "P2PNetwork",
    # Voting
    "ContributorStats",
    "expertise_weight",
    "ReputationTracker",
    "ExpertiseWeightedAggregator",
    "DiscreteVote",
]
