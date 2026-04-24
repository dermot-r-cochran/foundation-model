"""Tests for P2P gossip-based distributed learning."""

import pytest
import torch

from foundation_model import FoundationModel, FoundationModelConfig
from foundation_model.p2p import Peer, P2PNetwork, _blend


@pytest.fixture
def tiny_config() -> FoundationModelConfig:
    return FoundationModelConfig.tiny()


def _batch(config: FoundationModelConfig, B: int = 2, T: int = 8):
    ids = torch.randint(0, config.vocab_size, (B, T))
    labels = torch.randint(0, config.vocab_size, (B, T))
    return ids, labels


# ------------------------------------------------------------------
# _blend helper
# ------------------------------------------------------------------


def test_blend_equal_weights() -> None:
    a = {"w": torch.tensor([0.0, 4.0])}
    b = {"w": torch.tensor([4.0, 0.0])}
    result = _blend(a, b, alpha=0.5)
    assert torch.allclose(result["w"], torch.tensor([2.0, 2.0]))


def test_blend_full_weight_on_a() -> None:
    a = {"w": torch.tensor([1.0])}
    b = {"w": torch.tensor([99.0])}
    result = _blend(a, b, alpha=1.0)
    assert torch.allclose(result["w"], torch.tensor([1.0]))


# ------------------------------------------------------------------
# Peer
# ------------------------------------------------------------------


def test_peer_gossip_equal_alpha(tiny_config: FoundationModelConfig) -> None:
    """With alpha=0.5 both peers should reach the exact midpoint."""
    peer_a = Peer(0, tiny_config, gossip_alpha=0.5)
    peer_b = Peer(1, tiny_config, gossip_alpha=0.5)

    state_a = peer_a.get_parameters()
    state_b = peer_b.get_parameters()

    peer_a.gossip_with(peer_b)

    first_key = next(iter(state_a))
    expected = (state_a[first_key].float() + state_b[first_key].float()) / 2
    assert torch.allclose(peer_a.get_parameters()[first_key].float(), expected, atol=1e-5)
    assert torch.allclose(peer_b.get_parameters()[first_key].float(), expected, atol=1e-5)


def test_peer_gossip_expertise_weighted(tiny_config: FoundationModelConfig) -> None:
    """The expert's params should barely change; the novice's should move toward expert."""
    expert = Peer(0, tiny_config)
    novice = Peer(1, tiny_config)
    expert.val_loss = 0.01   # very good
    novice.val_loss = 10.0   # poor

    params_expert_before = expert.get_parameters()

    # Make them different
    novice.set_parameters(
        {k: v + 5.0 for k, v in expert.get_parameters().items()}
    )

    expert.gossip_with(novice, expertise_weighted=True)

    first_key = next(iter(params_expert_before))
    expert_change = (
        expert.get_parameters()[first_key].float() - params_expert_before[first_key].float()
    ).abs().mean().item()

    # Expert should change very little (alpha_self ≈ 1 - 0.01/10 ≈ 0.999)
    assert expert_change < 0.1


def test_peer_train_step(tiny_config: FoundationModelConfig) -> None:
    peer = Peer(0, tiny_config)
    ids, labels = _batch(tiny_config)
    result = peer.train_step(ids, labels)
    assert "loss" in result
    assert result["loss"] > 0
    assert peer.val_loss is not None


# ------------------------------------------------------------------
# P2PNetwork
# ------------------------------------------------------------------


def test_network_creation(tiny_config: FoundationModelConfig) -> None:
    net = P2PNetwork.create(n_peers=4, model_config=tiny_config)
    assert len(net.peers) == 4


def test_network_requires_at_least_two_peers(tiny_config: FoundationModelConfig) -> None:
    with pytest.raises(ValueError):
        P2PNetwork([Peer(0, tiny_config)])


def test_gossip_round_does_not_raise(tiny_config: FoundationModelConfig) -> None:
    net = P2PNetwork.create(n_peers=4, model_config=tiny_config)
    net.gossip_round(n_pairs=2)


@pytest.mark.parametrize("expertise_weighted", [False, True])
def test_gossip_round_expertise_flag(
    tiny_config: FoundationModelConfig, expertise_weighted: bool
) -> None:
    net = P2PNetwork.create(n_peers=4, model_config=tiny_config)
    ids, labels = _batch(tiny_config)
    for peer in net.peers:
        peer.train_step(ids, labels)
    net.gossip_round(expertise_weighted=expertise_weighted)


def test_broadcast_copies_params(tiny_config: FoundationModelConfig) -> None:
    net = P2PNetwork.create(n_peers=4, model_config=tiny_config)
    source = net.peers[0]
    net.broadcast(source)

    source_params = source.get_parameters()
    first_key = next(iter(source_params))
    for peer in net.peers[1:]:
        assert torch.allclose(
            peer.get_parameters()[first_key].float(),
            source_params[first_key].float(),
        )


def test_average_parameters_shape(tiny_config: FoundationModelConfig) -> None:
    net = P2PNetwork.create(n_peers=3, model_config=tiny_config)
    avg = net.average_parameters()
    expected_keys = set(net.peers[0].get_parameters().keys())
    assert set(avg.keys()) == expected_keys


def test_gossip_convergence(tiny_config: FoundationModelConfig) -> None:
    """After many gossip rounds all peers should converge to the same parameters."""
    net = P2PNetwork.create(n_peers=4, model_config=tiny_config)
    for _ in range(100):
        net.gossip_round()

    states = [p.get_parameters() for p in net.peers]
    first_key = next(iter(states[0]))
    for state in states[1:]:
        assert torch.allclose(
            state[first_key].float(),
            states[0][first_key].float(),
            atol=1e-4,
        )
