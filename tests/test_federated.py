"""Tests for federated learning (FedAvg and expertise-weighted)."""

import pytest
import torch

from foundation_model import FoundationModel, FoundationModelConfig
from foundation_model.federated import (
    FederatedClient,
    FederatedConfig,
    FederatedServer,
    fedavg_aggregate,
)


@pytest.fixture
def tiny_config() -> FoundationModelConfig:
    return FoundationModelConfig.tiny()


def _batch(config: FoundationModelConfig, B: int = 4, T: int = 8):
    ids = torch.randint(0, config.vocab_size, (B, T))
    labels = torch.randint(0, config.vocab_size, (B, T))
    return ids, labels


# ------------------------------------------------------------------
# fedavg_aggregate
# ------------------------------------------------------------------


def test_fedavg_equal_weights() -> None:
    state_a = {"w": torch.tensor([0.0, 2.0])}
    state_b = {"w": torch.tensor([4.0, 6.0])}
    result = fedavg_aggregate([state_a, state_b], weights=[1.0, 1.0])
    assert torch.allclose(result["w"], torch.tensor([2.0, 4.0]))


def test_fedavg_unequal_weights() -> None:
    state_a = {"w": torch.tensor([0.0])}
    state_b = {"w": torch.tensor([10.0])}
    result = fedavg_aggregate([state_a, state_b], weights=[3.0, 1.0])
    assert torch.allclose(result["w"], torch.tensor([2.5]))


# ------------------------------------------------------------------
# FederatedClient
# ------------------------------------------------------------------


def test_client_fit_returns_params_and_count(tiny_config: FoundationModelConfig) -> None:
    client = FederatedClient(0, tiny_config)
    ids, labels = _batch(tiny_config, B=4)
    params, n = client.fit(ids, labels)
    assert n == 4
    assert set(params.keys()) == set(client.model.state_dict().keys())


def test_client_fit_with_max_samples_subsamples(tiny_config: FoundationModelConfig) -> None:
    client = FederatedClient(0, tiny_config)
    ids, labels = _batch(tiny_config, B=20)
    params, n = client.fit(ids, labels, max_samples=5)
    assert n == 5


def test_client_fit_no_subsampling_when_within_limit(tiny_config: FoundationModelConfig) -> None:
    client = FederatedClient(0, tiny_config)
    ids, labels = _batch(tiny_config, B=4)
    params, n = client.fit(ids, labels, max_samples=10)
    assert n == 4  # full dataset used


def test_client_set_get_parameters(tiny_config: FoundationModelConfig) -> None:
    client_a = FederatedClient(0, tiny_config)
    client_b = FederatedClient(1, tiny_config)
    params_a = client_a.get_parameters()
    client_b.set_parameters(params_a)
    first_key = next(iter(params_a))
    assert torch.allclose(
        client_b.get_parameters()[first_key],
        params_a[first_key],
    )


# ------------------------------------------------------------------
# FederatedServer – FedAvg strategy
# ------------------------------------------------------------------


def test_server_fedavg_round(tiny_config: FoundationModelConfig) -> None:
    fed_cfg = FederatedConfig(n_clients=3, strategy="fedavg")
    server = FederatedServer(tiny_config, fed_cfg)
    clients = [FederatedClient(i, tiny_config) for i in range(3)]

    global_params = server.get_parameters()
    updates = []
    for client in clients:
        client.set_parameters(global_params)
        ids, labels = _batch(tiny_config)
        params, n = client.fit(ids, labels)
        updates.append((params, n))

    server.aggregate(updates)
    assert server.round == 1


def test_server_multiple_rounds(tiny_config: FoundationModelConfig) -> None:
    server = FederatedServer(tiny_config)
    clients = [FederatedClient(i, tiny_config) for i in range(2)]

    for _ in range(3):
        params = server.get_parameters()
        updates = []
        for c in clients:
            c.set_parameters(params)
            ids, labels = _batch(tiny_config)
            updates.append(c.fit(ids, labels))
        server.aggregate(updates)

    assert server.round == 3


# ------------------------------------------------------------------
# FederatedServer – expertise strategy
# ------------------------------------------------------------------


def test_server_expertise_round(tiny_config: FoundationModelConfig) -> None:
    fed_cfg = FederatedConfig(n_clients=3, strategy="expertise")
    server = FederatedServer(tiny_config, fed_cfg)
    clients = [FederatedClient(i, tiny_config) for i in range(3)]

    global_params = server.get_parameters()
    updates, val_losses = [], []
    for i, client in enumerate(clients):
        client.set_parameters(global_params)
        ids, labels = _batch(tiny_config)
        params, n = client.fit(ids, labels)
        updates.append((params, n))
        val_losses.append(0.5 + i * 0.2)  # different expertise levels

    server.aggregate(updates, val_losses=val_losses)
    assert server.round == 1


def test_server_empty_update_is_noop(tiny_config: FoundationModelConfig) -> None:
    server = FederatedServer(tiny_config)
    params_before = {k: v.clone() for k, v in server.global_model.state_dict().items()}
    server.aggregate([])
    params_after = server.get_parameters()
    first_key = next(iter(params_before))
    assert torch.allclose(params_before[first_key], params_after[first_key])
    assert server.round == 0
