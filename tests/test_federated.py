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


# ------------------------------------------------------------------
# FederatedServer – what the global model becomes (the README's federated row)
# ------------------------------------------------------------------


def _shifted(base: dict[str, torch.Tensor], delta: float) -> dict[str, torch.Tensor]:
    return {k: v.float() + delta for k, v in base.items()}


def test_server_weighs_by_dataset_size_under_fedavg_and_by_validation_loss_under_expertise(
    tiny_config: FoundationModelConfig,
) -> None:
    """The same two updates move the global model toward different clients.

    This is the README's federated row: FedAvg weights by dataset size, the
    expertise strategy by validation loss, "regardless of dataset size" as
    FederatedServer's docstring puts it. Two synthetic updates, each a
    constant shift of every parameter of a fresh server's global model, are
    handed to a server once per strategy, with the same validation losses in
    both cases. A large, poor client (1000 samples, val_loss 5.0) sends
    base + 1.0; a small, expert client (10 samples, val_loss 0.1) sends
    base - 1.0.

    Under FedAvg the weights are the sample counts, so the global model moves
    by (1000 - 10) / 1010 = +0.9802 on every parameter; FedAvg ignores the
    validation losses it is handed. Under expertise weighting the weights are
    reputation / (val_loss + 1e-8) with a fresh reputation of 1.0, that is 0.2
    and 10.0, so the global model moves by (0.2 - 10.0) / 10.2 = -0.9608.

    Tolerance: float32 weighted sums of values of order 1 lose about 1e-7, the
    1e-8 eps moves the expected shift by less than 1e-8, and an absolute
    tolerance of 1e-4 is still some ten thousand times smaller than the 1.94
    between the two strategies' results.

    It turns red if the server stops loading the aggregate into the global
    model, FedAvg stops weighting by dataset size, the expertise strategy
    stops using the validation losses, or either strategy runs the other. Its
    limit: it is one round from fresh reputations (1.0 each), so it says
    nothing about reputation carried across rounds.
    """
    n_big, n_expert = 1000, 10
    loss_big, loss_expert = 5.0, 0.1
    expected_shift = {
        "fedavg": (n_big - n_expert) / (n_big + n_expert),
        "expertise": (1.0 / loss_big - 1.0 / loss_expert)
        / (1.0 / loss_big + 1.0 / loss_expert),
    }

    for strategy, shift in expected_shift.items():
        server = FederatedServer(tiny_config, FederatedConfig(strategy=strategy))
        base = server.get_parameters()
        server.aggregate(
            [(_shifted(base, 1.0), n_big), (_shifted(base, -1.0), n_expert)],
            val_losses=[loss_big, loss_expert],
        )
        after = server.get_parameters()
        for key, value in base.items():
            assert torch.allclose(
                after[key], value.float() + shift, rtol=0.0, atol=1e-4
            ), f"{strategy}: {key} did not move by {shift:.4f}"


def test_server_trim_fraction_reaches_the_aggregator(
    tiny_config: FoundationModelConfig,
) -> None:
    """FederatedConfig.trim_fraction, the knob the README's federated example
    sets, reaches the aggregator.

    The trim itself, by weight and not by parameter value, is tested in
    test_voting.py; this test proves only the wiring through the server.
    aggregate() takes no reputations, only the validation losses, so the
    high-weight tail here is a claimed loss too good to be true. Five updates
    of 10 samples each go to an expertise server:
    three honest (base + 0.1, val_loss 1.0, weight 1.0 each), one claiming an
    implausibly good loss (base + 100.0, val_loss 0.001, weight 1000) and one
    with a very poor loss (base - 100.0, val_loss 50.0, weight 0.02).

    With trim_fraction=0.2 of five contributors, one is dropped from each end
    of the weight order, so the global model is base + 0.1 on every key.
    Tolerance: three equal weights of 1/3 summed in float32 round at about
    1e-7, so 1e-5 is over a million times smaller than the poisoned shifts. With
    trim_fraction=0.0 the low-loss claimant dominates: the mean shift on the
    first key is (0.3 + 100000 - 2) / 1003.02, about 99.7, and the test asserts
    only that it exceeds 90.0, as the untrimmed case does in test_voting.py.

    It turns red if the server stops passing trim_fraction to its aggregator.
    """
    honest = 0.1

    def run(trim_fraction: float) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        server = FederatedServer(
            tiny_config,
            FederatedConfig(strategy="expertise", trim_fraction=trim_fraction),
        )
        base = server.get_parameters()
        server.aggregate(
            [
                (_shifted(base, honest), 10),
                (_shifted(base, honest), 10),
                (_shifted(base, honest), 10),
                (_shifted(base, 100.0), 10),  # claims a loss too good to be true
                (_shifted(base, -100.0), 10),  # very poor loss
            ],
            val_losses=[1.0, 1.0, 1.0, 0.001, 50.0],
        )
        return base, server.get_parameters()

    base, trimmed = run(0.2)
    for key, value in base.items():
        assert torch.allclose(trimmed[key], value.float() + honest, rtol=0.0, atol=1e-5)

    base, untrimmed = run(0.0)
    first_key = next(iter(base))
    shift = (untrimmed[first_key] - base[first_key].float()).mean().item()
    assert shift > 90.0  # weight 1000 against 3 + 0.02: the claimed loss wins
