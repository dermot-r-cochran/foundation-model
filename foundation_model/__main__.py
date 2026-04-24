"""Demo entry-point.

Set the DEMO_MODE environment variable to control which demo runs:
  single      – single-node epistemic inference
  federated   – FedAvg and expertise-weighted federated learning
  p2p         – gossip-based P2P distributed learning
  all         – run all demos (default)
"""

from __future__ import annotations

import os

import torch

from .config import FoundationModelConfig
from .federated import FederatedClient, FederatedConfig, FederatedServer
from .p2p import P2PNetwork
from .uncertainty import compute_uncertainty


def _demo_single(device: torch.device) -> None:
    from .model import FoundationModel

    print("\n── Single-Node Epistemic Inference ────────────────────")
    config = FoundationModelConfig.small()
    model = FoundationModel(config).to(device)
    print(f"  Parameters : {model.n_parameters:,}")

    input_ids = torch.randint(0, config.vocab_size, (1, 32), device=device)
    result = compute_uncertainty(model, input_ids, n_samples=10)
    print(f"  Epistemic  : {result['epistemic'].mean().item():.6f}")
    print(f"  Aleatoric  : {result['aleatoric'].mean().item():.6f}")


def _demo_federated() -> None:
    print("\n── Federated Learning Demo (FedAvg + Expertise) ───────")
    config = FoundationModelConfig.tiny()

    for strategy in ("fedavg", "expertise"):
        fed_cfg = FederatedConfig(n_clients=3, local_epochs=1, strategy=strategy)
        server = FederatedServer(config, fed_cfg)
        clients = [FederatedClient(i, config) for i in range(fed_cfg.n_clients)]

        B, T = 4, 16
        for _ in range(2):
            global_params = server.get_parameters()
            updates, val_losses = [], []
            for client in clients:
                client.set_parameters(global_params)
                ids = torch.randint(0, config.vocab_size, (B * 4, T))
                lbs = torch.randint(0, config.vocab_size, (B * 4, T))
                # Sub-sample to B representative rows before local training
                params, n = client.fit(ids, lbs, max_samples=B)
                updates.append((params, n))
                val_losses.append(0.5 + 0.1 * client.client_id)
            server.aggregate(updates, val_losses=val_losses)

        print(f"  [{strategy:>9s}] round {server.round} complete "
              f"({fed_cfg.n_clients} clients, sub-sampled to {B} rows each)")


def _demo_p2p() -> None:
    print("\n── P2P Gossip Demo (uniform + expertise-weighted) ─────")
    config = FoundationModelConfig.tiny()

    for expertise_weighted in (False, True):
        net = P2PNetwork.create(n_peers=4, model_config=config)
        B, T = 2, 8
        for peer in net.peers:
            ids = torch.randint(0, config.vocab_size, (B, T))
            lbs = torch.randint(0, config.vocab_size, (B, T))
            peer.train_step(ids, lbs)

        for _ in range(10):
            net.gossip_round(expertise_weighted=expertise_weighted)

        label = "expertise-weighted" if expertise_weighted else "uniform"
        print(f"  [{label}] 10 gossip rounds over {len(net.peers)} peers – done")


def main() -> None:
    device_name = os.environ.get("DEVICE", "cpu")
    device = torch.device(
        device_name
        if torch.cuda.is_available() and device_name == "cuda"
        else "cpu"
    )
    mode = os.environ.get("DEMO_MODE", "all")
    print(f"Foundation Model  |  device={device}  |  mode={mode}")

    if mode in ("single", "all"):
        _demo_single(device)
    if mode in ("federated", "all"):
        _demo_federated()
    if mode in ("p2p", "all"):
        _demo_p2p()

    print("\n✓ Done")


if __name__ == "__main__":
    main()
