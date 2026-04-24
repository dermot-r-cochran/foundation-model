# Foundation Model

An **Epistemic, Experience-Driven Foundation Model** suitable for deployment
on a local device or cloud VM, scalable as needed, with built-in support for
peer-to-peer, distributed, and federated learning.

## Overview

The framework provides five interlocking capabilities:

| Capability | Key idea |
|---|---|
| **Epistemic uncertainty** | MC-Dropout: the model knows what it doesn't know |
| **Experience-driven learning** | Priority replay buffer with representative sub-sampling |
| **Representative sub-sampling** | Stratified, diversity (coreset), and uncertainty-driven selection |
| **Federated learning** | FedAvg *and* expertise-weighted aggregation |
| **P2P distributed learning** | Gossip averaging with expertise weighting |
| **Expertise-weighted voting** | Prevents poorly-performing majorities from overriding expert minorities |

---

## Architecture

```
Input tokens ──► Token + Position Embeddings (MC-Dropout)
                 │
                 ▼
           N × Transformer Blocks
           (Causal Self-Attention + Feed-Forward, MC-Dropout)
                 │
                 ▼
           LayerNorm + LM Head
                 │
       ┌─────────┼──────────┐
       ▼         ▼          ▼
  Mean logits  Epistemic  Aleatoric
               uncertainty uncertainty
```

The transformer uses **weight-tied** input and output embeddings. All
`Dropout` modules are replaced by `MCDropout`, which stays active at
inference time so that multiple stochastic forward passes can be used to
estimate uncertainty.

---

## Scale Presets

| Preset | d\_model | Layers | Heads | d\_ff | Typical target |
|--------|---------|--------|-------|------|----------------|
| `tiny`  | 64  | 2  | 2  | 256  | Unit tests / CI |
| `small` | 256 | 4  | 4  | 1024 | Laptop / edge device |
| `base`  | 512 | 6  | 8  | 2048 | Mid-range cloud VM |
| `large` | 1024| 12 | 16 | 4096 | High-end cloud VM |

---

## Features

### 1 · Epistemic Uncertainty (MC-Dropout)

Running *N* stochastic forward passes yields a distribution over predictions.
Total predictive uncertainty is decomposed into:

- **Epistemic** (model uncertainty) = mutual information between weights and
  predictions — high for out-of-distribution inputs.
- **Aleatoric** (data uncertainty) = mean entropy of individual MC
  predictions — irreducible noise inherent in the data.

```python
from foundation_model import FoundationModel, FoundationModelConfig, compute_uncertainty
import torch

config = FoundationModelConfig.small()
model  = FoundationModel(config)

input_ids = torch.randint(0, config.vocab_size, (1, 64))
result = compute_uncertainty(model, input_ids, n_samples=20)

print("Epistemic:", result["epistemic"].mean().item())
print("Aleatoric:", result["aleatoric"].mean().item())
```

### 2 · Experience-Driven Continual Learning

An `ExperienceBuffer` stores past inputs weighted by training loss.
The `Trainer` replays these experiences to prevent catastrophic forgetting.

```python
from foundation_model import FoundationModel, FoundationModelConfig
from foundation_model.trainer import Trainer

model   = FoundationModel(FoundationModelConfig.small())
trainer = Trainer(model, model.config)

for input_ids, labels in my_dataloader:
    trainer.train_step(input_ids, labels)      # auto-stores in buffer
    trainer.replay_step(batch_size=8,
                        strategy="stratified") # replay representative past
```

### 3 · Representative Sub-Sampling

Four samplers prevent large or skewed datasets from dominating training:

| Sampler | Strategy |
|---|---|
| `StratifiedSampler` | Quantile buckets — equal quota from every priority tier |
| `DiversitySampler` | Greedy k-centers — maximise minimum embedding distance |
| `UncertaintySampler` | Top-k epistemic uncertainty — most informative examples |
| `RepresentativeSampler` | Combines diversity + uncertainty in one pass |

```python
from foundation_model import StratifiedSampler, RepresentativeSampler
import torch

# Stratified: equal coverage of easy and hard examples
sampler = StratifiedSampler(n_strata=4)
chosen  = sampler.sample(indices=list(range(10_000)),
                         priorities=loss_per_sample,
                         k=512)

# Combined diversity + uncertainty
rep_sampler = RepresentativeSampler(diversity_fraction=0.5)
chosen = rep_sampler.sample(embeddings=emb,       # [N, D]
                            uncertainties=unc,    # [N]
                            indices=list(range(N)),
                            k=512)
```

Federated clients automatically sub-sample large local datasets before
local training when `max_samples` is passed to `FederatedClient.fit()`.

### 4 · Federated Learning

Two aggregation strategies are available:

| Strategy | Description |
|---|---|
| `"fedavg"` | Classic FedAvg — weight by dataset size |
| `"expertise"` | Weight by `reputation / validation_loss`; optional Byzantine-robust tail trimming |

```python
from foundation_model.federated import FederatedClient, FederatedConfig, FederatedServer

config  = FoundationModelConfig.small()
fed_cfg = FederatedConfig(n_clients=10, strategy="expertise", trim_fraction=0.1)
server  = FederatedServer(config, fed_cfg)
clients = [FederatedClient(i, config) for i in range(10)]

for fl_round in range(num_rounds):
    global_params = server.get_parameters()
    updates, val_losses = [], []
    for client in clients:
        client.set_parameters(global_params)
        # Large dataset? sub-sample to 256 representative rows per round
        params, n = client.fit(local_ids, local_labels, max_samples=256)
        updates.append((params, n))
        val_losses.append(evaluate(client))
    server.aggregate(updates, val_losses=val_losses)
```

### 5 · P2P Gossip Distributed Learning

No central server.  Peers exchange parameters with random neighbours each
round.  With `expertise_weighted=True`, the exchange is biased toward the
better-performing peer — preventing knowledge dilution by many novices.

```python
from foundation_model.p2p import P2PNetwork

net = P2PNetwork.create(n_peers=8, model_config=config)

for step in range(1000):
    peer = net.peers[step % len(net.peers)]
    peer.train_step(local_ids, local_labels)
    net.gossip_round(expertise_weighted=True)
```

### 6 · Expertise-Weighted Voting

Any discrete decision (hyperparameter choice, architecture variant,
model update acceptance) can be put to an expertise-weighted vote.
A single expert contributor can legitimately out-weigh a majority of
poorly-performing participants.

```python
from foundation_model.voting import DiscreteVote, ReputationTracker

tracker = ReputationTracker()
# Pre-register contributors with their known validation losses
for cid, loss in zip(contributor_ids, val_losses):
    s = tracker.register(cid)
    s.validation_loss = loss

vote = DiscreteVote(tracker)
winner, scores = vote.tally([
    (expert_id, "option_a"),
    (novice_1,  "option_b"),
    (novice_2,  "option_b"),
])
# winner == "option_a" even though outvoted 2:1
```

---

## Installation

```bash
# CPU (local device)
pip install -e .

# Development (includes pytest)
pip install -e ".[dev]"
```

---

## Deployment

### Local device (CPU)

```bash
docker build --target cpu -t foundation-model:cpu .
docker run foundation-model:cpu
```

### Cloud VM (GPU)

```bash
docker build --target gpu \
  --build-arg TORCH_EXTRA="--index-url https://download.pytorch.org/whl/cu121" \
  -t foundation-model:gpu .
docker run --gpus all foundation-model:gpu
```

### Docker Compose

```bash
# Single-node demo
docker compose up foundation-model-cpu

# Federated learning demo
docker compose up foundation-model-federated

# P2P gossip demo
docker compose up foundation-model-p2p

# GPU / cloud VM
docker compose up foundation-model-gpu
```

---

## Testing

```bash
pip install -e ".[dev]"
pytest
```

---

## License

Eclipse Public License v2.0 — see [LICENSE](LICENSE).