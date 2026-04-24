"""Scalable configuration presets for the Foundation Model."""

from dataclasses import dataclass


@dataclass
class FoundationModelConfig:
    """Configuration for the Foundation Model architecture and training."""

    # Architecture
    vocab_size: int = 32000
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 1024
    max_seq_len: int = 512
    dropout_rate: float = 0.1

    # Epistemic uncertainty (MC-Dropout samples)
    mc_dropout_samples: int = 10

    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 0.01

    # Experience replay buffer
    buffer_capacity: int = 10000

    @classmethod
    def tiny(cls) -> "FoundationModelConfig":
        """Minimal config for unit tests and CI."""
        return cls(
            d_model=64,
            n_heads=2,
            n_layers=2,
            d_ff=256,
            max_seq_len=128,
            buffer_capacity=1000,
        )

    @classmethod
    def small(cls) -> "FoundationModelConfig":
        """Suitable for local-device deployment (laptop/edge)."""
        return cls(
            d_model=256,
            n_heads=4,
            n_layers=4,
            d_ff=1024,
            max_seq_len=512,
        )

    @classmethod
    def base(cls) -> "FoundationModelConfig":
        """Suitable for a mid-range cloud VM."""
        return cls(
            d_model=512,
            n_heads=8,
            n_layers=6,
            d_ff=2048,
            max_seq_len=1024,
            buffer_capacity=50000,
        )

    @classmethod
    def large(cls) -> "FoundationModelConfig":
        """Suitable for a high-end cloud VM / multi-GPU node."""
        return cls(
            d_model=1024,
            n_heads=16,
            n_layers=12,
            d_ff=4096,
            max_seq_len=2048,
            buffer_capacity=100000,
        )
