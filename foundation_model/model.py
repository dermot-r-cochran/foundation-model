"""Core transformer model with MC-Dropout for epistemic uncertainty."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FoundationModelConfig


class MCDropout(nn.Dropout):
    """Dropout that stays active at inference time for MC-Dropout uncertainty."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.dropout(x, p=self.p, training=True)


class MultiHeadAttention(nn.Module):
    def __init__(self, config: FoundationModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.d_model = config.d_model

        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.attn_dropout = MCDropout(config.dropout_rate)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        scale = math.sqrt(self.d_head)
        attn = (q @ k.transpose(-2, -1)) / scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, config: FoundationModelConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = MCDropout(config.dropout_rate)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, config: FoundationModelConfig) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(config)
        self.ff = FeedForward(config)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout = MCDropout(config.dropout_rate)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class FoundationModel(nn.Module):
    """
    Epistemic, Experience-Driven Foundation Model.

    Uses MC-Dropout to quantify epistemic (model) uncertainty at inference
    time without any changes to the training procedure.
    """

    def __init__(self, config: FoundationModelConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)
        self.embed_dropout = MCDropout(config.dropout_rate)

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: share embedding and output projection weights
        self.head.weight = self.token_embed.weight

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        return torch.tril(torch.ones(T, T, device=device)).view(1, 1, T, T)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        device = input_ids.device

        positions = torch.arange(T, device=device).unsqueeze(0)
        x = self.embed_dropout(
            self.token_embed(input_ids) + self.pos_embed(positions)
        )

        mask = self._causal_mask(T, device)
        for block in self.blocks:
            x = block(x, mask)

        return self.head(self.norm(x))

    def predict_with_uncertainty(
        self,
        input_ids: torch.Tensor,
        n_samples: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run MC-Dropout inference and return mean logits + epistemic uncertainty.

        Returns:
            mean_logits:           [B, T, V]
            epistemic_uncertainty: [B, T]  (higher = model is less certain)
        """
        n_samples = n_samples or self.config.mc_dropout_samples
        self.train()  # ensure MCDropout is active

        with torch.no_grad():
            logits_list = [self.forward(input_ids) for _ in range(n_samples)]

        logits_stack = torch.stack(logits_list, dim=0)  # [S, B, T, V]
        probs_stack = torch.softmax(logits_stack, dim=-1)

        mean_logits = logits_stack.mean(dim=0)
        epistemic_uncertainty = probs_stack.var(dim=0).mean(dim=-1)  # [B, T]

        return mean_logits, epistemic_uncertainty

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
