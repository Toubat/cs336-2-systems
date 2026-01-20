from __future__ import annotations

from typing import cast

import torch
from einops import einsum, repeat
from jaxtyping import Float, Int
from torch import Tensor, nn

__rope_modules: dict[tuple[int, float, int], RotaryPositionalEmbedding] = {}


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        Construct the RoPE module and create buffers if needed.

        Args:
            theta (float): Θ value for the RoPE
            d_k (int): dimension of query and key vectors
            max_seq_len (int): maximum sequence length
            device (torch.device | None): device to store the buffer on
        """
        super().__init__()

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        pos = torch.arange(max_seq_len, dtype=torch.float32, device=device)  # (max_seq_len)
        pair_idx = torch.arange(0, d_k, 2, dtype=torch.float32, device=device)  # (d_k // 2)
        inv_freq = 1.0 / (theta ** (pair_idx / d_k))  # (d_k // 2)

        theta_ik = einsum(pos, inv_freq, "i, k -> i k")  # (max_seq_len, d_k // 2)
        self.register_buffer("sin_table", torch.sin(theta_ik), persistent=False)
        self.register_buffer("cos_table", torch.cos(theta_ik), persistent=False)

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_k"],
        token_positions: Int[Tensor, "... seq_len"] | None = None,
    ) -> Float[Tensor, "... seq_len d_k"]:
        """
        Apply RoPE to the input tensor.

        Args:
            x (torch.Tensor): input tensor of shape (..., seq_len, d_k)
            token_positions (torch.Tensor): positions of shape (..., seq_len)

        Returns:
            torch.Tensor: output tensor of the same shape as x
        """
        sin_table = cast(Tensor, self.sin_table)
        cos_table = cast(Tensor, self.cos_table)

        if token_positions is None:
            token_positions = torch.arange(x.shape[-2], device=x.device, dtype=torch.long)
            token_positions = repeat(token_positions, "... -> 1 ...")

        rope_sin = sin_table[token_positions]  # (..., seq_len, d_k // 2)
        rope_cos = cos_table[token_positions]  # (..., seq_len, d_k // 2)

        x_even, x_odd = x[..., 0::2], x[..., 1::2]
        x_even_rot = x_even * rope_cos - x_odd * rope_sin
        x_odd_rot = x_even * rope_sin + x_odd * rope_cos

        x_rot = torch.zeros_like(x)
        x_rot[..., 0::2] = x_even_rot
        x_rot[..., 1::2] = x_odd_rot
        return x_rot


def apply_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    x: Float[Tensor, "... seq_len d_k"],
    token_positions: Int[Tensor, "... seq_len"] | None = None,
    device: torch.device | None = None,
) -> Float[Tensor, "... seq_len d_k"]:
    key = (d_k, theta, max_seq_len)

    if key not in __rope_modules:
        __rope_modules[key] = RotaryPositionalEmbedding(theta=theta, d_k=d_k, max_seq_len=max_seq_len, device=device)

    module = __rope_modules[key]
    return module(x, token_positions)
