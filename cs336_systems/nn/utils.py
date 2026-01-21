from dataclasses import dataclass
from math import sqrt

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch.profiler import record_function


@dataclass
class RoPEConfig:
    theta: float
    d_k: int
    max_seq_len: int
    token_positions: Int[Tensor, " ... seq_len"] | None = None


def scaled_dot_product_attention(
    q: Float[Tensor, "... s k"],
    k: Float[Tensor, "... s k"],
    v: Float[Tensor, "... s v"],
    mask: Bool[Tensor, "s s"] | None = None,
) -> Float[Tensor, "... s v"]:
    d_k = sqrt(q.shape[-1])

    with record_function("attn/attention_scores"):
        # Use torch.matmul for optimized cuBLAS kernels
        q_k = torch.matmul(q, k.transpose(-2, -1)) / d_k

    if mask is not None:
        q_k = q_k.masked_fill(~mask, value=-torch.inf)

    with record_function("attn/softmax"):
        # Use F.softmax for optimized kernel
        q_k = F.softmax(q_k, dim=-1)

    with record_function("attn/output_matmul"):
        # Use torch.matmul for optimized cuBLAS kernels
        return torch.matmul(q_k, v)
