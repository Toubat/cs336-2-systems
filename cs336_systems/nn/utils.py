from dataclasses import dataclass
from math import sqrt

import torch
from einops import einsum, reduce
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch.profiler import record_function


@dataclass
class RoPEConfig:
    theta: float
    d_k: int
    max_seq_len: int
    token_positions: Int[Tensor, " ... seq_len"] | None = None


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    x = x.swapdims(-1, dim)
    x_max = reduce(x, "... d -> ... 1", reduction="max")

    x = (x - x_max).exp()
    x_sum = reduce(x, "... d -> ... 1", reduction="sum")
    x = x / x_sum

    return x.swapdims(-1, dim)


def scaled_dot_product_attention(
    q: Float[Tensor, "... s k"],
    k: Float[Tensor, "... s k"],
    v: Float[Tensor, "... s v"],
    mask: Bool[Tensor, "s s"] | None = None,
) -> Float[Tensor, "... s v"]:
    d_k = sqrt(q.shape[-1])

    with record_function("sdpa/attention_scores"):
        q_k = einsum(q, k, "... s_in k, ... s_out k -> ... s_in s_out") / d_k

    if mask is not None:
        with record_function("sdpa/mask"):
            q_k = q_k.masked_fill(~mask, value=-torch.inf)

    with record_function("sdpa/softmax"):
        q_k = softmax(q_k, dim=-1)

    with record_function("sdpa/output_matmul"):
        return einsum(q_k, v, "... s_in s_out, ... s_out v -> ... s_in v")
