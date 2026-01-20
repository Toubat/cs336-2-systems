import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from cs336_systems.nn.linear import Linear
from cs336_systems.nn.rope import apply_rope
from cs336_systems.nn.utils import RoPEConfig, scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: torch.device | None = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.qkv_proj = Linear(d_model, d_model * 3, device=device)
        self.o_proj = Linear(d_model, d_model, device=device)

    def forward(
        self,
        x: Float[Tensor, "bs seq_len d_model"],
        rope_config: RoPEConfig | None = None,
    ) -> Float[Tensor, "bs seq_len d_model"]:
        qkv = self.qkv_proj(x)  # (bs seq_len d_model * 3)
        q, k, v = rearrange(
            qkv,
            "bs seq_len (three num_heads d_k) -> three bs num_heads seq_len d_k",
            num_heads=self.num_heads,
            d_k=self.d_k,
        )  # (bs, num_heads, seq_len, d_k)

        if rope_config is not None:
            theta = rope_config.theta
            d_k = rope_config.d_k
            max_seq_len = rope_config.max_seq_len
            token_positions = rope_config.token_positions
            q = apply_rope(d_k, theta, max_seq_len, q, token_positions, device=x.device)
            k = apply_rope(d_k, theta, max_seq_len, k, token_positions, device=x.device)

        seq_len = q.shape[-2]
        mask = torch.tril(torch.ones((seq_len, seq_len))).bool().to(x.device)  # (seq_len, seq_len)
        o = scaled_dot_product_attention(q, k, v, mask)  # (bs, num_heads, seq_len, d_k)
        o = rearrange(o, "bs num_heads seq_len d_k -> bs seq_len (num_heads d_k)")

        return self.o_proj(o)
