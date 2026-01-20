import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.profiler import record_function

from cs336_systems.nn import FFN, MultiHeadAttention, RMSNorm
from cs336_systems.nn.utils import RoPEConfig


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        device: torch.device | None = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.ln1 = RMSNorm(d_model, device=device)
        self.ln2 = RMSNorm(d_model, device=device)
        self.attn = MultiHeadAttention(d_model, num_heads, device=device)
        self.ffn = FFN(d_model, d_ff, device=device)

    def forward(
        self,
        x: Float[Tensor, "bs seq_len d_model"],
        rope_config: RoPEConfig | None = None,
    ) -> Float[Tensor, "bs seq_len d_model"]:
        # Use regular addition instead of inplace to avoid MPS gradient issues
        with record_function("block/ln1"):
            normed1 = self.ln1(x)
        with record_function("block/attention"):
            x = x + self.attn(normed1, rope_config=rope_config)
        with record_function("block/ln2"):
            normed2 = self.ln2(x)
        with record_function("block/ffn"):
            x = x + self.ffn(normed2)
        return x
