import torch
from jaxtyping import Float
from torch import Tensor, nn

from cs336_basics.nn import FFN, MultiHeadAttention, RMSNorm
from cs336_basics.nn.utils import RoPEConfig


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
        x = x + self.attn(self.ln1(x), rope_config=rope_config)
        x = x + self.ffn(self.ln2(x))
        return x
