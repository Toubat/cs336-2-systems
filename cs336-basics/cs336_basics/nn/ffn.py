import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import functional as F


class FFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        from cs336_basics.nn import Linear

        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype

        # d_ff = int(8 * d_model / 3)
        # if d_ff % 64 != 0:
        #     d_ff += 64 - d_ff % 64  # round up to nearest multiple of 64

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        return self.w2(silu(self.w1(x)) * self.w3(x))


def silu(x: Tensor) -> Tensor:
    return x * F.sigmoid(x)
