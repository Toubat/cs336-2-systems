import torch
from einops import einsum, repeat
from jaxtyping import Float
from torch import Tensor, nn


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Float[Tensor, " ... d"]) -> Float[Tensor, " ... d"]:
        in_type = x.dtype

        rms = einsum(x, x, "... d, ... d -> ...")
        rms = repeat(rms, "... -> ... 1")

        rms = rms / self.d_model + self.eps
        rms = rms.sqrt()

        # upscale to float32 for numerical stability
        x = x.to(torch.float32)
        x = einsum(x / rms, self.gamma, "... d, d -> ... d")
        return x.to(in_type)
