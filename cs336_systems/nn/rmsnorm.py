import torch
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
        # in_type = x.dtype

        # upscale to float32 for numerical stability
        # x = x.to(torch.float32)

        # Use torch operations instead of einsum
        rms = (x * x).sum(dim=-1, keepdim=True)
        rms = rms / self.d_model + self.eps
        rms = rms.rsqrt()  # rsqrt is faster than sqrt + divide

        x = x * rms * self.gamma
        # return x.to(in_type)
        return x
