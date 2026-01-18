import torch
from einops import rearrange
from jaxtyping import Float, Int
from torch import Tensor, nn


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        weight = torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        nn.init.trunc_normal_(weight, mean=0.0, std=1.0, a=-3, b=3)

        self.weight = nn.Parameter(weight)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def forward(self, token_ids: Int[Tensor, "b s"]) -> Float[Tensor, "b s d"]:
        bs = token_ids.shape[0]
        flattened_token_ids = token_ids.flatten()
        token_embeddings = self.weight[flattened_token_ids, :]

        return rearrange(token_embeddings, "(b s) d -> b s d", b=bs)
