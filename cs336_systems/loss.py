from einops import reduce, repeat
from jaxtyping import Float, Int
from torch import Tensor, nn


class CrossEntropyLoss(nn.Module):
    def forward(
        self,
        logits: Float[Tensor, "... vocab_size"],
        targets: Int[Tensor, "..."],
    ) -> Tensor:
        logits_max = reduce(logits, "... c -> ... 1", reduction="max")

        exp = (logits - logits_max).exp()
        sum_exp = reduce(exp, "... c -> ... 1", reduction="sum")
        log_sum_exp = sum_exp.log()  # (bs, seq_len, 1)

        targets = repeat(targets, "... -> ... 1")
        target_logits = logits.gather(dim=-1, index=targets)  # (bs, seq_len, 1)
        loss = log_sum_exp + logits_max - target_logits  # (bs, seq_len, 1)
        loss = loss.mean()

        return loss
