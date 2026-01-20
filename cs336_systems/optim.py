import math
from collections.abc import Callable, Iterable
from typing import cast

import torch
from torch.optim.optimizer import ParamsT


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        defaults = {"lr": lr, "weight_decay": weight_decay, "betas": betas, "eps": eps}
        super().__init__(params, defaults=defaults)

    def step(self, closure: Callable[[], float] | None = None) -> float | None:  # type: ignore
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = cast(float, group["lr"])
            beta1, beta2 = cast(tuple[float, float], group["betas"])
            eps = cast(float, group["eps"])
            weight_decay = cast(float, group["weight_decay"])

            for p in group["params"]:
                if p.grad is None:
                    continue

                m = self.state[p].get("m", 0)
                v = self.state[p].get("v", 0)
                t = self.state[p].get("t", 1)

                grad = p.grad.data
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad**2

                lr_t = lr * (1 - beta2**t) ** 0.5 / (1 - beta1**t)
                p.data -= lr_t * m / ((v**0.5) + eps)
                p.data -= lr * weight_decay * p.data

                self.state[p]["m"] = m
                self.state[p]["v"] = v
                self.state[p]["t"] = t + 1

        return loss


class CosineAnnealingLRScheduler(torch.optim.lr_scheduler.LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        lr_max: float,
        lr_min: float,
        warmup_t: int,
        cosine_cycle_t: int,
        t_0: int = -1,
    ):
        self.t_0 = t_0
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.warmup_t = warmup_t
        self.cosine_cycle_t = cosine_cycle_t
        super().__init__(optimizer)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            raise UserWarning("To get the last learning rate computed by the scheduler, please use `get_last_lr()`.")

        self.t_0 += 1
        lr = lr_cosine_schedule(self.t_0, self.lr_max, self.lr_min, self.warmup_t, self.cosine_cycle_t)
        return [lr for _ in self.optimizer.param_groups]


def lr_cosine_schedule(
    t: int,
    lr_max: float,
    lr_min: float,
    warmup_t: int,
    cosine_cycle_t: int,
) -> float:
    if t < warmup_t:
        return t * lr_max / warmup_t

    if t <= cosine_cycle_t:
        weight = 0.5 * (1 + math.cos(math.pi * (t - warmup_t) / (cosine_cycle_t - warmup_t)))
        return lr_min + weight * (lr_max - lr_min)

    return lr_min


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """
    Clips gradients by their global L2 norm.

    Args:
        parameters: Iterable of parameters with gradients
        max_l2_norm: Maximum L2 norm threshold
    """
    epsilon = 1e-6

    # Convert to list to allow multiple iterations
    params_with_grad = [p for p in parameters if p.grad is not None]

    if len(params_with_grad) == 0:
        return

    # Compute total L2 norm across all gradients
    # Initialize on same device as first gradient to avoid device mismatch
    first_grad = params_with_grad[0].grad
    assert first_grad is not None
    device = first_grad.device

    total_norm_squared = torch.tensor(0.0, device=device, requires_grad=False)
    for p in params_with_grad:
        assert p.grad is not None
        total_norm_squared = total_norm_squared.add_(torch.sum(p.grad.data**2))

    total_norm = total_norm_squared.sqrt_()
    if total_norm < max_l2_norm:
        return

    factor = max_l2_norm / (total_norm + epsilon)
    for p in params_with_grad:
        assert p.grad is not None
        p.grad.data.mul_(factor)
