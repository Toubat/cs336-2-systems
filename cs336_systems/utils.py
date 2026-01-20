import os
from pathlib import Path
from typing import IO, BinaryIO

import numpy as np
import numpy.typing as npt
import torch
from loguru import logger


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    N, M = dataset.shape[0], context_length

    starts = np.random.randint(0, N - M, batch_size)
    indices = np.zeros((batch_size, M + 1), dtype=np.int32)

    for row in range(batch_size):
        indices[row, :] = np.arange(starts[row], starts[row] + M + 1)

    # Convert uint16 to int64 (torch.long) since PyTorch doesn't support uint16
    batch = torch.from_numpy(dataset[indices].astype(np.int64))

    x, y = batch[:, :-1], batch[:, 1:]

    if "cuda" in device:
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)

    return x, y


def get_random_batch(vocab_size: int, batch_size: int, context_length: int, device: str):
    """
    Get a random batch of data (no dataset is required, just shape)
    """
    x = torch.randint(0, vocab_size, (batch_size, context_length))

    if "cuda" in device:
        x = x.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
    return x


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    return torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
        },
        out,
    )


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]


def get_latest_checkpoint_path(checkpoint_dir: os.PathLike) -> Path | None:
    os.makedirs(checkpoint_dir, exist_ok=True)

    ckpt_files = list(Path(checkpoint_dir).glob("*.pt"))
    if not ckpt_files:
        return None

    numeric_ckpts = []
    for f in ckpt_files:
        try:
            numeric_ckpts.append((int(f.stem), f))
        except ValueError:
            logger.warning(f"Ignoring non-numeric checkpoint file: {f.name}")
            continue

    if not numeric_ckpts:
        logger.warning("No valid numeric checkpoint files found")
        return None

    ckpt_file = max(numeric_ckpts, key=lambda x: x[0])[1]
    logger.info(f"Loading checkpoint from {ckpt_file.name}")

    return ckpt_file
