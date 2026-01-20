import shutil
from pathlib import Path
from typing import Literal, cast

import chz
import numpy as np
import numpy.typing as npt
import torch
import wandb
from dotenv import find_dotenv, load_dotenv
from loguru import logger
from tqdm import tqdm

from cs336_basics.bpe.tokenizer import Tokenizer
from cs336_basics.config import (
    DATASET_TRAIN_PATHS,
    DATASET_VALID_PATHS,
    TOKENIZER_MERGES_PATHS,
    TOKENIZER_VOCAB_PATHS,
    TrainingConfig,
)
from cs336_basics.loss import CrossEntropyLoss
from cs336_basics.optim import AdamW, CosineAnnealingLRScheduler, gradient_clipping
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.utils import get_batch, load_checkpoint, save_checkpoint

load_dotenv(find_dotenv())


def get_dataset(config: TrainingConfig, mode: Literal["train", "valid"]) -> npt.NDArray:
    rel_path = DATASET_TRAIN_PATHS[config.dataset] if mode == "train" else DATASET_VALID_PATHS[config.dataset]

    if config.remote:
        dest_path = Path(f"./{config.name}_{mode}_data.npy")
        logger.info(f"Run on remote, copying dataset to local: {config.volume_path / rel_path} -> {dest_path}")
        shutil.copy(config.volume_path / rel_path, dest_path, follow_symlinks=False)
        path = dest_path
    else:
        path = config.volume_path / rel_path

    logger.info(f"Loading {mode} dataset from {path}")
    return np.lib.format.open_memmap(path, mode="r", dtype=np.uint16)


def get_tokenizer(config: TrainingConfig) -> Tokenizer:
    return Tokenizer.from_file(
        vocab_path=config.volume_path / TOKENIZER_VOCAB_PATHS[config.dataset],
        merges_path=config.volume_path / TOKENIZER_MERGES_PATHS[config.dataset],
        special_tokens=["<|endoftext|>"],
    )


def train(config: TrainingConfig):
    device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Clear MPS cache from previous runs
    if device == "mps":
        torch.mps.empty_cache()
        logger.info("Cleared MPS cache")
    elif "cuda" in device:
        torch.set_float32_matmul_precision("high")

    model = TransformerLM(
        vocab_size=config.model.vocab_size,
        context_length=config.model.context_length,
        num_layers=config.model.num_layers,
        d_model=config.model.d_model,
        num_heads=config.model.num_heads,
        d_ff=config.model.d_ff,
        theta=config.model.theta,
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), weight_decay=0.001)  # Reduced from default 0.01

    logger.info("Compiling model...")
    if device == "mps":
        model = cast(TransformerLM, torch.compile(model, backend="aot_eager"))
    else:
        model = cast(TransformerLM, torch.compile(model))

    if config.checkpoint_data_path is not None:
        logger.info(f"Loading checkpoint from {config.checkpoint_data_path.name}")
        t0 = load_checkpoint(config.checkpoint_data_path, model, optimizer)
    else:
        t0 = -1
        logger.info("No checkpoint found, starting from scratch")

    lr_scheduler = CosineAnnealingLRScheduler(
        optimizer,
        t_0=t0,
        lr_max=config.lr_max,
        lr_min=config.lr_min,
        warmup_t=config.warmup_t,
        cosine_cycle_t=config.cosine_cycle_t,
    )

    train, valid = get_dataset(config, "train"), get_dataset(config, "valid")

    logger.info(f"Total model size: {sum(p.numel() for p in model.parameters()) / 1024**2:.2f} MB")
    logger.info(f"Initial learning rate: {lr_scheduler.get_last_lr()}")
    logger.info(f"Train dataset size: {train.shape[0]}")
    logger.info(f"Valid dataset size: {valid.shape[0]}")

    criterion = CrossEntropyLoss()
    with wandb.init(
        entity="yoasobyin-n-a",
        project="cs336",
        id=config.wandb_id,
        name=config.name,
        config=config.wandb_config,
    ) as run:
        run.watch(model, log="all", log_freq=100)

        model.train()
        for step in tqdm(range(config.epochs), desc="Steps", total=config.epochs):
            if step <= t0 + 1:
                continue

            optimizer.zero_grad()

            batch_X, batch_y = get_batch(train, config.batch_size, config.model.context_length, device)

            logits: torch.Tensor = model(batch_X)  # (bs, seq_len, vocab_size)
            loss: torch.Tensor = criterion(logits, batch_y)
            train_accuracy = compute_accuracy(logits, batch_y)
            run.log(
                {"loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0], "train_accuracy (%)": train_accuracy},
                step=step,
            )

            loss.backward()

            # Clip RMSNorm gamma gradients separately to prevent collapse
            for name, param in model.named_parameters():
                if "gamma" in name and param.grad is not None:
                    param.grad.data.clamp_(-1.0, 1.0)

            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
            if step % 10 == 0:
                run.log({"gradient_norm": total_norm.item()}, step=step)

            gradient_clipping(model.parameters(), config.gradient_clipping_norm)

            optimizer.step()
            lr_scheduler.step()

            # Clear MPS cache periodically to avoid fragmentation
            if device == "mps" and step % 200 == 0:
                torch.mps.empty_cache()

            if step % config.valid_interval == 0:
                valid_loss, valid_accuracy = run_evaluation(model, valid, config, device)
                run.log({"valid_loss": valid_loss, "valid_accuracy (%)": valid_accuracy}, step=step)

            if step % 500 == 0:
                save_checkpoint(model, optimizer, step, config.checkpoint_dir / f"{step}.pt")


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    with torch.no_grad():
        # Use argmax directly on logits without softmax (same result, less memory)
        actual = logits.argmax(dim=-1)
        correct = (actual == targets).sum().item()
    return correct / targets.numel()


def run_evaluation(
    model: TransformerLM, valid: npt.NDArray, config: TrainingConfig, device: str
) -> tuple[float, float]:
    """Validate the model on the validation dataset."""
    model.eval()

    with torch.no_grad():
        valid_loss = 0.0
        valid_accuracy = 0.0

        for _ in range(config.valid_steps):
            batch_X, batch_y = get_batch(
                valid, config.batch_size, config.model.context_length, device
            )  # both are (bs, seq_len)
            valid_logits: torch.Tensor = model(batch_X)  # (bs, seq_len, vocab_size)
            valid_loss += CrossEntropyLoss()(valid_logits, batch_y).item()
            valid_accuracy += compute_accuracy(valid_logits, batch_y)

    model.train()
    return valid_loss / config.valid_steps, valid_accuracy / config.valid_steps


if __name__ == "__main__":
    chz.nested_entrypoint(train)
