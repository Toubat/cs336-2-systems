import hashlib
from pathlib import Path
from typing import Literal

import chz

from cs336_basics.utils import get_latest_checkpoint_path

Dataset = Literal["tinystories", "owt"]

DATASET_TRAIN_PATHS: dict[Dataset, str] = {
    "tinystories": "data/tinystories_gpt4_train.npy",
    "owt": "data/owt_train.npy",
}

DATASET_VALID_PATHS: dict[Dataset, str] = {
    "tinystories": "data/tinystories_gpt4_valid.npy",
    "owt": "data/owt_valid.npy",
}

TOKENIZER_VOCAB_PATHS: dict[Dataset, str] = {
    "tinystories": "tokenizers/tinystories_gpt4_train_vocab.json",
    "owt": "tokenizers/owt_train_vocab.json",
}

TOKENIZER_MERGES_PATHS: dict[Dataset, str] = {
    "tinystories": "tokenizers/tinystories_gpt4_train_merges.txt",
    "owt": "tokenizers/owt_train_merges.txt",
}


@chz.chz(typecheck=True)
class ModelConfig:
    vocab_size: int = 10000
    context_length: int = 512
    d_model: int = 1024
    d_ff: int = 2730
    theta: float = 10000.0
    num_layers: int = 8
    num_heads: int = 16


@chz.chz(typecheck=True)
class TrainingConfig:
    name: str
    volume_path: Path = Path(".")
    remote: bool = False
    dataset: Dataset = "owt"
    epochs: int = 20000
    batch_size: int = 64
    lr_max: float = 5e-3
    lr_min: float = 5e-6
    warmup_t: int = 2000
    cosine_cycle_t: int = 15000
    gradient_clipping_norm: float = 1.0  # Back to 1.0, will also clip RMSNorm separately
    model: ModelConfig
    valid_interval: int = 50
    valid_steps: int = 10

    @chz.init_property
    def wandb_id(self) -> str:
        return hashlib.sha256(self.name.encode()).hexdigest()

    @chz.init_property
    def wandb_config(self) -> dict:
        return chz.asdict(self)

    @chz.init_property
    def checkpoint_dir(self) -> Path:
        return self.volume_path / ".checkpoints" / self.name

    @chz.init_property
    def checkpoint_data_path(self) -> Path | None:
        return get_latest_checkpoint_path(self.checkpoint_dir)
