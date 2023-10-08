from dataclasses import dataclass


@dataclass
class Config:
    warmup_iters: int = 10000
    resume_ckpt_path: str
    
    # dataset
    dataset_path: str
    num_workers: int = 4

    # model args
    n_layers: int
    n_head: int
    n_embd: int
    block_size: int
    bias: bool
    vocab_size: int
    dropout: float

    # optimizer params
    weight_decay: float
    learning_rate: float
    beta1: float
    beta2: float

    # training params
    warmup_iters: int = 10000
    batch_size: int = 12

    # device data
    device: Device


@dataclass
class ExecutionState:
    iter: int = 0
    min_val_loss: float


def configure() -> Config:
    return Config()