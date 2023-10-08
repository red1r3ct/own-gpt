import os
import torch
import math
from dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset
from lib.model import GPT, GPTConfig


class Device:
    def __init__(self):
        pass

    def get_device_type(self):
        return "cpu"

    def get_data_type(self):
        return "bfloat16"


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


def make_model(config: Config):
    model_args = dict(
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        block_size=config.block_size,
        bias=config.bias,
        vocab_size=config.vocab_size,
        dropout=config.dropout,
    )
    state = ExecutionState(0, math.inf)
    if config.resume_ckpt_path == "":
        # init a new model from scratch
        print("Initializing a new model from scratch...")
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    else:
        # init from checkpoint
        print(f"Resuming training from {config.resume_ckpt_path}...")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(config.resume_ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location=config.device.get_device_type())
        checkpoint_model_args = checkpoint["model_args"]
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
            model_args[k] = checkpoint_model_args[k]
        # create the model
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint["model"]
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        state.min_val_loss = checkpoint["iter_num"]
        state.min_val_loss = checkpoint["best_val_loss"]
    # crop down the model block size if desired, using model surgery
    if config.block_size < model.config.block_size:
        model.crop_block_size(config.block_size)
        model_args[
            "block_size"
        ] = config.block_size  # so that the checkpoint will have the right value
    model.to(config.device.get_device_type())

    # optimizer
    optimizer = model.configure_optimizers(
        config.weight_decay,
        config.learning_rate,
        (config.beta1, config.beta2),
        config.device.get_device_type(),
    )
    if config.resume_ckpt_path != "":
        optimizer.load_state_dict(checkpoint["optimizer"])

    return model, optimizer, state


def make_device(config: Config) -> Device:
    return Device()

def make_dataset(config) -> torch.utils.data.DataLoader:
    

if __name__ == "__main__":
    config = configure()
    config.device = make_device(config)

    dataset = make_dataset(config)
    model, optimizer, state = make_model(config)

    state = train_loop(model, optimizer, dataset, state)
