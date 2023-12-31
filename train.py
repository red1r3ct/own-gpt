import os
import time
import math
import json
import sys
import torch

from contextlib import nullcontext
from datasets import load_from_disk

from lib.model import GPTConfig, GPT
from data.arxiv import make_arxiv_abstracts_dataset
from lib.encoders import byte_pair_encoder, vocab_size, byte_pair_decoder, pad_token
from lib.device import Device
from data.collate_fn import make_collate_fn
from datasets import load_from_disk


if __name__ == '__main__':
    with open(sys.argv[1], 'r') as f:
        run_cfg = json.load(f)
    
    eval_interval = run_cfg['train']['eval_interval']
    log_interval = run_cfg['train']['log_interval']
    eval_iters = run_cfg['train']['eval_iters']
    gradient_accumulation_steps = run_cfg['train']['gradient_accumulation_steps']
    always_save_checkpoint = True  # if True, always save a checkpoint after each eval
    init_from = "scratch"  # 'scratch' or 'resume' or 'gpt2*'
    # wandb logging
    wandb_log = True  # disabled by default
    wandb_project = run_cfg['wandb']['project']
    wandb_run_name = f"{run_cfg['name']}__{time.time()}"
    out_dir = f"./logs/{wandb_run_name}/"
    # data
    dataset = run_cfg['dataset']
    batch_size = run_cfg['train']['batch_size']  # if gradient_accumulation_steps > 1, this is the micro-batch size
    block_size = run_cfg['model']['block_size']
    # model
    n_layer = run_cfg['model']['n_layer']
    n_head = run_cfg['model']['n_head']
    n_embd = run_cfg['model']['n_embd_frac'] * n_head
    dropout = run_cfg['model']['dropout'] # for pretraining 0 is good, for finetuning try 0.1+
    bias = run_cfg['model']['bias']  # do we use bias inside LayerNorm and Linear layers?
    # adamw optimizer
    learning_rate = run_cfg['train']['learning_rate']  # max learning rate
    max_iters = run_cfg['train']['max_iters']  # total number of training iterations
    weight_decay = run_cfg['train']['weight_decay']
    beta1 = run_cfg['train']['beta1']
    beta2 = run_cfg['train']['beta2']
    grad_clip = run_cfg['train']['grad_clip']  # clip gradients at this value, or disable if == 0.0
    # learning rate decay settings
    decay_lr = run_cfg['train']['decay_lr']  # whether to decay the learning rate
    warmup_iters = run_cfg['train']['warmup_iters']  # how many steps to warm up for
    lr_decay_iters = run_cfg['train']['lr_decay_iters']  # should be ~= max_iters per Chinchilla
    min_lr = run_cfg['train']['min_lr']  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    # system
    device = Device()
    dtype = device.get_data_type()
    device_type = device.get_device_type()
    compile = True  # use PyTorch 2.0 to compile the model to be faster
    # -----------------------------------------------------------------------------
    config_keys = [
        k
        for k, v in globals().items()
        if not k.startswith("_") and isinstance(v, (int, float, bool, str))
    ]
    config = {k: globals()[k] for k in config_keys}  # will be useful for logging
    # -----------------------------------------------------------------------------

    master_process = True
    seed_offset = 0
    tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

    if master_process:
        os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(1337 + seed_offset)
    ctx = device.get_context()

    # poor man's data loader
    if not os.path.exists(dataset):
        make_arxiv_abstracts_dataset(dataset, encoder=byte_pair_encoder)

    all_data = load_from_disk(dataset).with_format('torch')
    train_loader = iter(torch.utils.data.DataLoader(
        all_data['train'],
        batch_size=batch_size,
        collate_fn=make_collate_fn(block_size, device=device_type),
    ))
    val_loader = iter(torch.utils.data.DataLoader(
        all_data['test'],
        batch_size=batch_size,
        collate_fn=make_collate_fn(block_size, device=device_type),
    ))


    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0
    best_val_loss = 1e9

    # model init
    model_args = dict(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
        bias=bias,
        vocab_size=None,
        dropout=dropout,
        pad_token_idx=pad_token,
        batch_size=batch_size,
        device=device_type,
    )  # start with model_args from command line
    if init_from == "scratch":
        # init a new model from scratch
        print("Initializing a new model from scratch")
        model_args["vocab_size"] = vocab_size
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif init_from == "resume":
        print(f"Resuming training from {out_dir}")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(out_dir, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=device_type)
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
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]
    # crop down the model block size if desired, using model surgery
    if block_size < model.config.block_size:
        model.crop_block_size(block_size)
        model_args[
            "block_size"
        ] = block_size  # so that the checkpoint will have the right value
    model.to(device_type)

    # optimizer
    optimizer = model.configure_optimizers(
        weight_decay, learning_rate, (beta1, beta2), device_type
    )
    if init_from == "resume":
        optimizer.load_state_dict(checkpoint["optimizer"])
    checkpoint = None  # free up memory

    # compile the model
    if device_type == 'mps':
        print("compiling the model for mps... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model, backend="aot_eager")
    if device_type == 'cuda':
        print("compiling the model for cuda... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)


    # logging
    if wandb_log and master_process:
        import wandb

        wandb.init(project=wandb_project, name=wandb_run_name, config=config)

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split, loader in [('train', train_loader), ('val', val_loader)]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = next(loader)
                with ctx:
                    _, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out


    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return min_lr + coeff * (learning_rate - min_lr)

    # training loop
    t0 = time.time()
    local_iter_num = 0  # number of iterations in the lifetime of this process
    raw_model = model  # unwrap DDP container if needed
    running_mfu = -1.0
    while True:
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % eval_interval == 0 and master_process:
            losses = estimate_loss()
            print(
                f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
            if wandb_log:
                wandb.log(
                    {
                        "iter": iter_num,
                        "train/loss": losses["train"],
                        "val/loss": losses["val"],
                        "lr": lr,
                    }
                )
            if losses["val"] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses["val"]
                if iter_num > 0:
                    checkpoint = {
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_args": model_args,
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                        "config": config,
                    }
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(gradient_accumulation_steps):
            X, Y = next(train_loader)
            with ctx:
                logits, loss = model(X, Y)
                loss = (
                    loss / gradient_accumulation_steps
                )  # scale the loss to account for gradient accumulation
            # backward pass, with gradient scaling if training in fp16
            loss.backward()
        # clip the gradient
        if grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # step the optimizer and scaler if training in fp16
        optimizer.step()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0 and master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * gradient_accumulation_steps
            print(
                f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms"
            )
        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > max_iters:
            break
