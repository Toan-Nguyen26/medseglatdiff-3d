"""
Distributed data-parallel helpers.

Shared by train_image_vae_ddp.py and train_latent_diffusion_ddp.py. Both
are launched with torchrun, which sets RANK / LOCAL_RANK / WORLD_SIZE in
the environment:

    torchrun --nproc_per_node=4 -m training.train_image_vae_ddp ...

Running the same module without torchrun falls back to single-process mode,
so the DDP scripts work unchanged on one GPU.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Setup / teardown
# ---------------------------------------------------------------------------

def ddp_setup() -> tuple[int, int, int, torch.device]:
    """
    Initialise the process group when launched under torchrun.

    Returns (rank, local_rank, world_size, device). When not launched under
    torchrun, returns (0, 0, 1, cuda:0-or-cpu) and never touches
    torch.distributed — so the same script runs single-process.
    """
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return 0, 0, 1, device

    rank       = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ["WORLD_SIZE"])

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    return rank, local_rank, world_size, device


def ddp_cleanup() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def barrier() -> None:
    if is_dist():
        dist.barrier()


# ---------------------------------------------------------------------------
# Rank-0 guards
# ---------------------------------------------------------------------------

def is_main(rank: int) -> bool:
    return rank == 0


def rank0_print(rank: int, *a, **kw) -> None:
    if rank == 0:
        print(*a, **kw)


@contextmanager
def rank0_only(rank: int):
    """
    Body runs on rank 0 only; all ranks sync on exit.

    The barrier matters for checkpoint writes — without it a fast rank can
    race ahead and read a file rank 0 has not finished writing.
    """
    try:
        yield rank == 0
    finally:
        barrier()


# ---------------------------------------------------------------------------
# Model unwrapping
# ---------------------------------------------------------------------------

def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """
    Strip the DDP wrapper before saving.

    Saving a DDP-wrapped model prefixes every key with `module.`, and the
    eval scripts (eval/infer_latent.py, eval/eval_image_vae.py) load a bare
    model — so the checkpoint fails to load, and you only find out after the
    full training run. Always save unwrap(model).state_dict().
    """
    return model.module if hasattr(model, "module") else model


# ---------------------------------------------------------------------------
# Metric reduction
# ---------------------------------------------------------------------------

def all_reduce_mean(value: float, device: torch.device) -> float:
    """Average a scalar across ranks. Returns the input unchanged if not distributed."""
    if not is_dist():
        return value
    t = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t / dist.get_world_size()).item()


# ---------------------------------------------------------------------------
# Learning rate
# ---------------------------------------------------------------------------

def scale_lr(base_lr: float, world_size: int, enabled: bool = True) -> float:
    """
    Linear scaling rule: lr grows with the effective batch size.

    With N ranks the effective batch is N× larger, so each optimizer step
    averages N× more samples and you take N× fewer steps per epoch. Without
    scaling the LR, more GPUs buys wall-clock time but costs convergence.
    """
    return base_lr * world_size if enabled else base_lr


def warmup_factor(step: int, warmup_steps: int) -> float:
    """Linear warmup multiplier in [0, 1]. Needed when LR is scaled up."""
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, (step + 1) / warmup_steps)


def apply_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = lr


# ---------------------------------------------------------------------------
# Startup banner
# ---------------------------------------------------------------------------

def print_ddp_banner(
    rank: int,
    world_size: int,
    *,
    per_gpu_batch: int,
    lr_base: float,
    lr_used: float,
    steps_per_epoch: int | None = None,
    num_epochs: int | None = None,
) -> None:
    """
    Print the effective-batch maths at startup.

    The epoch trap: DistributedSampler shards the data, so each rank sees
    1/N of it and one epoch is N× fewer optimizer steps. Keeping the same
    --num_epochs on 4 GPUs therefore trains 4× less, while looking
    identical in the logs. This banner makes that visible before the run.
    """
    if rank != 0:
        return

    print("\n" + "=" * 60)
    print("  Distributed training")
    print("=" * 60)
    print(f"  World size        : {world_size}")
    print(f"  Batch per GPU     : {per_gpu_batch}")
    print(f"  Effective batch   : {per_gpu_batch * world_size}")
    print(f"  LR (base → used)  : {lr_base:.2e} → {lr_used:.2e}")

    if steps_per_epoch is not None and num_epochs is not None:
        total = steps_per_epoch * num_epochs
        print(f"  Steps/epoch       : {steps_per_epoch}")
        print(f"  Total steps       : {total}")
        if world_size > 1:
            single = steps_per_epoch * world_size * num_epochs
            print("")
            print(f"  NOTE: on 1 GPU the same --num_epochs {num_epochs} would be")
            print(f"        ~{single} steps. Sharding across {world_size} ranks makes")
            print(f"        it {total}. To match the single-GPU step count, use")
            print(f"        --num_epochs {num_epochs * world_size}.")
    print("=" * 60 + "\n")
