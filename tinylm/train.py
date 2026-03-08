"""
Training loop for tinyLM-pt.

Features:
  • AdamW with cosine annealing + linear warmup
  • Gradient clipping
  • Periodic logging (loss, tokens/sec, lr)
  • Checkpoint saving
  • Apple MPS / CUDA / CPU auto-detection

Usage:
    python train.py [--max_steps 5000] [--batch_size 16] [--learning_rate 3e-4]
"""
import argparse
import math
import os
import time
import torch
from config import Config
from model import TinyLM
from dataset import create_dataloader


def get_lr(step: int, cfg: Config) -> float:
    """Linear warmup + cosine decay learning rate schedule."""
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    decay_ratio = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    decay_ratio = min(decay_ratio, 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.learning_rate * max(coeff, 0.1)   # floor at 10% of max lr


def train(cfg: Config):
    print("=" * 60)
    print("  tinyLM-pt — Training")
    print("=" * 60)
    print(f"  Device:       {cfg.device}")
    print(f"  Max steps:    {cfg.max_steps}")
    print(f"  Batch size:   {cfg.batch_size}")
    print(f"  Seq length:   {cfg.max_seq_len}")
    print(f"  LR:           {cfg.learning_rate}")
    print("=" * 60)

    device = torch.device(cfg.device)

    # --- Data ---
    dataloader = create_dataloader(cfg)
    data_iter = iter(dataloader)

    # --- Model ---
    model = TinyLM(cfg).to(device)
    total_params = model.count_parameters()
    print(f"\n🧠 Model: {total_params / 1e6:.1f}M parameters")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=cfg.weight_decay,
    )

    # --- Checkpoint dir ---
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # --- Training loop ---
    model.train()
    total_tokens = 0
    t0 = time.time()

    for step in range(1, cfg.max_steps + 1):
        # Get next batch (cycle through data)
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)

        x = x.to(device)
        y = y.to(device)

        # Update learning rate
        lr = get_lr(step, cfg)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Forward pass
        _, loss = model(x, targets=y)

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        optimizer.step()

        total_tokens += x.numel()

        # --- Logging ---
        if step % cfg.log_interval == 0:
            dt = time.time() - t0
            tokens_per_sec = total_tokens / dt
            print(
                f"  step {step:>5d}/{cfg.max_steps} │ "
                f"loss {loss.item():.4f} │ "
                f"lr {lr:.2e} │ "
                f"{tokens_per_sec:,.0f} tok/s"
            )

        # --- Save checkpoint ---
        if step % cfg.save_interval == 0 or step == cfg.max_steps:
            ckpt_path = os.path.join(cfg.checkpoint_dir, f"step_{step:06d}.pt")
            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": cfg,
                    "loss": loss.item(),
                },
                ckpt_path,
            )
            print(f"  💾 Checkpoint saved → {ckpt_path}")

    elapsed = time.time() - t0
    print(f"\n✅ Training complete in {elapsed / 60:.1f} minutes")
    print(f"   Total tokens processed: {total_tokens:,}")
    print(f"   Final loss: {loss.item():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train tinyLM-pt")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.learning_rate is not None:
        cfg.learning_rate = args.learning_rate
    if args.device is not None:
        cfg.device = args.device

    train(cfg)
