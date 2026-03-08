"""
Centralized configuration for tinyLM-pt.
All hyperparameters in one place.
"""
from dataclasses import dataclass, field
import torch


@dataclass
class Config:
    # ── Model Architecture ──────────────────────────────────────────
    vocab_size: int = 8192
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 8
    d_ff: int = 1376          # SwiGLU intermediate dim ≈ 2.69 × d_model
    max_seq_len: int = 512
    dropout: float = 0.1
    rope_theta: float = 10000.0

    # ── Training ────────────────────────────────────────────────────
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_steps: int = 5000
    warmup_steps: int = 200
    grad_clip: float = 1.0
    eval_interval: int = 250
    save_interval: int = 500
    log_interval: int = 10

    # ── Paths ───────────────────────────────────────────────────────
    data_dir: str = "data"
    tokenizer_path: str = "tokenizer.json"
    checkpoint_dir: str = "checkpoints"

    # ── Device ──────────────────────────────────────────────────────
    device: str = field(default_factory=lambda: Config._detect_device())

    @staticmethod
    def _detect_device() -> str:
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"
