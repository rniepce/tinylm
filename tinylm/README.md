# tinyLM-pt 🇧🇷

A tiny (~30M parameter) decoder-only transformer language model, built from scratch in PyTorch, trained on Portuguese markdown text. Designed to run locally on a MacBook Pro M3 Max.

## Architecture

This model uses the same building blocks as modern production LLMs (LLaMA, Gemini):

| Component | Detail |
|---|---|
| Normalization | RMSNorm (pre-norm) |
| Positional Encoding | Rotary Positional Embeddings (RoPE) |
| Feed-Forward | SwiGLU activation |
| Attention | Causal multi-head self-attention |
| Weight Tying | Embedding ↔ output projection |

### Hyperparameters

| Parameter | Value |
|---|---|
| Hidden dim (`d_model`) | 512 |
| Attention heads | 8 |
| Layers | 8 |
| FFN dim (SwiGLU) | 1376 |
| Vocab size (BPE) | 8,192 |
| Max sequence length | 512 |
| **Total parameters** | **~29.5M** |

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the tokenizer

```bash
python tokenizer_train.py
```

This trains a BPE tokenizer on all `.md` files in `data/` and saves `tokenizer.json`.

### 3. Train the model

```bash
python train.py
```

Training takes **~1–3 hours** on an M3 Max. The script auto-detects Apple MPS.

Optional overrides:

```bash
python train.py --max_steps 10000 --batch_size 8 --learning_rate 1e-4
```

### 4. Generate text

```bash
# Single prompt
python generate.py --prompt "O Brasil é" --max_tokens 200

# Interactive REPL
python generate.py --interactive

# With specific checkpoint
python generate.py --checkpoint checkpoints/step_005000.pt --prompt "A cidade de"
```

### Generation options

| Flag | Default | Description |
|---|---|---|
| `--temperature` | 0.8 | Sampling temperature |
| `--top_k` | 50 | Top-k filtering |
| `--top_p` | 0.9 | Nucleus sampling threshold |
| `--max_tokens` | 200 | Maximum tokens to generate |

## Project Structure

```
tiny-lm-pt/
├── config.py            # All hyperparameters
├── model.py             # Transformer architecture
├── tokenizer_train.py   # BPE tokenizer training
├── dataset.py           # Dataset + DataLoader
├── train.py             # Training loop (MPS/CUDA/CPU)
├── generate.py          # Text generation + REPL
├── requirements.txt     # Dependencies
├── data/
│   └── corpus.md        # Portuguese training corpus
├── checkpoints/         # Saved during training
└── tokenizer.json       # Created by tokenizer_train.py
```

## Device Support

The code auto-detects the best available device:

1. **MPS** (Apple Silicon GPU) — recommended for M3 Max
2. **CUDA** (NVIDIA GPU)
3. **CPU** (fallback)

## Extending the Dataset

Add more `.md` files to the `data/` directory. The tokenizer and dataset scripts will automatically include all markdown files found recursively in that directory.

For best results, ensure the text is clean Portuguese prose. More data = better model quality.
