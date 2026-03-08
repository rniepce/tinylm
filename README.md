# SLM Specialist 🧠

Fine-tuned specialist language models running locally on Apple Silicon via **MLX QLoRA**.

## Models

| # | Specialist | Base Model | Status |
|---|---|---|---|
| 3 | 📚 English Literature | Mistral 7B v0.3 | Ready |
| 2 | ⚖️ Jurídico TJMG | Llama 3.1 8B | Ready |
| 4 | 📊 Data Analysis | Qwen 2.5 7B | Ready |
| 1 | 🧑‍💻 Code Assistant | Qwen 2.5 Coder 7B | Ready |

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Train a specialist (e.g., literature)
chmod +x train.sh
./train.sh literature

# Chat with it
python serve.py --model literature
# Open http://localhost:8000
```

## Architecture

All models use **QLoRA** (Quantized Low-Rank Adaptation) on 4-bit quantized base models from `mlx-community` on HuggingFace. This enables fine-tuning 7B models with ~20GB RAM on Apple Silicon.

## Project Structure

```
models/<name>/config.yaml    — MLX LoRA training config
models/<name>/data/          — Training data (JSONL chat format)
adapters/<name>/             — LoRA weights (created after training)
serve.py                     — Unified chat server
static/index.html            — Chat UI with model selector
train.sh                     — One-command training
```
