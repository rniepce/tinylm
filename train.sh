#!/bin/bash
# ─────────────────────────────────────────────────────────────
# train.sh — Fine-tune a specialist SLM with MLX LoRA
#
# Usage:
#   ./train.sh literature        # Fine-tune Literature model
#   ./train.sh juridico          # Fine-tune Jurídico model
#   ./train.sh data-analyst      # Fine-tune Data Analysis model
#   ./train.sh code-assistant    # Fine-tune Code Assistant model
# ─────────────────────────────────────────────────────────────
set -e

MODEL_NAME="${1:?Usage: ./train.sh <model-name> (literature|juridico|data-analyst|code-assistant)}"
CONFIG="models/${MODEL_NAME}/config.yaml"

if [ ! -f "$CONFIG" ]; then
    echo "❌ Config not found: $CONFIG"
    echo "Available models:"
    ls -d models/*/config.yaml 2>/dev/null | sed 's|models/||;s|/config.yaml||' | sed 's/^/  • /'
    exit 1
fi

echo "============================================================"
echo "  SLM Specialist — LoRA Fine-Tuning"
echo "============================================================"
echo "  Model:  ${MODEL_NAME}"
echo "  Config: ${CONFIG}"
echo "============================================================"
echo ""

# Run MLX LoRA training
python -m mlx_lm.lora --config "$CONFIG"

echo ""
echo "✅ Training complete for: ${MODEL_NAME}"
echo "   Adapter saved to: adapters/${MODEL_NAME}/"
echo ""
echo "To test generation:"
echo "   python -m mlx_lm.generate \\"
echo "     --model \$(grep '^model:' $CONFIG | awk '{print \$2}' | tr -d '\"') \\"
echo "     --adapter-path adapters/${MODEL_NAME} \\"
echo "     --prompt 'Hello'"
