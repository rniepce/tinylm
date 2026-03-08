"""
Interactive text generation with a trained tinyLM-pt model.

Usage:
    # Single prompt
    python generate.py --prompt "O Brasil é" --max_tokens 100

    # Interactive REPL
    python generate.py --interactive

    # Specify checkpoint
    python generate.py --checkpoint checkpoints/step_005000.pt --prompt "A cidade de"
"""
import argparse
import glob
import os
import torch
from tokenizers import Tokenizer
from config import Config
from model import TinyLM


def load_model(checkpoint_path: str, device: str = None) -> tuple[TinyLM, Tokenizer, Config]:
    """Load model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    if device is not None:
        cfg.device = device

    model = TinyLM(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(torch.device(cfg.device))
    model.eval()

    tokenizer = Tokenizer.from_file(cfg.tokenizer_path)

    step = ckpt.get("step", "?")
    loss = ckpt.get("loss", "?")
    print(f"✅ Loaded checkpoint: {checkpoint_path}")
    print(f"   Step: {step} │ Loss: {loss}")
    print(f"   Device: {cfg.device}")
    print(f"   Parameters: {model.count_parameters() / 1e6:.1f}M")

    return model, tokenizer, cfg


def generate_text(
    model: TinyLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
) -> str:
    """Generate text from a prompt."""
    device = next(model.parameters()).device

    # Encode prompt
    bos_id = tokenizer.token_to_id("<bos>")
    encoded = tokenizer.encode(prompt)
    tokens = [bos_id] + encoded.ids
    idx = torch.tensor([tokens], dtype=torch.long, device=device)

    # Generate
    output_ids = model.generate(
        idx,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    # Decode (skip BOS token)
    all_ids = output_ids[0].tolist()
    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id in all_ids[1:]:
        all_ids = all_ids[1 : all_ids.index(eos_id, 1)]
    else:
        all_ids = all_ids[1:]

    return tokenizer.decode(all_ids)


def find_latest_checkpoint(checkpoint_dir: str = "checkpoints") -> "str | None":
    """Find the most recent checkpoint file."""
    pattern = os.path.join(checkpoint_dir, "step_*.pt")
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def main():
    parser = argparse.ArgumentParser(description="Generate text with tinyLM-pt")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt")
    parser.add_argument("--max_tokens", type=int, default=200, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--interactive", action="store_true", help="Interactive REPL mode")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    # Find checkpoint
    ckpt_path = args.checkpoint or find_latest_checkpoint()
    if ckpt_path is None:
        print("❌ No checkpoint found. Train the model first with: python train.py")
        return

    model, tokenizer, cfg = load_model(ckpt_path, args.device)

    if args.interactive:
        print("\n" + "=" * 60)
        print("  tinyLM-pt — Interactive Generation")
        print("  Type a prompt and press Enter. Type 'quit' to exit.")
        print("=" * 60 + "\n")

        while True:
            try:
                prompt = input("📝 Prompt: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 Bye!")
                break

            if prompt.lower() in ("quit", "exit", "q"):
                print("👋 Bye!")
                break
            if not prompt:
                continue

            text = generate_text(
                model, tokenizer, prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )
            print(f"\n🤖 {text}\n")

    elif args.prompt:
        text = generate_text(
            model, tokenizer, args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        print(f"\n{text}")

    else:
        print("Provide --prompt or --interactive. Use --help for options.")


if __name__ == "__main__":
    main()
