"""
Train a BPE tokenizer on the Portuguese markdown corpus.

Usage:
    python tokenizer_train.py [--vocab_size 8192] [--data_dir data]

Outputs:
    tokenizer.json  — HuggingFace-compatible tokenizer file
"""
import argparse
import glob
import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors


def train_tokenizer(data_dir: str = "data", vocab_size: int = 8192, output: str = "tokenizer.json"):
    # Collect all .md files in the data directory
    files = glob.glob(os.path.join(data_dir, "**", "*.md"), recursive=True)
    if not files:
        raise FileNotFoundError(f"No .md files found in {data_dir}/")

    print(f"📚 Found {len(files)} file(s) for tokenizer training")
    for f in files:
        size = os.path.getsize(f)
        print(f"   • {f} ({size / 1024:.1f} KB)")

    # Create a BPE tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    # Define special tokens
    special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
        min_frequency=2,
    )

    # Train on files
    tokenizer.train(files, trainer)
    tokenizer.save(output)

    print(f"\n✅ Tokenizer saved to {output}")
    print(f"   Vocab size: {tokenizer.get_vocab_size()}")

    # Quick roundtrip test
    test = "O Brasil é o maior país da América do Sul."
    encoded = tokenizer.encode(test)
    decoded = tokenizer.decode(encoded.ids)
    print(f"\n🔍 Roundtrip test:")
    print(f"   Input:   {test}")
    print(f"   Tokens:  {encoded.tokens[:20]}{'...' if len(encoded.tokens) > 20 else ''}")
    print(f"   IDs:     {encoded.ids[:20]}{'...' if len(encoded.ids) > 20 else ''}")
    print(f"   Decoded: {decoded}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BPE tokenizer on Portuguese markdown")
    parser.add_argument("--vocab_size", type=int, default=8192)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output", type=str, default="tokenizer.json")
    args = parser.parse_args()
    train_tokenizer(args.data_dir, args.vocab_size, args.output)
