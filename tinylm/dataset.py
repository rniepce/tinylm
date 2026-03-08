"""
Dataset and DataLoader for training tinyLM-pt.

Reads markdown files, tokenizes them, and creates fixed-length
sequences using a sliding window approach.
"""
import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from config import Config


class TextDataset(Dataset):
    """Tokenizes all .md files and produces (input, target) pairs of fixed length."""

    def __init__(self, cfg: Config):
        self.seq_len = cfg.max_seq_len

        # Load tokenizer
        if not os.path.exists(cfg.tokenizer_path):
            raise FileNotFoundError(
                f"Tokenizer not found at {cfg.tokenizer_path}. "
                "Run `python tokenizer_train.py` first."
            )
        self.tokenizer = Tokenizer.from_file(cfg.tokenizer_path)

        # Read and tokenize all markdown files
        files = sorted(glob.glob(os.path.join(cfg.data_dir, "**", "*.md"), recursive=True))
        if not files:
            raise FileNotFoundError(f"No .md files found in {cfg.data_dir}/")

        all_tokens = []
        bos_id = self.tokenizer.token_to_id("<bos>")
        eos_id = self.tokenizer.token_to_id("<eos>")

        for filepath in files:
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            encoded = self.tokenizer.encode(text)
            # Wrap each document with <bos> ... <eos>
            all_tokens.append(bos_id)
            all_tokens.extend(encoded.ids)
            all_tokens.append(eos_id)

        self.tokens = torch.tensor(all_tokens, dtype=torch.long)

        # Number of full sequences we can extract
        # Each sample is seq_len+1 tokens (input = first seq_len, target = last seq_len)
        self.n_samples = max(0, (len(self.tokens) - 1) // self.seq_len)

        print(f"📊 Dataset: {len(self.tokens):,} tokens from {len(files)} file(s)")
        print(f"   Sequences of length {self.seq_len}: {self.n_samples:,}")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = self.tokens[start:end]

        # If chunk is shorter than expected (end of corpus), pad
        if len(chunk) < self.seq_len + 1:
            pad = torch.full((self.seq_len + 1 - len(chunk),), -1, dtype=torch.long)
            chunk = torch.cat([chunk, pad])

        x = chunk[:-1]     # input
        y = chunk[1:]       # target (shifted by 1)
        return x, y


def create_dataloader(cfg: Config) -> DataLoader:
    """Create a DataLoader for training."""
    dataset = TextDataset(cfg)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,       # MPS doesn't benefit from multiprocess loading
        pin_memory=False,
        drop_last=True,
    )
