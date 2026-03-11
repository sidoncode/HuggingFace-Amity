"""
Project 3 — Dataset Preprocessing / Tokenization
File: 03_custom_dataset/preprocess.py

Loads the saved split from load_data.py, tokenizes it,
and saves a PyTorch-ready dataset to disk.

Run AFTER load_data.py:
    cd hf-projects
    python 03_custom_dataset/preprocess.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from datasets import load_from_disk
from transformers import AutoTokenizer

# ── Config ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

MODEL_NAME     = os.getenv("FINETUNE_BASE_MODEL", "distilbert-base-uncased")
SAVE_PATH      = os.getenv("PROCESSED_PATH",  "./03_custom_dataset/data/processed")
TOKENIZED_PATH = os.getenv("TOKENIZED_PATH",  "./03_custom_dataset/data/tokenized")
MAX_LENGTH     = int(os.getenv("MAX_LENGTH", "128"))

SAVE_PATH      = str(ROOT / SAVE_PATH.lstrip("./"))
TOKENIZED_PATH = str(ROOT / TOKENIZED_PATH.lstrip("./"))


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Project 3 — Tokenize Dataset")
    print("=" * 60)
    print(f"  Base model : {MODEL_NAME}")
    print(f"  Max length : {MAX_LENGTH}")
    print(f"  Input      : {SAVE_PATH}")
    print(f"  Output     : {TOKENIZED_PATH}")
    print("-" * 60)

    # Load split dataset from disk
    dataset   = load_from_disk(SAVE_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    print("\n  Tokenizing...")
    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # Save
    Path(TOKENIZED_PATH).mkdir(parents=True, exist_ok=True)
    tokenized.save_to_disk(TOKENIZED_PATH)

    # Verify
    sample = tokenized["train"][0]
    print(f"\n  Sample input_ids shape  : {sample['input_ids'].shape}")
    print(f"  Sample attention_mask   : {sample['attention_mask'].shape}")
    print(f"  Sample label            : {sample['label']}")

    print(f"\n  ✓ Tokenized dataset saved to:\n  {TOKENIZED_PATH}")
    print("=" * 60)
