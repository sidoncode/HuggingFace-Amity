"""
Project 3 — Custom Dataset Loading
File: 03_custom_dataset/load_data.py

Loads a CSV file, prints stats, splits train/test,
and saves to disk for use by preprocess.py.

Run:
    cd hf-projects
    python 03_custom_dataset/load_data.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from datasets import Dataset
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

CSV_PATH   = os.getenv("CUSTOM_CSV_PATH", "./03_custom_dataset/data/sample.csv")
SAVE_PATH  = os.getenv("PROCESSED_PATH",  "./03_custom_dataset/data/processed")

# Resolve paths relative to project root
CSV_PATH  = str(ROOT / CSV_PATH.lstrip("./"))
SAVE_PATH = str(ROOT / SAVE_PATH.lstrip("./"))


def load_csv_as_dataset(path: str) -> Dataset:
    """Load a CSV and return a HuggingFace Dataset."""
    df = pd.read_csv(path)
    print(f"\n  Loaded {len(df)} rows from:\n  {path}\n")
    print(df.to_string(index=False))
    return Dataset.from_pandas(df, preserve_index=False)


def print_stats(dataset: Dataset) -> None:
    """Print label distribution."""
    labels = dataset["label"]
    pos = sum(1 for l in labels if l == 1)
    neg = sum(1 for l in labels if l == 0)
    print(f"\n  Label distribution:")
    print(f"    Positive (1): {pos}")
    print(f"    Negative (0): {neg}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Project 3 — Load Custom Dataset")
    print("=" * 60)

    # Load
    ds = load_csv_as_dataset(CSV_PATH)
    print(f"\n  Features : {ds.features}")
    print_stats(ds)

    # Train / test split (80 / 20)
    split = ds.train_test_split(test_size=0.2, seed=42)
    print(f"\n  Split:")
    print(f"    Train : {len(split['train'])} rows")
    print(f"    Test  : {len(split['test'])} rows")

    # Save
    Path(SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    split.save_to_disk(SAVE_PATH)
    print(f"\n  ✓ Saved processed dataset to:\n  {SAVE_PATH}")
    print("=" * 60)
