"""
Project 4 — Fine-Tuning Evaluation
File: 04_fine_tuning/evaluate.py

Loads the fine-tuned model from OUTPUT_DIR and evaluates
it on the test split, printing a full classification report.

Run AFTER train.py:
    cd hf-projects
    python 04_fine_tuning/evaluate.py
"""

import os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import TrainConfig as C
from transformers import pipeline
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix

# ── Load model from output dir ────────────────────────────────────────────────
print("=" * 60)
print("  Project 4 — Evaluation")
print("=" * 60)
print(f"  Model dir : {C.OUTPUT_DIR}")
print(f"  Dataset   : {C.DATASET}")
print("-" * 60)

if not Path(C.OUTPUT_DIR).exists():
    print("\n  ERROR: Output directory not found.")
    print("  Run train.py first to generate a fine-tuned model.\n")
    sys.exit(1)

clf = pipeline("sentiment-analysis", model=C.OUTPUT_DIR)

# ── Load test set ─────────────────────────────────────────────────────────────
print("\n  Loading test split (first 500 examples)...")
test = load_dataset(C.DATASET, split="test[:500]")

# ── Label mapping ─────────────────────────────────────────────────────────────
label_map = {
    "LABEL_0": 0, "LABEL_1": 1,
    "NEGATIVE": 0, "POSITIVE": 1,
}

# ── Predict ───────────────────────────────────────────────────────────────────
print("  Running inference...")
preds   = [label_map[clf(t)[0]["label"]] for t in test["text"]]
actuals = test["label"]

# ── Report ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Classification Report")
print("=" * 60)
print(classification_report(actuals, preds, target_names=["Negative", "Positive"]))

print("  Confusion Matrix")
print("  (rows=actual, cols=predicted)\n")
cm = confusion_matrix(actuals, preds)
print(f"             Neg    Pos")
print(f"  Actual Neg  {cm[0][0]:4d}  {cm[0][1]:4d}")
print(f"  Actual Pos  {cm[1][0]:4d}  {cm[1][1]:4d}")
print("\n" + "=" * 60)
