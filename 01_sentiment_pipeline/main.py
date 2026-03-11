"""
Project 1 — Sentiment Analysis Pipeline
File: 01_sentiment_pipeline/main.py

Run:
    cd hf-projects
    python 01_sentiment_pipeline/main.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from transformers import pipeline

# ── Load .env from project root ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

MODEL_NAME = os.getenv(
    "SENTIMENT_MODEL",
    "distilbert-base-uncased-finetuned-sst-2-english",
)

# ── Build pipeline ───────────────────────────────────────────────────────────
print("=" * 60)
print("  Project 1 — Sentiment Analysis Pipeline")
print("=" * 60)
print(f"  Model : {MODEL_NAME}")
print("-" * 60)

classifier = pipeline("sentiment-analysis", model=MODEL_NAME)

# ── Sample texts ─────────────────────────────────────────────────────────────
samples = [
    "Hugging Face is an amazing platform for machine learning!",
    "The model took forever to download — very annoying.",
    "Results are decent but nothing particularly special.",
    "Absolutely loved this tutorial. Highly recommend it.",
    "I expected better. The documentation is confusing.",
    "An okay experience. Neither good nor bad.",
]

# ── Run inference ─────────────────────────────────────────────────────────────
results = classifier(samples)

print("\nResults:\n")
for text, result in zip(samples, results):
    label = result["label"]
    score = result["score"] * 100
    bar   = "█" * int(score / 5)
    print(f"  [{label:8s}  {score:5.1f}%]  {bar}")
    print(f"   {text}")
    print()

print("-" * 60)
print(f"  ✓ Classified {len(samples)} samples with {MODEL_NAME}")
print("=" * 60)
