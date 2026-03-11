"""
Project 5 — Image Classification
File: 05_image_classification/main.py

Classifies all JPG/PNG images found in IMAGE_DIR.
Drop your images into 05_image_classification/images/ and run.

Run:
    cd hf-projects
    python 05_image_classification/main.py
"""

import os
import glob
from pathlib import Path
from dotenv import load_dotenv
from transformers import pipeline
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

MODEL_NAME = os.getenv("IMAGE_MODEL",   "google/vit-base-patch16-224")
IMAGE_DIR  = os.getenv("IMAGE_DIR",     "./05_image_classification/images")
TOP_K      = int(os.getenv("IMAGE_TOP_K", "3"))

IMAGE_DIR  = str(ROOT / IMAGE_DIR.lstrip("./"))

# ── Print config ──────────────────────────────────────────────────────────────
print("=" * 60)
print("  Project 5 — Image Classification")
print("=" * 60)
print(f"  Model     : {MODEL_NAME}")
print(f"  Image dir : {IMAGE_DIR}")
print(f"  Top-k     : {TOP_K}")
print("-" * 60)

# ── Load pipeline ─────────────────────────────────────────────────────────────
classifier = pipeline("image-classification", model=MODEL_NAME)

# ── Find images ───────────────────────────────────────────────────────────────
image_paths = []
for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
    image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
    image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext.upper())))

image_paths = sorted(set(image_paths))

if not image_paths:
    print(f"\n  No images found in:\n  {IMAGE_DIR}")
    print("\n  Add .jpg or .png files to that folder and run again.")
    print("\n  Example: copy any image and rename it test.jpg")
else:
    for path in image_paths:
        img     = Image.open(path).convert("RGB")
        results = classifier(img, top_k=TOP_K)

        print(f"\n  File: {os.path.basename(path)}  ({img.width}×{img.height})")
        for rank, r in enumerate(results, 1):
            bar   = "█" * int(r["score"] * 30)
            score = r["score"] * 100
            label = r["label"]
            print(f"    {rank}. {score:5.1f}%  {bar:<30s}  {label}")

    print("\n" + "=" * 60)
    print(f"  ✓ Classified {len(image_paths)} image(s)")
    print("=" * 60)
