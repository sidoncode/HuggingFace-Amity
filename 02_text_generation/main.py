"""
Project 2 — Text Generation
File: 02_text_generation/main.py

Run:
    cd hf-projects
    python 02_text_generation/main.py
"""

import sys
from pathlib import Path

# Allow importing config.py from the same folder
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config
from transformers import pipeline

# ── Print config ──────────────────────────────────────────────────────────────
print("=" * 60)
print("  Project 2 — Text Generation")
print("=" * 60)
print(Config.summary())
print("-" * 60)

# ── Load pipeline ─────────────────────────────────────────────────────────────
generator = pipeline(
    "text-generation",
    model=Config.MODEL_NAME,
)

# ── Prompts ───────────────────────────────────────────────────────────────────
prompts = [
    "The future of artificial intelligence is",
    "In a world where machines can think,",
    "Scientists recently discovered that",
]

# ── Generate ──────────────────────────────────────────────────────────────────
for prompt in prompts:
    print(f"\n>>> Prompt: {prompt}")
    print("─" * 50)

    outputs = generator(
        prompt,
        max_new_tokens      = Config.MAX_NEW_TOKENS,
        num_return_sequences= Config.NUM_SEQUENCES,
        temperature         = Config.TEMPERATURE,
        top_p               = Config.TOP_P,
        do_sample           = Config.DO_SAMPLE,
        repetition_penalty  = Config.REPETITION_PENALTY,
    )

    for i, out in enumerate(outputs, 1):
        text = out["generated_text"]
        print(f"\n  [{i}] {text}")

print("\n" + "=" * 60)
print(f"  ✓ Generated {Config.NUM_SEQUENCES} sequences per prompt")
print("=" * 60)
