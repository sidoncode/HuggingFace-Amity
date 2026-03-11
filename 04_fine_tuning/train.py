"""
Project 4 — Fine-Tuning a Model with the Trainer API
File: 04_fine_tuning/train.py

Run:
    cd hf-projects
    python 04_fine_tuning/train.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import TrainConfig as C
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from datasets import load_dataset
import evaluate
import numpy as np

# ── Print config ──────────────────────────────────────────────────────────────
print("=" * 60)
print("  Project 4 — Fine-Tuning")
print("=" * 60)
print(C.summary())
print("-" * 60)

# ── 1. Load dataset ───────────────────────────────────────────────────────────
print("\n  [1/6] Loading dataset...")
raw = load_dataset(C.DATASET)

# ── 2. Tokenize ───────────────────────────────────────────────────────────────
print("  [2/6] Tokenizing...")
tokenizer = AutoTokenizer.from_pretrained(C.BASE_MODEL)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=C.MAX_LENGTH,
    )

tokenized = raw.map(tokenize, batched=True, remove_columns=["text"])

# ── 3. Load model ─────────────────────────────────────────────────────────────
print("  [3/6] Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    C.BASE_MODEL,
    num_labels=2,
    id2label={0: "NEGATIVE", 1: "POSITIVE"},
    label2id={"NEGATIVE": 0, "POSITIVE": 1},
)

# ── 4. Training arguments ─────────────────────────────────────────────────────
print("  [4/6] Setting up training arguments...")
Path(C.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

push = bool(C.HF_TOKEN and C.HF_USERNAME)

args = TrainingArguments(
    output_dir                  = C.OUTPUT_DIR,
    num_train_epochs            = C.NUM_EPOCHS,
    per_device_train_batch_size = C.BATCH_SIZE,
    per_device_eval_batch_size  = C.BATCH_SIZE * 2,
    learning_rate               = C.LR,
    warmup_steps                = C.WARMUP_STEPS,
    weight_decay                = C.WEIGHT_DECAY,
    fp16                        = C.FP16,
    evaluation_strategy         = "epoch",
    save_strategy               = "epoch",
    load_best_model_at_end      = True,
    logging_dir                 = f"{C.OUTPUT_DIR}/logs",
    logging_steps               = 100,
    metric_for_best_model       = "f1",
    push_to_hub                 = push,
    hub_model_id                = C.hub_repo() if push else None,
    hub_token                   = C.HF_TOKEN   if push else None,
    report_to                   = "none",
)

# ── 5. Metrics ────────────────────────────────────────────────────────────────
print("  [5/6] Setting up metrics...")
acc_metric = evaluate.load("accuracy")
f1_metric  = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        **acc_metric.compute(predictions=preds, references=labels),
        **f1_metric.compute(predictions=preds,  references=labels),
    }

# ── 6. Trainer ────────────────────────────────────────────────────────────────
print("  [6/6] Building Trainer...\n")

trainer = Trainer(
    model           = model,
    args            = args,
    train_dataset   = tokenized["train"],
    eval_dataset    = tokenized["test"],
    tokenizer       = tokenizer,
    data_collator   = DataCollatorWithPadding(tokenizer),
    compute_metrics = compute_metrics,
    callbacks       = [EarlyStoppingCallback(early_stopping_patience=2)],
)

# ── Train ─────────────────────────────────────────────────────────────────────
print("  Starting training...\n" + "=" * 60)
trainer.train()

# ── Push to Hub ───────────────────────────────────────────────────────────────
if push:
    print("\n  Pushing model to Hugging Face Hub...")
    trainer.push_to_hub()
    print(f"  ✓ Model available at: https://huggingface.co/{C.hub_repo()}")
else:
    print("\n  (Skipping Hub push — set HF_TOKEN and HF_USERNAME in .env to enable)")

print("\n" + "=" * 60)
print("  ✓ Training complete!")
print(f"  Model saved to: {C.OUTPUT_DIR}")
print("=" * 60)
