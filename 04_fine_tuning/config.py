"""
Project 4 — Fine-Tuning
File: 04_fine_tuning/config.py

All training hyper-parameters loaded from .env.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


class TrainConfig:
    # Model & data
    BASE_MODEL   = os.getenv("FINETUNE_BASE_MODEL", "distilbert-base-uncased")
    DATASET      = os.getenv("FINETUNE_DATASET",    "imdb")

    # Output
    OUTPUT_DIR   = str(ROOT / os.getenv("OUTPUT_DIR", "./04_fine_tuning/results").lstrip("./"))

    # Hyper-params
    NUM_EPOCHS   = int(os.getenv("NUM_EPOCHS",      "3"))
    BATCH_SIZE   = int(os.getenv("BATCH_SIZE",      "16"))
    LR           = float(os.getenv("LEARNING_RATE", "2e-5"))
    MAX_LENGTH   = int(os.getenv("MAX_LENGTH",      "256"))
    WARMUP_STEPS = int(os.getenv("WARMUP_STEPS",    "200"))
    WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY",  "0.01"))
    FP16         = os.getenv("USE_FP16", "True") == "True"

    # Hub
    HF_TOKEN     = os.getenv("HF_TOKEN")
    HF_USERNAME  = os.getenv("HF_USERNAME")

    @classmethod
    def hub_repo(cls) -> str:
        if cls.HF_USERNAME:
            return f"{cls.HF_USERNAME}/{cls.DATASET}-classifier"
        return ""

    @classmethod
    def summary(cls) -> str:
        return (
            f"Base model   : {cls.BASE_MODEL}\n"
            f"Dataset      : {cls.DATASET}\n"
            f"Output dir   : {cls.OUTPUT_DIR}\n"
            f"Epochs       : {cls.NUM_EPOCHS}\n"
            f"Batch size   : {cls.BATCH_SIZE}\n"
            f"Learning rate: {cls.LR}\n"
            f"Max length   : {cls.MAX_LENGTH}\n"
            f"Warmup steps : {cls.WARMUP_STEPS}\n"
            f"Weight decay : {cls.WEIGHT_DECAY}\n"
            f"FP16         : {cls.FP16}\n"
            f"Hub repo     : {cls.hub_repo() or '(not set)'}"
        )
