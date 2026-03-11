"""
Project 2 — Text Generation
File: 02_text_generation/config.py

Centralised config — all values loaded from .env.
Import this in main.py instead of repeating os.getenv() calls.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


class Config:
    # Model
    MODEL_NAME         = os.getenv("GENERATION_MODEL", "gpt2")

    # Generation hyper-params
    MAX_NEW_TOKENS     = int(os.getenv("GEN_MAX_TOKENS", "80"))
    NUM_SEQUENCES      = int(os.getenv("GEN_NUM_SEQ",    "3"))
    TEMPERATURE        = float(os.getenv("GEN_TEMP",     "0.9"))
    TOP_P              = float(os.getenv("GEN_TOP_P",    "0.95"))
    DO_SAMPLE          = os.getenv("GEN_SAMPLE", "True") == "True"
    REPETITION_PENALTY = float(os.getenv("GEN_REP_PEN", "1.1"))

    @classmethod
    def summary(cls) -> str:
        return (
            f"Model            : {cls.MODEL_NAME}\n"
            f"Max new tokens   : {cls.MAX_NEW_TOKENS}\n"
            f"Num sequences    : {cls.NUM_SEQUENCES}\n"
            f"Temperature      : {cls.TEMPERATURE}\n"
            f"Top-p            : {cls.TOP_P}\n"
            f"Do sample        : {cls.DO_SAMPLE}\n"
            f"Repetition pen.  : {cls.REPETITION_PENALTY}"
        )
