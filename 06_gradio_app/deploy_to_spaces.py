"""
Project 6 — Deploy to Hugging Face Spaces
File: 06_gradio_app/deploy_to_spaces.py

Automates Space creation and upload.
Set HF_TOKEN and HF_USERNAME in .env before running.

Run:
    cd hf-projects
    python 06_gradio_app/deploy_to_spaces.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

HF_TOKEN    = os.getenv("HF_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME")
APP_TITLE   = os.getenv("APP_TITLE", "my-hf-demo")

if not HF_TOKEN or HF_TOKEN.startswith("hf_xxx"):
    print("ERROR: Set a real HF_TOKEN in your .env file.")
    print("  Get one at: https://huggingface.co/settings/tokens")
    raise SystemExit(1)

if not HF_USERNAME or HF_USERNAME == "your-hf-username":
    print("ERROR: Set your HF_USERNAME in your .env file.")
    raise SystemExit(1)

# Derive repo name from APP_TITLE
REPO_NAME = APP_TITLE.lower().replace(" ", "-")
REPO_ID   = f"{HF_USERNAME}/{REPO_NAME}"

api = HfApi(token=HF_TOKEN)

# ── Create Space ──────────────────────────────────────────────────────────────
print(f"\n  Creating Space: {REPO_ID}")
api.create_repo(
    repo_id   = REPO_ID,
    repo_type = "space",
    space_sdk = "gradio",
    exist_ok  = True,
    private   = False,
)
print(f"  ✓ Space ready")

# ── Upload app folder ─────────────────────────────────────────────────────────
app_folder = Path(__file__).resolve().parent
print(f"\n  Uploading from: {app_folder}")

api.upload_folder(
    folder_path = str(app_folder),
    repo_id     = REPO_ID,
    repo_type   = "space",
    ignore_patterns=["*.pyc", "__pycache__", ".env", "deploy_to_spaces.py"],
)

print(f"\n  ✓ Upload complete!")
print(f"  Live URL: https://huggingface.co/spaces/{REPO_ID}")
print(f"\n  NOTE: On Spaces, add your secrets via:")
print(f"  https://huggingface.co/spaces/{REPO_ID}/settings")
