# 🤗 Hugging Face Projects

A collection of 6 self-contained Python projects using Hugging Face Transformers.
All configuration lives in a single `.env` file — no secrets in code.

---

## ⚡ Quick Start

```bash
# 1. Clone / download this folder
cd hf-projects

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your environment
cp .env.example .env
# Edit .env — add your HF_TOKEN and HF_USERNAME

# 5. Run any project
python 01_sentiment_pipeline/main.py
```

---

## 📁 Project Map

| # | Folder | Files | What it does |
|---|--------|-------|--------------|
| 1 | `01_sentiment_pipeline/` | `main.py` | Classify text sentiment via pipeline |
| 2 | `02_text_generation/` | `config.py`, `main.py` | Generate text completions with GPT-2 |
| 3 | `03_custom_dataset/` | `load_data.py`, `preprocess.py`, `data/sample.csv` | Load, split & tokenize your own CSV |
| 4 | `04_fine_tuning/` | `config.py`, `train.py`, `evaluate.py` | Fine-tune DistilBERT on IMDB |
| 5 | `05_image_classification/` | `main.py`, `images/` | Classify images with ViT |
| 6 | `06_gradio_app/` | `app.py`, `requirements.txt`, `deploy_to_spaces.py` | 3-tab web app + Spaces deploy |

---

## 🔑 .env Keys Reference

| Key | Used by | Default |
|-----|---------|---------|
| `HF_TOKEN` | 4, 6 deploy | *(required for Hub push)* |
| `HF_USERNAME` | 4, 6 deploy | *(required for Hub push)* |
| `SENTIMENT_MODEL` | 1, 6 | `distilbert-base-uncased-finetuned-sst-2-english` |
| `GENERATION_MODEL` | 2, 6 | `gpt2` |
| `IMAGE_MODEL` | 5, 6 | `google/vit-base-patch16-224` |
| `FINETUNE_BASE_MODEL` | 3, 4 | `distilbert-base-uncased` |
| `FINETUNE_DATASET` | 4 | `imdb` |
| `OUTPUT_DIR` | 4 | `./04_fine_tuning/results` |
| `NUM_EPOCHS` | 4 | `3` |
| `BATCH_SIZE` | 4 | `16` |
| `LEARNING_RATE` | 4 | `2e-5` |
| `MAX_LENGTH` | 3, 4 | `256` |
| `WARMUP_STEPS` | 4 | `200` |
| `WEIGHT_DECAY` | 4 | `0.01` |
| `USE_FP16` | 4 | `True` |
| `GEN_MAX_TOKENS` | 2, 6 | `80` |
| `GEN_NUM_SEQ` | 2 | `3` |
| `GEN_TEMP` | 2, 6 | `0.9` |
| `GEN_TOP_P` | 2 | `0.95` |
| `GEN_SAMPLE` | 2 | `True` |
| `GEN_REP_PEN` | 2 | `1.1` |
| `CUSTOM_CSV_PATH` | 3 | `./03_custom_dataset/data/sample.csv` |
| `PROCESSED_PATH` | 3 | `./03_custom_dataset/data/processed` |
| `TOKENIZED_PATH` | 3 | `./03_custom_dataset/data/tokenized` |
| `IMAGE_DIR` | 5 | `./05_image_classification/images` |
| `IMAGE_TOP_K` | 5 | `3` |
| `GRADIO_PORT` | 6 | `7860` |
| `GRADIO_SHARE` | 6 | `False` |
| `APP_TITLE` | 6 | `My HF Demo` |

---

## 🚀 Run Order for Project 3 & 4

```bash
# Project 3 — must run load_data first, then preprocess
python 03_custom_dataset/load_data.py
python 03_custom_dataset/preprocess.py

# Project 4 — must run train first, then evaluate
python 04_fine_tuning/train.py
python 04_fine_tuning/evaluate.py
```

---

## 🌐 Deploy Project 6 to Hugging Face Spaces

```bash
# Make sure HF_TOKEN and HF_USERNAME are set in .env
python 06_gradio_app/deploy_to_spaces.py
```

Your app will be live at `https://huggingface.co/spaces/<HF_USERNAME>/<APP_TITLE>`.

---

## 📦 Requirements

- Python 3.8+
- ~4 GB disk (model weights cached in `~/.cache/huggingface/`)
- GPU optional — all projects run on CPU

---

*Built with 🤗 Hugging Face Transformers · Datasets · Gradio*
