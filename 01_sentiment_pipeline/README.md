# Project 1 — Sentiment Analysis Pipeline

Uses the Hugging Face `pipeline()` API to classify text sentiment.
Model name is read from `.env` — change `SENTIMENT_MODEL` to swap models instantly.

## Files
```
01_sentiment_pipeline/
└── main.py   ← entry point
```

## Run
```bash
cd hf-projects
python 01_sentiment_pipeline/main.py
```

## .env keys used
| Key | Default |
|-----|---------|
| `SENTIMENT_MODEL` | `distilbert-base-uncased-finetuned-sst-2-english` |

## Swap the model
Edit `.env`:
```
SENTIMENT_MODEL=cardiffnlp/twitter-roberta-base-sentiment-latest
```
No code changes needed.
