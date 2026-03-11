"""
Project 6 — Gradio Multi-Tab Web App
File: 06_gradio_app/app.py

Launches a 3-tab Gradio interface:
  • Tab 1 — Sentiment Analysis
  • Tab 2 — Text Generation
  • Tab 3 — Image Classification

All model names, port, and title come from .env.

Run locally:
    cd hf-projects
    python 06_gradio_app/app.py

Then open: http://localhost:7860
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from transformers import pipeline
import gradio as gr

# ── Config ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

SENTIMENT_MODEL = os.getenv("SENTIMENT_MODEL",
                  "distilbert-base-uncased-finetuned-sst-2-english")
GEN_MODEL       = os.getenv("GENERATION_MODEL", "gpt2")
IMAGE_MODEL     = os.getenv("IMAGE_MODEL",      "google/vit-base-patch16-224")
APP_TITLE       = os.getenv("APP_TITLE",        "My HF Demo")
PORT            = int(os.getenv("GRADIO_PORT",  "7860"))
SHARE           = os.getenv("GRADIO_SHARE", "False").lower() == "true"

# ── Load models ───────────────────────────────────────────────────────────────
print("=" * 60)
print(f"  {APP_TITLE}")
print("=" * 60)
print("  Loading models — this may take a moment on first run...\n")

print(f"  [1/3] Sentiment  : {SENTIMENT_MODEL}")
sentiment_clf = pipeline("sentiment-analysis", model=SENTIMENT_MODEL)

print(f"  [2/3] Generation : {GEN_MODEL}")
text_gen      = pipeline("text-generation",       model=GEN_MODEL)

print(f"  [3/3] Vision     : {IMAGE_MODEL}")
img_clf       = pipeline("image-classification",  model=IMAGE_MODEL)

print("\n  ✓ All models loaded!\n" + "=" * 60)


# ── Tab 1: Sentiment ──────────────────────────────────────────────────────────
def run_sentiment(text: str) -> str:
    if not text.strip():
        return "Please enter some text."
    r     = sentiment_clf(text)[0]
    label = r["label"]
    score = r["score"] * 100
    bar   = "█" * int(score / 5)
    return f"{label}\n{score:.1f}% confidence\n\n{bar}"


# ── Tab 2: Text Generation ────────────────────────────────────────────────────
def run_generation(prompt: str, max_tokens: int, temperature: float) -> str:
    if not prompt.strip():
        return "Please enter a prompt."
    outputs = text_gen(
        prompt,
        max_new_tokens      = max_tokens,
        temperature         = temperature,
        do_sample           = True,
        repetition_penalty  = 1.1,
    )
    return outputs[0]["generated_text"]


# ── Tab 3: Image Classification ───────────────────────────────────────────────
def run_image(image) -> str:
    if image is None:
        return "Please upload an image."
    results = img_clf(image, top_k=5)
    lines   = []
    for i, r in enumerate(results, 1):
        bar   = "█" * int(r["score"] * 40)
        score = r["score"] * 100
        lines.append(f"{i}. {score:5.1f}%  {bar}  {r['label']}")
    return "\n".join(lines)


# ── Build UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(title=APP_TITLE, theme=gr.themes.Soft()) as demo:

    gr.Markdown(f"# 🤗 {APP_TITLE}")
    gr.Markdown(
        "Powered by **Hugging Face Transformers** · "
        "Models configured in `.env`\n\n"
        f"| Sentiment | Generation | Vision |\n"
        f"|-----------|------------|--------|\n"
        f"| `{SENTIMENT_MODEL}` | `{GEN_MODEL}` | `{IMAGE_MODEL}` |"
    )

    # ── Tab 1 ────────────────────────────────────────────────────────────────
    with gr.Tab("💬 Sentiment Analysis"):
        gr.Markdown("Classify text as POSITIVE or NEGATIVE.")
        with gr.Row():
            with gr.Column():
                txt_in = gr.Textbox(
                    label="Input Text",
                    lines=4,
                    placeholder="Type or paste any text here...",
                )
                gr.Examples(
                    examples=[
                        ["Hugging Face is an amazing platform!"],
                        ["This tutorial is confusing and too long."],
                        ["Decent results, nothing special."],
                    ],
                    inputs=txt_in,
                )
                btn1 = gr.Button("Analyse Sentiment", variant="primary")
            with gr.Column():
                txt_out = gr.Textbox(label="Result", lines=5)

        btn1.click(run_sentiment, inputs=txt_in, outputs=txt_out)

    # ── Tab 2 ────────────────────────────────────────────────────────────────
    with gr.Tab("✍️ Text Generation"):
        gr.Markdown("Complete a prompt using a language model.")
        with gr.Row():
            with gr.Column():
                prompt   = gr.Textbox(
                    label="Prompt",
                    lines=3,
                    placeholder="The future of AI is...",
                )
                with gr.Row():
                    max_tok = gr.Slider(20, 300, value=80,  step=10, label="Max New Tokens")
                    temp    = gr.Slider(0.1, 1.5, value=0.9, step=0.1, label="Temperature")
                gr.Examples(
                    examples=[
                        ["The future of artificial intelligence is", 80, 0.9],
                        ["In a world where machines can think,",     100, 1.0],
                        ["Scientists recently discovered that",      60, 0.7],
                    ],
                    inputs=[prompt, max_tok, temp],
                )
                btn2 = gr.Button("Generate Text", variant="primary")
            with gr.Column():
                gen_out = gr.Textbox(label="Generated Text", lines=10)

        btn2.click(run_generation, inputs=[prompt, max_tok, temp], outputs=gen_out)

    # ── Tab 3 ────────────────────────────────────────────────────────────────
    with gr.Tab("🖼️ Image Classification"):
        gr.Markdown("Upload an image to classify it (top 5 predictions).")
        with gr.Row():
            with gr.Column():
                img_in = gr.Image(type="pil", label="Upload Image")
                btn3   = gr.Button("Classify Image", variant="primary")
            with gr.Column():
                img_out = gr.Textbox(label="Top 5 Predictions", lines=8)

        btn3.click(run_image, inputs=img_in, outputs=img_out)

    # ── Footer ────────────────────────────────────────────────────────────────
    gr.Markdown("---\n*Swap any model by editing `.env` — no code changes needed.*")


# ── Launch ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n  Launching on http://localhost:{PORT}")
    print(f"  Public share : {SHARE}")
    print("  Press Ctrl+C to stop.\n")
    demo.launch(server_port=PORT, share=SHARE)
