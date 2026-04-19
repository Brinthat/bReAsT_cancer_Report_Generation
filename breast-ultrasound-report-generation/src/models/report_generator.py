"""
report_generator.py
-------------------
Fine-tune distilGPT-2 on radiology reports and generate new reports
given a prompt derived from the most similar image's clinical data.

Usage
-----
    # Training
    from src.models.report_generator import train_model
    model, tokenizer = train_model(dataloader)

    # Inference
    from src.models.report_generator import generate_report
    text = generate_report(prompt_text, model_path="saved_models/gpt2_finetuned")
"""

import os
import sys
import torch
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.config import (
    BASE_LM_NAME,
    TOKENIZER_NAME,
    MODEL_SAVE_DIR,
    LEARNING_RATE,
    NUM_EPOCHS,
    MAX_TOKEN_LEN,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(
    dataloader,
    epochs: int       = NUM_EPOCHS,
    lr: float         = LEARNING_RATE,
    save_dir: str     = MODEL_SAVE_DIR,
    plot_loss: bool   = True,
):
    """
    Fine-tune distilGPT-2 on the radiology report dataloader.

    Parameters
    ----------
    dataloader : torch DataLoader  from text_preprocessor.build_dataloader
    epochs     : int
    lr         : float
    save_dir   : str  where to save the fine-tuned model
    plot_loss  : bool plot and save the training loss curve

    Returns
    -------
    (model, tokenizer)
    """
    model = GPT2LMHeadModel.from_pretrained(BASE_LM_NAME).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=lr)

    avg_losses = []
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            if loss is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        avg = total_loss / len(dataloader)
        avg_losses.append(avg)
        print(f"  Epoch {epoch + 1} — Average Loss: {avg:.4f}")

    # Save model
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    print(f"[report_generator] Model saved → {save_dir}")

    if plot_loss:
        _plot_loss_curve(avg_losses, save_dir)

    tokenizer = GPT2Tokenizer.from_pretrained(TOKENIZER_NAME)
    return model, tokenizer


def _plot_loss_curve(losses: list, save_dir: str):
    epochs = range(1, len(losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, marker="o", linestyle="-", color="b", label="Training Loss")
    plt.title("Fine-Tuning: Epoch vs Loss", fontsize=16)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.xticks(list(epochs))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(save_dir, "training_loss.png")
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"[report_generator] Loss curve saved → {out_path}")


# ── Inference ─────────────────────────────────────────────────────────────────

def generate_report(
    prompt_text: str,
    model_path: str    = MODEL_SAVE_DIR,
    max_length: int    = MAX_TOKEN_LEN,
    temperature: float = 0.5,
    top_k: int         = 50,
    top_p: float       = 0.95,
    rep_penalty: float = 1.2,
) -> str:
    """
    Generate a radiology report from a text prompt.

    Parameters
    ----------
    prompt_text  : str  Clinical prompt (typically from the most similar image)
    model_path   : str  Path to the fine-tuned GPT-2 model directory
    max_length   : int
    temperature  : float
    top_k        : int
    top_p        : float
    rep_penalty  : float

    Returns
    -------
    str  Generated report text
    """
    tokenizer = GPT2Tokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(model_path).to(DEVICE)
    model.eval()

    input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=rep_penalty,
            do_sample=True,
            num_return_sequences=1,
        )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text


# ── Perplexity Evaluation ─────────────────────────────────────────────────────

def compute_perplexity(model, dataloader) -> float:
    """
    Compute perplexity of the fine-tuned model on a validation DataLoader.

    Returns
    -------
    float  perplexity score (lower is better)
    """
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(DEVICE)

            outputs     = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            total_loss += outputs.loss.item()

    avg_loss   = total_loss / len(dataloader)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    print(f"[report_generator] Perplexity: {perplexity:.2f}")
    return perplexity


if __name__ == "__main__":
    # Quick generation demo (requires a trained model at MODEL_SAVE_DIR)
    sample_prompt = "Patient is 45 years old with tissue composition of heterogeneous."
    report = generate_report(sample_prompt)
    print("\nGenerated Report:\n", report)
