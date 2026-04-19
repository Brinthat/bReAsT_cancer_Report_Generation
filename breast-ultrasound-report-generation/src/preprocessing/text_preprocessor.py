"""
text_preprocessor.py
---------------------
Text cleaning, tokenisation, and GPT-2 dataset preparation.

Steps
-----
1. Clean raw text (lowercase, strip special tokens)
2. Combine input + target into a single prompt string
3. Tokenise with distilGPT-2 tokeniser
4. Wrap in a PyTorch Dataset / DataLoader

Usage
-----
    from src.preprocessing.text_preprocessor import build_dataloader
    dataloader = build_dataloader(df)
"""

import os
import sys
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.config import (
    TOKENIZER_NAME,
    MAX_TOKEN_LEN,
    BATCH_SIZE,
    PROCESSED_TXT,
    DATASET_CSV,
)


# ── Text Cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Lowercase, strip artefact tokens, normalise whitespace."""
    text = str(text)
    for token in ["nan", "xxxx", "x-xxxx"]:
        text = text.replace(token, "")
    text = text.lower().replace("\n", " ").strip()
    return text


def build_combined_reports(df: pd.DataFrame, save_path: str = PROCESSED_TXT) -> list:
    """
    Combine index + input_text + target_text into a single line per record
    and save to a plain-text file for tokenisation.

    Format:  <idx> [IDX] <input_text> [SEP] <target_text>
    """
    df = df.copy()
    df["input_text"]  = df["input_text"].astype(str).apply(clean_text)
    df["target_text"] = df["target_text"].astype(str).apply(clean_text)
    df["index"] = df.index

    df["combined_report"] = (
        df["index"].astype(str)
        + " [IDX] "
        + df["input_text"]
        + " [SEP] "
        + df["target_text"]
    )

    reports = df["combined_report"].tolist()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        for report in reports:
            f.write(report + "\n")

    print(f"[text_preprocessor] Saved {len(reports)} reports → {save_path}")
    return reports


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class ReportsDataset(Dataset):
    """Wraps the tokenised encodings for GPT-2 fine-tuning."""

    def __init__(self, encodings: dict):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int) -> dict:
        return {key: val[idx] for key, val in self.encodings.items()}


# ── DataLoader Factory ────────────────────────────────────────────────────────

def build_dataloader(
    df: pd.DataFrame = None,
    dataset_csv: str = DATASET_CSV,
    processed_txt: str = PROCESSED_TXT,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = True,
) -> DataLoader:
    """
    Full pipeline: clean → combine → tokenise → DataLoader.

    Parameters
    ----------
    df          : pd.DataFrame (optional)  If None, loads from dataset_csv
    dataset_csv : str                      Fallback CSV path
    batch_size  : int
    shuffle     : bool

    Returns
    -------
    torch.utils.data.DataLoader
    """
    if df is None:
        df = pd.read_csv(dataset_csv)

    reports = build_combined_reports(df, save_path=processed_txt)

    tokenizer = GPT2Tokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(
        reports,
        return_tensors="pt",
        max_length=MAX_TOKEN_LEN,
        truncation=True,
        padding="max_length",
    )
    inputs["labels"] = inputs["input_ids"].clone()

    dataset    = ReportsDataset(inputs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    total_tokens = sum(len(ids) for ids in inputs["input_ids"])
    print(f"[text_preprocessor] Total tokens : {total_tokens:,}")
    print(f"[text_preprocessor] Batches/epoch: {len(dataloader)}")
    return dataloader


if __name__ == "__main__":
    df = pd.read_csv(DATASET_CSV)
    dl = build_dataloader(df)
    batch = next(iter(dl))
    print("Sample batch keys:", list(batch.keys()))
    print("input_ids shape :", batch["input_ids"].shape)
