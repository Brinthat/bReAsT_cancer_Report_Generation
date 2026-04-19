"""
main.py
-------
End-to-end pipeline for Breast Ultrasound Report Generation.

Steps
-----
1. Load & preprocess CSV data           (data_loader)
2. Exploratory Data Analysis            (eda)
3. Build image tf.data pipeline         (image_preprocessor)
4. Extract ViT features                 (feature_extractor)
5. Tokenise text & build DataLoader     (text_preprocessor)
6. Fine-tune distilGPT-2                (report_generator)
7. Inference on a query image           (feature_extractor + report_generator)
8. Evaluate generated reports           (metrics)

Usage
-----
    python main.py                        # full pipeline
    python main.py --step train           # training only
    python main.py --step infer           # inference only
    python main.py --query case001.png    # infer on a specific image
"""

import os
import argparse
import pandas as pd

from configs.config import (
    DATASET_CSV,
    FEATURES_NPY,
    MODEL_SAVE_DIR,
    RAW_IMAGE_DIR,
    RESULTS_DIR,
)
from src.preprocessing.data_loader       import load_and_build_reports
from src.preprocessing.eda               import run_eda
from src.preprocessing.image_preprocessor import build_tf_dataloader
from src.preprocessing.text_preprocessor  import build_dataloader
from src.models.feature_extractor        import (
    load_vit_model,
    extract_and_save_features,
    find_similar_image,
)
from src.models.report_generator         import (
    train_model,
    generate_report,
    compute_perplexity,
)
from src.evaluation.metrics              import evaluate_reports


def run_full_pipeline(query_image: str = None):
    """Execute the complete training + evaluation pipeline."""

    # ── Step 1: Data preprocessing ────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  STEP 1 — Data Preprocessing")
    print("═" * 60)
    df = load_and_build_reports()

    # ── Step 2: EDA ───────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  STEP 2 — Exploratory Data Analysis")
    print("═" * 60)
    run_eda(df)

    # ── Step 3: Image Pipeline ────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  STEP 3 — Image tf.data Pipeline")
    print("═" * 60)
    tf_loader = build_tf_dataloader(df)

    # ── Step 4: ViT Feature Extraction ────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  STEP 4 — ViT Feature Extraction")
    print("═" * 60)
    vit_model = load_vit_model()
    extract_and_save_features(tf_loader, model=vit_model)

    # ── Step 5: Text Tokenisation ─────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  STEP 5 — Text Tokenisation & DataLoader")
    print("═" * 60)
    torch_loader = build_dataloader(df)

    # ── Step 6: Model Training ────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  STEP 6 — Fine-Tuning distilGPT-2")
    print("═" * 60)
    model, tokenizer = train_model(torch_loader)

    # ── Step 7: Inference ─────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  STEP 7 — Report Generation (Inference)")
    print("═" * 60)

    if query_image is None:
        # Default: use the first image in the dataset
        query_image = df.iloc[0]["image_path"]

    idx, sim = find_similar_image(query_image, model=vit_model)
    prompt   = df.iloc[idx]["target_text"]

    generated = generate_report(prompt)
    reference = df.iloc[idx]["input_text"] + " " + df.iloc[idx]["target_text"]

    print(f"\n  Reference:\n  {reference}\n")
    print(f"  Generated:\n  {generated}\n")

    # ── Step 8: Evaluation ────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  STEP 8 — Evaluation Metrics")
    print("═" * 60)
    evaluate_reports([generated], [reference])

    perplexity = compute_perplexity(model, torch_loader)
    print(f"  Final Perplexity: {perplexity:.2f}")

    print("\n✅  Pipeline complete.  Results saved to:", RESULTS_DIR)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Breast Ultrasound Report Generation Pipeline"
    )
    parser.add_argument(
        "--step",
        choices=["full", "train", "infer", "eda"],
        default="full",
        help="Which step to run (default: full pipeline)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Path to a query image for inference",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.step == "eda":
        df = pd.read_csv(DATASET_CSV)
        run_eda(df)

    elif args.step == "train":
        df          = pd.read_csv(DATASET_CSV)
        tf_loader   = build_tf_dataloader(df)
        vit_model   = load_vit_model()
        extract_and_save_features(tf_loader, model=vit_model)
        torch_loader = build_dataloader(df)
        train_model(torch_loader)

    elif args.step == "infer":
        df      = pd.read_csv(DATASET_CSV)
        query   = args.query or df.iloc[0]["image_path"]
        idx, _  = find_similar_image(query)
        prompt  = df.iloc[idx]["target_text"]
        report  = generate_report(prompt)
        print("\nGenerated Report:\n", report)

    else:
        run_full_pipeline(query_image=args.query)
