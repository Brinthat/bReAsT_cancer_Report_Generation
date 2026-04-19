"""
config.py
---------
Central configuration file for the Breast Ultrasound Report Generation project.
Update paths here before running any module.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR        = os.path.join(BASE_DIR, "data")
RAW_IMAGE_DIR   = os.path.join(DATA_DIR, "images")          # BrEaST-Lesions PNG files
REPORTS_CSV     = os.path.join(DATA_DIR, "reports.csv")     # Original CSV from dataset
DATASET_CSV     = os.path.join(DATA_DIR, "dataset.csv")     # Preprocessed CSV
PROCESSED_TXT   = os.path.join(DATA_DIR, "processed_reports.txt")
FEATURES_NPY    = os.path.join(DATA_DIR, "features.npy")    # ViT-extracted features

RESULTS_DIR     = os.path.join(BASE_DIR, "results")
FIGURES_DIR     = os.path.join(RESULTS_DIR, "figures")
METRICS_DIR     = os.path.join(RESULTS_DIR, "metrics")
SAMPLE_DIR      = os.path.join(RESULTS_DIR, "sample_reports")

MODEL_SAVE_DIR  = os.path.join(BASE_DIR, "saved_models", "gpt2_finetuned")

# ─── Image Preprocessing ──────────────────────────────────────────────────────
IMAGE_SIZE      = (224, 224)
NORMALIZE_MEAN  = [0.485, 0.456, 0.406]
NORMALIZE_STD   = [0.229, 0.224, 0.225]

# ─── ViT Feature Extraction ───────────────────────────────────────────────────
VIT_MODEL_NAME  = "vit_base_patch16_224"
VIT_FEATURE_DIM = 768
VIT_SEQ_LEN     = 197          # 196 patches + 1 CLS token

# ─── GPT-2 / Language Model ───────────────────────────────────────────────────
TOKENIZER_NAME  = "distilgpt2"
BASE_LM_NAME    = "distilgpt2"
MAX_TOKEN_LEN   = 218

# ─── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE      = 8
LEARNING_RATE   = 5e-5
NUM_EPOCHS      = 10
TRAIN_RATIO     = 0.7
VAL_RATIO       = 0.1
TEST_RATIO      = 0.2
RANDOM_SEED     = 42

# ─── Evaluation ───────────────────────────────────────────────────────────────
BLEU_N_GRAMS    = [1, 2, 3, 4]

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL       = "INFO"
