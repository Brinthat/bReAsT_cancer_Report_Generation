"""
feature_extractor.py
--------------------
Extract image features using a pre-trained ViT (Vision Transformer)
from the `timm` library, then find the most visually similar image
to a query image via cosine similarity.

Usage
-----
    from src.models.feature_extractor import extract_and_save_features, find_similar_image
    extract_and_save_features(data_loader)
    idx, score = find_similar_image("path/to/query.png")
"""

import os
import sys
import numpy as np
import torch
import timm
from torchvision import transforms
from PIL import Image
from io import BytesIO
from numpy.linalg import norm
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.config import (
    VIT_MODEL_NAME,
    VIT_FEATURE_DIM,
    VIT_SEQ_LEN,
    FEATURES_NPY,
    IMAGE_SIZE,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
)


# ── Device Setup ──────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Image Transforms ──────────────────────────────────────────────────────────

PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE[0]),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
])


# ── Model Loader ──────────────────────────────────────────────────────────────

def load_vit_model(model_name: str = VIT_MODEL_NAME) -> torch.nn.Module:
    """Load and return a pre-trained ViT model in eval mode."""
    model = timm.create_model(model_name, pretrained=True)
    model = model.to(DEVICE)
    model.eval()
    print(f"[feature_extractor] Loaded {model_name} on {DEVICE}")
    return model


# ── Single Image Feature Extraction ──────────────────────────────────────────

def load_image_bytes(image_path: str) -> bytes:
    with open(image_path, "rb") as f:
        return f.read()


def extract_single_image_features(
    image_path: str, model: torch.nn.Module
) -> np.ndarray:
    """
    Extract ViT features from a single image.

    Returns
    -------
    np.ndarray  shape (VIT_SEQ_LEN, VIT_FEATURE_DIM)  e.g. (197, 768)
    """
    image_bytes = load_image_bytes(image_path)
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    tensor = PREPROCESS(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        features = model.forward_features(tensor)

    return features.cpu().numpy().reshape(VIT_SEQ_LEN, VIT_FEATURE_DIM)


# ── Batch Feature Extraction ──────────────────────────────────────────────────

def extract_and_save_features(
    data_loader,           # tf.data.Dataset  (yields batches of shape B,H,W,3)
    model: torch.nn.Module = None,
    save_path: str = FEATURES_NPY,
) -> np.ndarray:
    """
    Run the full dataset through ViT and save features as a .npy file.

    Parameters
    ----------
    data_loader : tf.data.Dataset
    model       : ViT model (loaded if None)
    save_path   : str  path to save the .npy file

    Returns
    -------
    np.ndarray  shape (N, VIT_SEQ_LEN, VIT_FEATURE_DIM)
    """
    if model is None:
        model = load_vit_model()

    features_list = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Extracting ViT features"):
            # Convert TF tensor → PyTorch (B,C,H,W)
            inputs = torch.from_numpy(batch.numpy()).permute(0, 3, 1, 2).to(DEVICE)
            outputs = model.forward_features(inputs)
            features_list.append(outputs.cpu().detach().numpy())

    features = np.concatenate(features_list, axis=0)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, features)
    print(f"[feature_extractor] Saved features {features.shape} → {save_path}")
    return features


# ── Cosine Similarity Search ──────────────────────────────────────────────────

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between vector a and each row of matrix b."""
    return np.dot(a, b.T) / (norm(a) * norm(b, axis=1) + 1e-10)


def find_similar_image(
    query_image_path: str,
    features_path: str = FEATURES_NPY,
    model: torch.nn.Module = None,
) -> tuple:
    """
    Find the most visually similar image to a query in the pre-extracted dataset.

    Parameters
    ----------
    query_image_path : str  path to the PNG you want to query
    features_path    : str  path to the saved .npy feature matrix
    model            : ViT model (loaded if None)

    Returns
    -------
    (index: int, similarity_score: float)
    """
    if model is None:
        model = load_vit_model()

    # Query features: (VIT_SEQ_LEN * VIT_FEATURE_DIM,)
    query_feats = extract_single_image_features(query_image_path, model).reshape(-1)

    # Dataset features: (N, VIT_SEQ_LEN * VIT_FEATURE_DIM)
    dataset_feats = np.load(features_path)
    dataset_feats_flat = dataset_feats.reshape(dataset_feats.shape[0], -1)

    similarities   = _cosine_similarity(query_feats, dataset_feats_flat)
    best_idx       = int(np.argmax(similarities))
    best_score     = float(similarities[best_idx])

    print(f"[feature_extractor] Most similar index={best_idx}, similarity={best_score:.4f}")
    return best_idx, best_score


if __name__ == "__main__":
    # Quick smoke-test
    model = load_vit_model()
    print("ViT model loaded successfully.")
