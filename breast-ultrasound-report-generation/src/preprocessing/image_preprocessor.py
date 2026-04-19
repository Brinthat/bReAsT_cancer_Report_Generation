"""
image_preprocessor.py
---------------------
TensorFlow image loading, normalisation, and tf.data pipeline.

Usage
-----
    from src.preprocessing.image_preprocessor import build_tf_dataloader
    data_loader = build_tf_dataloader(df)
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.config import IMAGE_SIZE, DATASET_CSV, BATCH_SIZE


# ── Single Image Preprocessing ────────────────────────────────────────────────

def preprocess_image(image_path: str) -> np.ndarray:
    """
    Load one PNG, resize to IMAGE_SIZE, normalize to [0,1],
    then apply per-image standardisation.

    Returns
    -------
    np.ndarray  shape (H, W, 3)  dtype float32
    """
    from tensorflow.keras.preprocessing.image import load_img, img_to_array

    image = load_img(image_path, target_size=IMAGE_SIZE)
    image = img_to_array(image)
    image /= 255.0
    image = tf.image.per_image_standardization(image).numpy()
    return image


# ── Dataset Class ─────────────────────────────────────────────────────────────

class ImageDataset:
    """Iterable wrapper around a DataFrame of image paths."""

    def __init__(self, dataframe: pd.DataFrame, base_dir: str = ""):
        self.dataframe = dataframe.reset_index(drop=True)
        self.base_dir  = base_dir

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> np.ndarray:
        img_path = self.dataframe.iloc[idx]["image_path"]
        if self.base_dir and not os.path.isabs(img_path):
            img_path = os.path.join(self.base_dir, img_path)
        return preprocess_image(img_path)


# ── tf.data Pipeline ──────────────────────────────────────────────────────────

def build_tf_dataloader(
    df: pd.DataFrame = None,
    dataset_csv: str  = DATASET_CSV,
    base_dir: str     = "",
    batch_size: int   = BATCH_SIZE,
) -> tf.data.Dataset:
    """
    Build an efficient tf.data pipeline from a DataFrame of image paths.

    Parameters
    ----------
    df          : pd.DataFrame (optional)  If None, loads from dataset_csv
    dataset_csv : str                      Fallback CSV
    base_dir    : str                      Prefix for relative image paths
    batch_size  : int

    Returns
    -------
    tf.data.Dataset   yields batches of shape (batch, H, W, 3)
    """
    if df is None:
        df = pd.read_csv(dataset_csv)

    image_ds = ImageDataset(df, base_dir=base_dir)

    def generator():
        for i in range(len(image_ds)):
            yield image_ds[i]

    data_loader = tf.data.Dataset.from_generator(
        generator,
        output_signature=tf.TensorSpec(
            shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=tf.float32
        ),
    )
    data_loader = data_loader.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    print(f"[image_preprocessor] Pipeline ready – {len(df)} images, batch={batch_size}")
    return data_loader


# ── Visualisation Helper ──────────────────────────────────────────────────────

def display_sample_images(data_loader: tf.data.Dataset, n: int = 4):
    """Display the first n images from a tf.data batch."""
    batch = next(iter(data_loader))
    plt.figure(figsize=(12, 4))
    for i in range(min(n, len(batch))):
        ax = plt.subplot(1, n, i + 1)
        plt.imshow(batch[i].numpy())
        plt.axis("off")
    plt.suptitle("Sample Preprocessed Images")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df  = pd.read_csv(DATASET_CSV)
    dl  = build_tf_dataloader(df)
    display_sample_images(dl)
