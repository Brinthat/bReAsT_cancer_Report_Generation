"""
data_loader.py
--------------
Loads the raw BrEaST-Lesions CSV and builds structured text reports
by concatenating clinical columns (Age, Tissue_composition, Shape,
Margin, Echogenicity, Posterior_features, Signs, Symptoms, Interpretation).

Usage
-----
    from src.preprocessing.data_loader import load_and_build_reports
    df = load_and_build_reports()
"""

import pandas as pd
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.config import (
    REPORTS_CSV,
    DATASET_CSV,
    RAW_IMAGE_DIR,
)


def load_and_build_reports(
    reports_csv: str = REPORTS_CSV,
    image_dir: str = RAW_IMAGE_DIR,
    save_path: str = DATASET_CSV,
) -> pd.DataFrame:
    """
    Load the raw reports CSV, build natural-language input_text from
    clinical columns, and return a tidy DataFrame with columns:
        CaseID | input_text | target_text | image_path

    Parameters
    ----------
    reports_csv : str   Path to the original reports.csv
    image_dir   : str   Directory containing PNG images
    save_path   : str   Where to save the preprocessed dataset CSV

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(reports_csv)

    # Drop columns not needed for report generation
    drop_cols = ["Mask_tumor_filename", "Mask_other_filename", "Pixel_size"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Build absolute image path
    df["image_path"] = image_dir + os.sep + df["Image_filename"]

    # Build natural-language input & target text
    df["input_text"], df["target_text"] = zip(
        *df.apply(_convert_row_to_text, axis=1)
    )

    # Keep only required columns
    df = df[["CaseID", "input_text", "target_text", "image_path"]]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"[data_loader] Saved preprocessed dataset → {save_path}")
    print(f"[data_loader] Total records : {len(df)}")
    print(f"[data_loader] Unique reports: {df['input_text'].nunique()}")
    return df


def _convert_row_to_text(row) -> tuple:
    """
    Convert one CSV row into (input_text, target_text).

    input_text  – structured clinical description (age, tissue, shape …)
    target_text – the radiologist's interpretation (ground truth)
    """
    input_text = (
        f"Patient is {row['Age']} years old with tissue composition of "
        f"{row['Tissue_composition']}. "
        f"Symptoms include {row['Signs']} and {row['Symptoms']}. "
        f"Shape of the lesion is {row['Shape']}, with margin {row['Margin']}, "
        f"echogenicity {row['Echogenicity']}, and posterior features "
        f"{row['Posterior_features']}."
    )
    target_text = row["Interpretation"]
    return input_text, target_text


if __name__ == "__main__":
    df = load_and_build_reports()
    print(df.head(2))
