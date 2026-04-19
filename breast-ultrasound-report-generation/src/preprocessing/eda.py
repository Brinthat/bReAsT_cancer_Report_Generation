"""
eda.py
------
Exploratory Data Analysis for the breast ultrasound text reports.

Plots generated
---------------
1. Word cloud of input reports
2. Report length distribution histogram
3. Top-10 unigrams bar chart
4. Top-10 bigrams bar chart
5. Top-10 trigrams bar chart

Usage
-----
    from src.preprocessing.eda import run_eda
    run_eda(df)
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import spacy
import textstat
from collections import Counter

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.config import FIGURES_DIR, DATASET_CSV


def run_eda(df: pd.DataFrame = None, save_figures: bool = True):
    """
    Run the complete EDA pipeline on the report DataFrame.

    Parameters
    ----------
    df           : pd.DataFrame (optional) — loaded from DATASET_CSV if None
    save_figures : bool  — save PNG plots to FIGURES_DIR
    """
    if df is None:
        df = pd.read_csv(DATASET_CSV)

    os.makedirs(FIGURES_DIR, exist_ok=True)

    print(f"[EDA] Total records      : {len(df)}")
    print(f"[EDA] Unique input texts : {df['input_text'].nunique()}")
    print(f"[EDA] Avg report length  : {df['input_text'].str.len().mean():.1f} chars")

    _word_cloud(df, save=save_figures)
    _report_length_distribution(df, save=save_figures)
    _top_ngrams(df, n=1, save=save_figures)
    _top_ngrams(df, n=2, save=save_figures)
    _top_ngrams(df, n=3, save=save_figures)
    _lda_topics(df)
    _readability(df)


# ── Plot Helpers ──────────────────────────────────────────────────────────────

def _word_cloud(df: pd.DataFrame, save: bool = True):
    text = " ".join(df["input_text"].dropna())
    wc = WordCloud(width=800, height=800, background_color="white").generate(text)
    plt.figure(figsize=(8, 8))
    plt.imshow(wc)
    plt.axis("off")
    plt.tight_layout()
    _save_or_show("wordcloud.png", save)


def _report_length_distribution(df: pd.DataFrame, save: bool = True):
    df = df.copy()
    df["report_length"] = df["input_text"].str.len()
    mean_val = df["report_length"].mean()

    plt.figure(figsize=(10, 6))
    sns.histplot(df["report_length"], kde=True, color="skyblue")
    plt.axvline(mean_val, color="red", linestyle="--")
    plt.text(mean_val + 5, plt.ylim()[1] * 0.8, f"Mean: {mean_val:.1f}", color="red")
    plt.title("Distribution of Ultrasound Report Lengths", fontsize=14, fontweight="bold")
    plt.xlabel("Length (characters)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True, linestyle="--", linewidth=0.5)
    sns.despine()
    plt.tight_layout()
    _save_or_show("report_length_distribution.png", save)


def _top_ngrams(df: pd.DataFrame, n: int = 1, save: bool = True):
    label_map = {1: "Unigram", 2: "Bigram", 3: "Trigram"}
    label = label_map.get(n, f"{n}-gram")
    corpus = df["input_text"].dropna()

    vec = CountVectorizer(ngram_range=(n, n), stop_words="english").fit(corpus)
    bow = vec.transform(corpus)
    sum_w = bow.sum(axis=0)
    freq = sorted(
        [(w, sum_w[0, i]) for w, i in vec.vocabulary_.items()],
        key=lambda x: x[1],
        reverse=True,
    )[:10]

    top_df = pd.DataFrame(freq, columns=[label, "Frequency"])

    plt.figure(figsize=(10, 6))
    orient = "v" if n == 1 else "h"
    x, y = (label, "Frequency") if orient == "v" else ("Frequency", label)
    sns.barplot(x=x, y=y, data=top_df, palette="Blues_d", orient=orient)
    plt.title(f"Top 10 {label}s in Ultrasound Reports", fontsize=14, fontweight="bold")
    plt.grid(True, linestyle="--", linewidth=0.5)
    sns.despine()
    plt.tight_layout()
    _save_or_show(f"top_{n}grams.png", save)


def _lda_topics(df: pd.DataFrame, n_topics: int = 5, top_words: int = 10):
    print("\n[EDA] LDA Topics:")
    corpus = df["input_text"].dropna()
    vec = CountVectorizer(stop_words="english").fit(corpus)
    bow = vec.transform(corpus)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(bow)
    feat_names = vec.get_feature_names_out()
    for i, topic in enumerate(lda.components_):
        words = " | ".join([feat_names[j] for j in topic.argsort()[: -top_words - 1 : -1]])
        print(f"  Topic {i}: {words}")


def _readability(df: pd.DataFrame):
    sample = df["input_text"].dropna().iloc[0]
    print("\n[EDA] Readability Scores (sample report):")
    print(f"  Flesch Reading Ease : {textstat.flesch_reading_ease(sample):.1f}")
    print(f"  Gunning Fog Index   : {textstat.gunning_fog(sample):.1f}")


def _save_or_show(filename: str, save: bool):
    if save:
        path = os.path.join(FIGURES_DIR, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[EDA] Saved → {path}")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    run_eda()
