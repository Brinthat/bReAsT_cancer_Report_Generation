"""
metrics.py
----------
Evaluation metrics for generated radiology reports:
    - BLEU  (1-gram)
    - ROUGE-L (F1)
    - BERTScore (Precision, Recall, F1)

Usage
-----
    from src.evaluation.metrics import evaluate_reports
    results = evaluate_reports(generated_reports, reference_reports)
"""

import os
import sys
import json
from typing import List, Dict

import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from rouge import Rouge
from bert_score import score as bert_score

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.config import METRICS_DIR

# Ensure NLTK punkt tokeniser is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)


# ── Individual Metrics ────────────────────────────────────────────────────────

def compute_bleu(reference: str, hypothesis: str) -> float:
    """Sentence-level BLEU (unigram) score."""
    ref_tokens  = word_tokenize(reference.lower())
    hyp_tokens  = word_tokenize(hypothesis.lower())
    return sentence_bleu([ref_tokens], hyp_tokens)


def compute_rouge(reference: str, hypothesis: str) -> Dict[str, float]:
    """
    ROUGE-1, ROUGE-2, ROUGE-L F1 scores.

    Returns
    -------
    dict  {'rouge-1': f1, 'rouge-2': f1, 'rouge-l': f1}
    """
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference)[0]
    return {
        "rouge-1": scores["rouge-1"]["f"],
        "rouge-2": scores["rouge-2"]["f"],
        "rouge-l": scores["rouge-l"]["f"],
    }


def compute_bertscore(
    references: List[str], hypotheses: List[str], lang: str = "en"
) -> Dict[str, float]:
    """
    BERTScore over a list of (reference, hypothesis) pairs.

    Returns
    -------
    dict  {'precision': float, 'recall': float, 'f1': float}
    """
    P, R, F1 = bert_score(hypotheses, references, lang=lang, verbose=False)
    return {
        "precision": P.mean().item(),
        "recall":    R.mean().item(),
        "f1":        F1.mean().item(),
    }


# ── Aggregate Evaluation ──────────────────────────────────────────────────────

def evaluate_reports(
    generated_reports:  List[str],
    reference_reports:  List[str],
    save_results: bool  = True,
    results_dir: str    = METRICS_DIR,
) -> Dict[str, object]:
    """
    Compute BLEU, ROUGE, and BERTScore over a list of report pairs
    and optionally save results to JSON.

    Parameters
    ----------
    generated_reports : List[str]
    reference_reports : List[str]
    save_results      : bool
    results_dir       : str

    Returns
    -------
    dict with keys: bleu_scores, rouge_scores, bertscore
    """
    assert len(generated_reports) == len(reference_reports), (
        "generated and reference lists must have the same length"
    )

    bleu_list  = []
    rouge_list = []

    for gen, ref in zip(generated_reports, reference_reports):
        bleu_list.append(compute_bleu(ref, gen))
        rouge_list.append(compute_rouge(ref, gen))

    bert = compute_bertscore(reference_reports, generated_reports)

    results = {
        "num_samples":   len(generated_reports),
        "bleu_mean":     sum(bleu_list) / len(bleu_list),
        "rouge_1_mean":  sum(r["rouge-1"] for r in rouge_list) / len(rouge_list),
        "rouge_2_mean":  sum(r["rouge-2"] for r in rouge_list) / len(rouge_list),
        "rouge_l_mean":  sum(r["rouge-l"] for r in rouge_list) / len(rouge_list),
        "bertscore_precision": bert["precision"],
        "bertscore_recall":    bert["recall"],
        "bertscore_f1":        bert["f1"],
    }

    print("\n══ Evaluation Results ══════════════════")
    for k, v in results.items():
        print(f"  {k:<28}: {v}")
    print("════════════════════════════════════════\n")

    if save_results:
        os.makedirs(results_dir, exist_ok=True)
        out_path = os.path.join(results_dir, "evaluation_results.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[metrics] Results saved → {out_path}")

    return results


if __name__ == "__main__":
    # Smoke test
    gen = ["Suspicion of malignancy with irregular shape."]
    ref = ["Suspicion of malignancy. Tissue composition is heterogeneous."]
    evaluate_reports(gen, ref, save_results=False)
