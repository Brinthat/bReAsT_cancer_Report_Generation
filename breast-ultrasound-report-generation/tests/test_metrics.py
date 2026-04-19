"""
test_metrics.py
---------------
Unit tests for the evaluation metrics module.

Run with:
    pytest tests/test_metrics.py -v
"""

import os
import sys
import json
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.evaluation.metrics import compute_bleu, compute_rouge, evaluate_reports


# ── compute_bleu Tests ────────────────────────────────────────────────────────

class TestComputeBleu:

    def test_perfect_match_is_one(self):
        text = "suspicion of malignancy with irregular shape"
        score = compute_bleu(text, text)
        assert score == pytest.approx(1.0, abs=1e-3)

    def test_zero_for_completely_different(self):
        ref = "suspicion of malignancy"
        hyp = "patient shows normal tissue composition"
        score = compute_bleu(ref, hyp)
        assert score == pytest.approx(0.0, abs=1e-3)

    def test_partial_overlap_is_between_zero_and_one(self):
        ref = "suspicion of malignancy with irregular shape"
        hyp = "suspicion of malignancy present"
        score = compute_bleu(ref, hyp)
        assert 0.0 <= score <= 1.0

    def test_returns_float(self):
        score = compute_bleu("hello world", "hello there")
        assert isinstance(score, float)


# ── compute_rouge Tests ───────────────────────────────────────────────────────

class TestComputeRouge:

    def test_returns_dict_with_correct_keys(self):
        result = compute_rouge("hello world", "hello there")
        assert set(result.keys()) == {"rouge-1", "rouge-2", "rouge-l"}

    def test_perfect_match_rouge1_is_one(self):
        text = "suspicion of malignancy"
        result = compute_rouge(text, text)
        assert result["rouge-1"] == pytest.approx(1.0, abs=1e-3)

    def test_scores_between_zero_and_one(self):
        ref = "the tissue composition is heterogeneous predominantly fibroglandular"
        hyp = "the tissue composition is homogeneous fibroglandular"
        result = compute_rouge(ref, hyp)
        for v in result.values():
            assert 0.0 <= v <= 1.0


# ── evaluate_reports Tests ────────────────────────────────────────────────────

class TestEvaluateReports:

    @pytest.fixture
    def sample_pairs(self):
        generated = [
            "suspicion of malignancy with irregular shape",
            "fibroadenoma with oval shape",
        ]
        reference = [
            "suspicion of malignancy with irregular shape",
            "fibroadenoma with oval shape and homogeneous tissue",
        ]
        return generated, reference

    def test_returns_dict(self, sample_pairs):
        gen, ref = sample_pairs
        result = evaluate_reports(gen, ref, save_results=False)
        assert isinstance(result, dict)

    def test_contains_expected_keys(self, sample_pairs):
        gen, ref = sample_pairs
        result = evaluate_reports(gen, ref, save_results=False)
        expected_keys = {
            "num_samples", "bleu_mean",
            "rouge_1_mean", "rouge_2_mean", "rouge_l_mean",
            "bertscore_precision", "bertscore_recall", "bertscore_f1",
        }
        assert expected_keys.issubset(result.keys())

    def test_num_samples_correct(self, sample_pairs):
        gen, ref = sample_pairs
        result = evaluate_reports(gen, ref, save_results=False)
        assert result["num_samples"] == 2

    def test_all_scores_between_zero_and_one(self, sample_pairs):
        gen, ref = sample_pairs
        result = evaluate_reports(gen, ref, save_results=False)
        score_keys = [
            "bleu_mean", "rouge_1_mean", "rouge_2_mean", "rouge_l_mean",
            "bertscore_precision", "bertscore_recall", "bertscore_f1",
        ]
        for k in score_keys:
            assert 0.0 <= result[k] <= 1.0, f"{k} = {result[k]} out of [0, 1]"

    def test_saves_json_file(self, sample_pairs, tmp_path):
        gen, ref = sample_pairs
        evaluate_reports(gen, ref, save_results=True, results_dir=str(tmp_path))
        out_file = tmp_path / "evaluation_results.json"
        assert out_file.exists()
        with open(out_file) as f:
            data = json.load(f)
        assert "bleu_mean" in data

    def test_raises_on_length_mismatch(self):
        with pytest.raises(AssertionError):
            evaluate_reports(["one report"], ["ref1", "ref2"], save_results=False)
