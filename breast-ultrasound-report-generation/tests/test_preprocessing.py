"""
test_preprocessing.py
---------------------
Unit tests for data_loader and text_preprocessor modules.

Run with:
    pytest tests/test_preprocessing.py -v
"""

import os
import sys
import pytest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocessing.text_preprocessor import clean_text, build_combined_reports


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Minimal DataFrame mimicking the real dataset."""
    return pd.DataFrame({
        "CaseID":       [1, 2, 3],
        "input_text":   [
            "Patient is 45 years old with heterogeneous tissue.",
            "Patient is 60 years old with homogeneous: fat tissue.",
            "nan xxxx x-xxxx",
        ],
        "target_text":  [
            "Suspicion of malignancy.",
            "Fibroadenoma.",
            "Dysplasia.",
        ],
        "image_path":   [
            "data/images/case001.png",
            "data/images/case002.png",
            "data/images/case003.png",
        ],
    })


# ── clean_text Tests ──────────────────────────────────────────────────────────

class TestCleanText:

    def test_lowercase(self):
        assert clean_text("HELLO WORLD") == "hello world"

    def test_removes_nan_token(self):
        result = clean_text("Patient is nan years old.")
        assert "nan" not in result

    def test_removes_xxxx_token(self):
        result = clean_text("Signs: xxxx and x-xxxx")
        assert "xxxx" not in result
        assert "x-xxxx" not in result

    def test_handles_newline(self):
        result = clean_text("Line one\nLine two")
        assert "\n" not in result
        assert "line one line two" == result

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_strips_whitespace(self):
        assert clean_text("  hello  ") == "hello"


# ── build_combined_reports Tests ──────────────────────────────────────────────

class TestBuildCombinedReports:

    def test_returns_list(self, sample_df, tmp_path):
        out = tmp_path / "reports.txt"
        reports = build_combined_reports(sample_df, save_path=str(out))
        assert isinstance(reports, list)

    def test_length_matches_dataframe(self, sample_df, tmp_path):
        out = tmp_path / "reports.txt"
        reports = build_combined_reports(sample_df, save_path=str(out))
        assert len(reports) == len(sample_df)

    def test_contains_sep_token(self, sample_df, tmp_path):
        out = tmp_path / "reports.txt"
        reports = build_combined_reports(sample_df, save_path=str(out))
        for report in reports:
            assert "[SEP]" in report

    def test_contains_idx_token(self, sample_df, tmp_path):
        out = tmp_path / "reports.txt"
        reports = build_combined_reports(sample_df, save_path=str(out))
        for report in reports:
            assert "[IDX]" in report

    def test_saves_file(self, sample_df, tmp_path):
        out = tmp_path / "reports.txt"
        build_combined_reports(sample_df, save_path=str(out))
        assert out.exists()
        lines = out.read_text().strip().split("\n")
        assert len(lines) == len(sample_df)

    def test_nan_tokens_removed(self, sample_df, tmp_path):
        out = tmp_path / "reports.txt"
        reports = build_combined_reports(sample_df, save_path=str(out))
        combined = " ".join(reports)
        assert "xxxx" not in combined


# ── data_loader Tests ─────────────────────────────────────────────────────────

class TestConvertRowToText:

    def test_output_is_tuple(self):
        from src.preprocessing.data_loader import _convert_row_to_text
        row = pd.Series({
            "Age": 45,
            "Tissue_composition": "heterogeneous: fibroglandular",
            "Signs": "no",
            "Symptoms": "not available",
            "Shape": "irregular",
            "Margin": "not circumscribed",
            "Echogenicity": "hypoechoic",
            "Posterior_features": "no",
            "Interpretation": "Suspicion of malignancy",
        })
        result = _convert_row_to_text(row)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_input_text_contains_age(self):
        from src.preprocessing.data_loader import _convert_row_to_text
        row = pd.Series({
            "Age": 52,
            "Tissue_composition": "homogeneous: fat",
            "Signs": "no", "Symptoms": "no",
            "Shape": "oval", "Margin": "circumscribed",
            "Echogenicity": "anechoic", "Posterior_features": "enhancement",
            "Interpretation": "Cyst",
        })
        input_text, _ = _convert_row_to_text(row)
        assert "52" in input_text

    def test_target_text_is_interpretation(self):
        from src.preprocessing.data_loader import _convert_row_to_text
        row = pd.Series({
            "Age": 30, "Tissue_composition": "x",
            "Signs": "x", "Symptoms": "x",
            "Shape": "x", "Margin": "x",
            "Echogenicity": "x", "Posterior_features": "x",
            "Interpretation": "Fibroadenoma",
        })
        _, target = _convert_row_to_text(row)
        assert target == "Fibroadenoma"
