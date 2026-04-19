# 🔬 Breast Ultrasound Report Generation
### Hybrid ViT + distilGPT-2 with Explainable AI (Grad-CAM & Attention Maps)

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange?logo=tensorflow)](https://tensorflow.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red?logo=pytorch)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/🤗_Transformers-4.35-yellow)](https://huggingface.co)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![University](https://img.shields.io/badge/MSc_AI-University_of_East_London-purple)](https://uel.ac.uk)

> **MSc Artificial Intelligence & Data Science — Final Thesis**
> University of East London | Supervised by Dr. Shaheen Khatoon

---

## 📌 Overview

Breast cancer is the most prevalent cancer in women worldwide. Radiologists must manually
interpret ultrasound images and produce detailed BI-RADS-compliant reports — a time-consuming,
error-prone process with up to **30% misdiagnosis rates** in difficult cases.

This project builds an **end-to-end AI pipeline** that:

1. Extracts visual features from breast ultrasound images using a **Vision Transformer (ViT)**
2. Finds the most visually similar case using **cosine similarity**
3. Generates a structured radiology report using a **fine-tuned distilGPT-2** language model
4. Explains model decisions with **Grad-CAM heatmaps** and **Transformer attention maps**

---

## 🏗️ System Architecture

```
Breast Ultrasound Image
        │
        ▼
┌───────────────────┐
│  ViT Feature      │  timm vit_base_patch16_224
│  Extractor        │  → (197, 768) feature vectors
└────────┬──────────┘
         │  Cosine Similarity Search
         ▼
┌───────────────────┐
│  Most Similar     │  Match against pre-extracted
│  Image Retrieval  │  dataset features (.npy)
└────────┬──────────┘
         │  Clinical prompt text
         ▼
┌───────────────────┐
│  distilGPT-2      │  Fine-tuned on 256 annotated
│  Report Generator │  breast ultrasound reports
└────────┬──────────┘
         │
         ▼
┌───────────────────┐     ┌─────────────────────┐
│  Generated Report │     │  XAI Explanations   │
│  (text)           │     │  Grad-CAM + Attn Map│
└───────────────────┘     └─────────────────────┘
```

---

## 📊 Key Results

| Metric           | Score  |
|------------------|--------|
| BLEU-1           | 0.486  |
| BLEU-2           | 0.315  |
| BLEU-3           | 0.224  |
| BLEU-4           | 0.168  |
| METEOR           | 0.203  |
| ROUGE-L          | 0.118  |
| CIDEr            | 0.134  |

### Comparison with Baselines

| Method          | BLEU-1 | METEOR | ROUGE-L |
|-----------------|--------|--------|---------|
| RMAP            | 0.416  | 0.161  | 0.303   |
| WGAM-KIRN       | 0.488  | 0.298  | 0.433   |
| R2GenGPT        | 0.488  | 0.211  | 0.377   |
| **Ours**        | **0.486** | **0.203** | **0.118** |

---

## 🛠️ Tech Stack

| Component          | Technology                              |
|--------------------|-----------------------------------------|
| Language           | Python 3.11                             |
| Image features     | ViT (timm `vit_base_patch16_224`)       |
| Language model     | distilGPT-2 (HuggingFace Transformers)  |
| Image pipeline     | TensorFlow 2.x / tf.data               |
| Training           | PyTorch 2.0 + AdamW                     |
| XAI                | Grad-CAM, Transformer Attention Maps    |
| Text evaluation    | BLEU, ROUGE, BERTScore                  |
| EDA                | Pandas, Seaborn, WordCloud, spaCy       |

---

## 📁 Repository Structure

```
breast-ultrasound-report-generation/
│
├── main.py                          ← End-to-end pipeline entry point
├── requirements.txt
├── README.md
├── DATA.md                          ← Dataset access instructions
├── .gitignore
│
├── configs/
│   └── config.py                   ← All paths and hyperparameters
│
├── src/
│   ├── preprocessing/
│   │   ├── data_loader.py          ← CSV → structured text reports
│   │   ├── eda.py                  ← Word clouds, n-grams, LDA topics
│   │   ├── image_preprocessor.py  ← TF image pipeline (resize, normalise)
│   │   └── text_preprocessor.py   ← Cleaning, tokenisation, DataLoader
│   │
│   ├── models/
│   │   ├── feature_extractor.py   ← ViT feature extraction + similarity search
│   │   └── report_generator.py    ← GPT-2 fine-tuning + report generation
│   │
│   └── evaluation/
│       └── metrics.py             ← BLEU, ROUGE-L, BERTScore
│
├── notebooks/
│   ├── 01_EDA_and_Preprocessing.ipynb
│   ├── 02_Feature_Extraction.ipynb
│   ├── 03_Model_Training.ipynb
│   └── 04_Evaluation_and_XAI.ipynb
│
└── results/
    ├── figures/                    ← EDA plots, Grad-CAM heatmaps
    ├── metrics/                    ← evaluation_results.json
    └── sample_reports/             ← Generated vs ground truth examples
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/breast-ultrasound-report-generation.git
cd breast-ultrasound-report-generation
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Set Up Dataset

See [DATA.md](DATA.md) for full instructions.
Download the BrEaST-Lesions dataset from:
👉 https://doi.org/10.1038/s41597-024-02984-z

Place images in `data/images/` and `reports.csv` in `data/`.

### 3. Configure Paths

Edit `configs/config.py` if your data directory differs from the default.

### 4. Run the Full Pipeline

```bash
# Full pipeline (preprocess → train → evaluate)
python main.py

# Individual steps
python main.py --step eda       # EDA only
python main.py --step train     # Training only
python main.py --step infer --query data/images/case001.png
```

---

## 🔍 XAI Explanations

The project includes two complementary explainability methods:

**Grad-CAM** — highlights image regions that most influenced the model's feature extraction, overlaid as a heatmap on the original ultrasound image.

**Attention Maps** — visualise which image patches the Transformer attended to at each step of report generation, showing the model's internal focus during text production.

Both are demonstrated in `notebooks/04_Evaluation_and_XAI.ipynb`.

---

## 📂 Dataset

The **BrEaST-Lesions** dataset (Pawłowska et al., 2024) comprises 256 expert-annotated
breast ultrasound images collected from five radiologists across multiple Polish medical
centres (2019–2022). It includes BI-RADS descriptors, histological diagnoses, and biopsy
confirmation. See [DATA.md](DATA.md) for access instructions.

> ⚠️ Raw images are NOT included in this repository due to ethical restrictions
> (ethics approval: Lower Silesian Chamber of Medicine no. 2/BNR/2022).

---

## 📄 Thesis

**Title:** Leveraging Integrated CNN-Transformer Model and eXplainable Artificial Intelligence (XAI) Technique for Enhanced Breast Ultrasound Report Generation

**Degree:** MSc Artificial Intelligence and Data Science
**Institution:** University of East London
**Supervisor:** Dr. Shaheen Khatoon
**Year:** 2024

---

## 📬 Contact

**Brintha Thirunavukkarasu**
- 🔗 [LinkedIn](https://linkedin.com/in/YOUR_PROFILE)
- 📧 your.email@example.com

---

## 📑 Citation

If you use this work, please cite:

```
Thirunavukkarasu, B. (2024). Breast Ultrasound Report Generation using
Hybrid ViT + GPT-2 with Explainable AI. MSc Thesis,
University of East London.
```

And the dataset:

```
Pawłowska, A. et al. (2024). Curated benchmark dataset for
ultrasound-based breast lesion analysis. Scientific Data, 11, 148.
https://doi.org/10.1038/s41597-024-02984-z
```
