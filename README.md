# 🔬 Breast Ultrasound Report Generation using CNN-Transformer + XAI

[![Python](https://img.shields.io/badge/Python-3.11-blue)]()
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)]()
[![University](https://img.shields.io/badge/MSc%20AI-University%20of%20East%20London-purple)]()

> Automated radiology report generation from breast ultrasound images 
> using a hybrid ResNet101 + Transformer architecture, enhanced with 
> Grad-CAM and Attention Heatmap explainability.

---

## 📌 Project Overview

 > Designed a hybrid ResNet101 + Transformer encoder-decoder architecture to automatically generate structured 
radiology reports from ultrasound images, targeting real-world clinical deployment.
> Applied attention mechanisms for visual feature extraction from the 256-image BUS dataset; evaluated generation 
quality using BLEU, METEOR, and CIDEr metrics.
> Incorporated Explainable AI (XAI) using Grad-CAM and attention heatmaps to improve model transparency and 
interpretability, directly addressing trust in clinical AI.
> Demonstrated end-to-end pipeline: preprocessing, model training, multi-metric evaluation, and qualitative analysis.
> 
## 🏗️ Architecture

<img width="561" height="422" alt="Capture" src="https://github.com/user-attachments/assets/6fcd6d52-77d3-491f-beec-4228478c683b" />


## 📊 Results

| Metric  | Score |
|---------|-------|
| BLEU-1  | 0.486 |
| METEOR  | 0.203 |
| CIDEr   | 0.118 |
| ROUGE-L | 0.168 |

## 🔍 XAI Visualizations

<img width="794" height="394" alt="grad cam 1" src="https://github.com/user-attachments/assets/59055abf-5710-48f2-a4fa-2666e943631c" />
<img width="1048" height="656" alt="heat map done" src="https://github.com/user-attachments/assets/03f46787-67ac-4fd1-a6d8-ecd937a9256e" />


## 🚀 How to Run

pip install -r requirements.txt
python src/models/hybrid_model.py

## 📂 Dataset



## 🛠️ Tech Stack
- Python 3.11 | TensorFlow | Keras
- ResNet101 | Transformer Encoder-Decoder
- Grad-CAM | Attention Heatmaps
- BLEU / METEOR / ROUGE / CIDEr evaluation

## 📄 Thesis
MSc Artificial Intelligence and Data Science — University of East London, 2024
Supervised by Dr. Shaheen Khatoon

## 📬 Contact
linkedin.com/in/brinthat | brintha13@gmail.com
