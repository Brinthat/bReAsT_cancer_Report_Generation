# Dataset Information

## ⚠️ Important — Raw Data Not Included

This repository does **not** include raw patient images or clinical reports.
The dataset contains identifiable medical information and is subject to
ethical approval granted by the **Lower Silesian Chamber of Medicine's
Bioethics Committee (no. 2/BNR/2022)**.

---

## Dataset: BrEaST-Lesions (Pawłowska et al., 2024)

| Property         | Detail                                       |
|------------------|----------------------------------------------|
| Images           | 256 expert-annotated breast ultrasound PNGs  |
| Annotations      | 5 radiologists across multiple Polish centres |
| Collection period| 2019 – 2022                                  |
| Cases            | 154 benign · 98 malignant · 4 normal         |
| Biopsy confirmed | 197 biopsies                                 |

**Citation:**
> Pawłowska, A. et al. (2024). Curated benchmark dataset for ultrasound-based
> breast lesion analysis. *Scientific Data*, 11, 148.
> https://doi.org/10.1038/s41597-024-02984-z

---

## How to Access the Dataset

1. Visit the official dataset page:
   👉 https://doi.org/10.1038/s41597-024-02984-z

2. Download:
   - `BrEaST-Lesions_USG-images_and_masks/` — PNG images + masks
   - `reports.csv` — Clinical annotations

3. Place files in the following structure:

```
breast-ultrasound-report-generation/
└── data/
    ├── reports.csv
    └── images/
        ├── case001.png
        ├── case002.png
        └── ...
```

4. Update `configs/config.py` if your paths differ:

```python
RAW_IMAGE_DIR = "data/images"
REPORTS_CSV   = "data/reports.csv"
```

---

## CSV Column Reference

| Column              | Description                                  |
|---------------------|----------------------------------------------|
| CaseID              | Unique patient case identifier               |
| Image_filename      | PNG filename                                 |
| Age                 | Patient age (18–87)                          |
| Tissue_composition  | Breast tissue type                           |
| Signs               | Clinical signs                               |
| Symptoms            | Patient-reported symptoms                    |
| Shape               | Lesion shape (oval / irregular / round)      |
| Margin              | Lesion margin descriptor                     |
| Echogenicity        | Hypoechoic / hyperechoic / etc.              |
| Posterior_features  | Enhancement / shadowing / none               |
| Halo                | Presence of halo                             |
| Calcifications      | Yes / No                                     |
| Skin_thickening     | Yes / No                                     |
| Interpretation      | Radiologist's interpretation (target text)   |
| BIRADS              | BI-RADS category (1 / 2 / 3 / 4a / 4b / 4c / 5) |
| Verification        | Biopsy / Follow-up                           |
| Diagnosis           | Histological diagnosis                       |
| Classification      | Benign / Malignant                           |
