# Radiomic Feature Sensitivity and Classification in Photoacoustic Imaging and CT (MPhil Project)

This repository contains the full code and data processing pipeline used for my MPhil project, structured into two main sections:

- **Section A:** Reproduction of Escudero Sanchez et al. (2022) — Full factorial ANOVA sensitivity analysis, feature selection, and classification on photoacoustic radiomics.
- **Section B:** Extension study applying the same radiomic pipeline to the LIDC-IDRI lung CT dataset.

---

## Project Structure

```bash
sn665/
│
├── data/                   # CSV data files used for analysis (radiomics, metadata, etc.)
│   └── Extension/          # DICOM image files for the LIDC-IDRI extension dataset
│
├── MATLAB CODE (ANOVA)/    # MATLAB scripts for full factorial ANOVA sensitivity analysis
│
├── plots/                  # Figures and plots generated for Section A
├── plots_extension/        # Figures and plots generated for Section B (LIDC extension)
│
├── Dev_notebook.ipynb      # Section A: Full ANOVA sensitivity analysis (photoacoustic)
├── Dev_notebook_2.ipynb    # Section A: IQR plots for feature variability
├── Dev_notebook_3.ipynb    # Section A: Feature selection, ML models, SHAP analysis
├── Dev_notebook_4.ipynb    # Section B: LIDC extension (full pipeline: preprocessing, radiomics, ML, SHAP)
│
├── figs.pptx               # Exported figures for inclusion in final report
└── Instructions.md         # Internal notes and steps for reproduction
```

--


## Environment Setup

You can recreate the environment using Python 3.9+ and the following packages:

```bash
pip install -r requirements.txt
```

Main dependencies include:

- pandas, numpy, scikit-learn, imblearn, shap, xgboost
- pydicom, SimpleITK, pyradiomics
- matplotlib, seaborn, tqdm

--


## How to Reproduce

Section A — Photoacoustic (Escudero Sanchez reproduction)
- Run: Dev_notebook.ipynb → ANOVA sensitivity analysis
- Run: Dev_notebook_2.ipynb → IQR variability plots
- Run: Dev_notebook_3.ipynb → Feature selection, ML classification, SHAP explanations


Section B — LIDC Extension (CT dataset)
- Run: Dev_notebook_4.ipynb → Full pipeline for lung CT extension



--

## Notes

- DICOM files are stored under data/Extension/ and are required for radiomic extraction.
- Full factorial ANOVA is computed using MATLAB (see MATLAB CODE (ANOVA)/).
- All final radiomics CSVs and ML-ready datasets are available under data/ for reproducibility.

