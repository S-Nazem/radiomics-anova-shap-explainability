# Radiomic Feature Sensitivity and Classification in Photoacoustic Imaging and CT (MPhil Project)

This repository contains the complete code, data pipeline, and analysis for my MPhil project on radiomic feature robustness and classification across two imaging modalities:

- **Section A:** Reproduction of Escudero Sánchez et al. (2022) using full factorial ANOVA, feature selection, machine learning, and SHAP interpretability on photoacoustic imaging of breast cancer PDXs.
- **Section B:** Extension study applying the same radiomics pipeline to the LIDC-IDRI CT dataset for lung nodule classification.

---

## Project Structure

```bash
sn665/
│
├── data/                         # Radiomic CSVs and metadata
│   ├── LIDC_Extension/           # Includes DICOM + XMLs from LIDC-IRDI plus csvs
│   ├── Photoacoustic_Study/      # PAI radiomics and metadata csvs
│   └── ModelsUncorrected/        # Radiomic Features (PAI Study) uncorrected
│
├── Finalised Notebooks/         # Final cleaned Jupyter notebooks
│   ├── ANOVA.ipynb              # Section A: ANOVA Sensitivity analysis
│   ├── MODEL_PERFORMANCE.ipynb  # Section A: MOdel Discrimination: ML + SHAP
│   ├── EXTENSION.ipynb          # Section B: LIDC-IRDI Extension
│   └── volume_results.pkl       # Precomputed results for SHAP rank plots
│
├── Development Notebooks/       # Early exploratory development notebooks (included for interest)
├── MATLAB scripts (ANOVA)/      # MATLAB code for ANOVA calculations
│
├── src/                         # Custom scripts (clustering, SHAP, plots, etc.)
├── plots/                       # Figures from Section A (PAI)
│   ├── ANOVA_plots/
│   ├── Distributions_plots/
│   ├── SHAP_plots/
│   └── misc/
│
├── plots_extension/             # Figures from Section B (LIDC)
│   ├── feature_selection_plots/
│   ├── ML_plots/
│   ├── shap_plots/
│   └── misc/
│
├── Instructions.md              # Internal dev instructions
├── requirements.txt             # Environment dependencies
└── README.md                    # This file

```

--


## Environment Setup

Recommended: Python 3.9+

Install required packages using:

```bash
pip install -r requirements.txt
```

Main dependencies include:

- pandas, numpy, scikit-learn, imblearn, shap, xgboost
- pydicom, SimpleITK, pyradiomics
- matplotlib, seaborn, tqdm

--


## Reproducibility Instructions

Section A — Photoacoustic (Escudero Sanchez reproduction)
- Run: Dev_notebook.ipynb → ANOVA sensitivity analysis
- Run: Dev_notebook_2.ipynb → IQR variability plots
- Run: Dev_notebook_3.ipynb → Feature selection, ML classification, SHAP explanations


Section B — LIDC Extension (CT dataset)
- Run: Dev_notebook_4.ipynb → Full pipeline for lung CT extension


--

## Notes

- The MATLAB scripts compute full-factorial ANOVA statistics (η²) and were used for reproduction consistency.
- DICOM files used for LIDC radiomics are referenced in data/LIDC_Extension/, but not pushed due to size.
- SHAP stability was assessed using 1000 seeds with saved output (volume_results.pkl).
- To generate new results from scratch (using a new random seed), uncomment the three "to_csv" lines scattered throughout EXTENSION.ipynb. This will regenerate:
    - 3D volume reconstructions
    - Radiomic feature extraction
    - Feature selection and ML classification
    - SHAP interpretation
    - Slight variation in outputs is expected due to randomness in SMOTE and model training.


--


## Use of Generative Tools 

- I used Github's Copilot to help me automatically finish off some code blocks and also to quickly docstring my functions.

- I used LLMs (ChatGPT) to help me create professional looking plots and occasionally to help me debug errors when i implemented something incorrectly.

