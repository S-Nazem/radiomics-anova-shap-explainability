# Radiomic Feature Sensitivity and Classification in Photoacoustic Imaging and CT (MPhil Project)

This repository contains the complete code, data pipeline, and analysis for my MPhil project on radiomic feature robustness and classification across two imaging modalities:

- **Section A:** Reproduction of Escudero Sánchez et al. using full factorial ANOVA, feature selection, machine learning, and SHAP interpretability on photoacoustic imaging of breast cancer PDXs.
- **Section B:** Extension study applying the same radiomics pipeline to the LIDC-IDRI CT dataset for lung nodule classification.

---


## Key Results

**Section A:** PAI Radiomics Study

1. Sensitivity Analysis (ANOVA)

Full-factorial ANOVA revealed key radiomic features sensitive to the model type and not to confounding factors (ie: FO Skewness, FO Kurtosis).

<p align="center"> <img src="plots/ANOVA_plots/ANOVA_full.png" width="600"/> </p>

2. SHAP Analysis

SHAP values identified FO 10th percentile, FO 90th Percentile (RMS), FO Skewness, FO Variance, and GLCM Imc2 as the most impactful features for the RF classifier

<p align="center"> <img src="plots/SHAP_plots/shap_beeswarm.png" width="600"/> </p>


**Section B:** LIDC-IRDI Extension Study

1. Classifier Performance Plot (RFE vs Boruta) for both original and resampled sets

<p align="center"> <img src="plots_extension/ML_plots/combined_rfe_boruta_smote_shaded.png" width="600"/> </p>

2. SHAP Analysis

SHAP analysis identified GLSZM Gray Level Non uniformity and its normalised counterpart as the most impactful features. Very little overlap between the PAI study and this, as could be expected.

<p align="center"> <img src="plots_extension/shap_plots/shap_bar_full.png" width="600"/> </p>

3. SHAP Consistency Box plots

SHAP analysis repeated over 1000 independant seeds unanimously showed GLSZM Gray Level Non Unifomity as the most influential feature.

<p align="center"> <img src="plots_extension/shap_plots/shap_rank_stability_horizontal_nosmote.png" width="600"/> </p>


--

## Project Structure

```bash
sn665/
│
├── data/                         # Radiomic CSVs and metadata
│   ├── LIDC_Extension/csv           # Includes DICOM + XMLs from LIDC-IDRI plus csvs
│   ├── Photoacoustic_Study/csv      # PAI radiomics and metadata csvs
│
├── Finalised Notebooks/          # Final cleaned Jupyter notebooks
│   ├── ANOVA.ipynb               # Section A: ANOVA Sensitivity analysis
│   ├── MODEL_PERFORMANCE.ipynb   # Section A: Model Discrimination (ML + SHAP)
│   ├── EXTENSION.ipynb           # Section B: LIDC-IDRI Extension
│
├── Development Notebooks/        # Early exploratory notebooks (included for reference)
│
├── MATLAB scripts (ANOVA)/       # MATLAB scripts used for full factorial ANOVA
│
├── src/                          # Custom Python modules (importable functions)
│   ├── Extension.py              # Add code used for the LIDC-IRDI extension
│   ├── feature_selection.py      # Feature selection utilities
│   ├── ML.py                     # ML training
│   ├── PAI.py                    # SHAP, ML, and ANOVA utils for Section A (PAI)
│   └── plotting.py               # Plotting functions for section A
│
├── plots/                        # Figures from Section A (PAI)
│   ├── ANOVA_plots/
│   ├── Distributions_plots/
│   ├── SHAP_plots/
│   └── misc/
│
├── plots_extension/              # Figures from Section B (LIDC-IDRI extension)
│   ├── feature_selection_plots/
│   ├── ML_plots/
│   ├── shap_plots/
│   └── misc/
│
├── Instructions.md               # Internal usage instructions
├── requirements.txt              # Python environment dependencies
├── README.md                     # This file
└── .gitignore                    # Files and folders to ignore in Git



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
- ANOVA.ipynb → ANOVA sensitivity analysis
- MODEL_PERFORMANCE.ipynb → IQR variability plots, feature selection, ML and SHAP


Section B — LIDC Extension (CT dataset)
- EXTENSION.ipynb → Feature selection (RFE/Boruta), ML classification, SHAP explanations, SHAP boxplots

--

## Notes

- The MATLAB scripts compute full-factorial ANOVA statistics (η²).
- DICOM files used for LIDC radiomics were referenced in data/LIDC_Extension/, but not pushed due to size.
- data/Photoacoustic_Study/csv contain all required radiomic CSVs for Part A.
- data/LIDC_Extension/csv includes the csvs for Part B.
- data/LIDC_Extension/volume_results.pkl (used in EXTENSION.ipynb) is not pushed due to large size (>2GB).
- To reproduce the results, run the volume creation block in the notebook.
- When prompted Compute new volumes? [Y/n], type Y to generate the file.
- All core functions (radiomic loading, feature selection, ML, SHAP, plotting) are implemented in the src/ folder.

--


## Use of Generative Tools 

- I used Github's Copilot to help me automatically finish off some code blocks and also to quickly docstring my functions.

- I used LLMs (ChatGPT) to help me create professional looking plots and occasionally to help me debug errors when i implemented something incorrectly.