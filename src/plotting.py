import matplotlib.pyplot as plt
import numpy as np
import shap
import pandas as pd
import pingouin as pg
import seaborn as sns



def plot_iqr_by_wavelength(df, features, titles, y_lims, recons, save_path):
    """
    Plot IQR plots for radiomic features across wavelength and tumour model types.

    Args:
        df (pd.DataFrame): DataFrame containing features, Model, Wavelength, Reconstruction.
        features (List[str]): List of feature column names to plot.
        titles (List[str]): Titles for the corresponding subplots.
        y_lims (List[Tuple[float, float]]): y-axis limits for each feature plot.
        recons (List[str]): List of reconstruction types, e.g. ["BP", "MB"].
        save_path (str or Path): Output path for the saved figure.
    """
    colors = {"basal": "red", "luminal": "black"}
    markers = {"basal": "^", "luminal": "o"}

    fig, axs = plt.subplots(len(features), len(recons), figsize=(12, 8), sharex=True, dpi=300)

    for col, recon in enumerate(recons):
        for row, feature in enumerate(features):
            ax = axs[row, col]
            df_sub = df[df["Reconstruction"] == recon]

            for model in ["basal", "luminal"]:
                df_model = df_sub[df_sub["Model"] == model]
                grouped = df_model.groupby("Wavelength")[feature]
                med = grouped.median()
                iqr_min = grouped.quantile(0.25)
                iqr_max = grouped.quantile(0.75)

                wl = med.index.values
                ax.plot(wl, med.values, markers[model], color=colors[model],
                        label=model.capitalize() if row == 0 else "", markersize=8)
                ax.vlines(wl, iqr_min.values, iqr_max.values,
                          color=colors[model], linestyle='dotted')

            ax.set_title(f"{titles[row]} ({'Backprojection' if recon == 'BP' else 'Model-based'})")
            ax.set_ylim(y_lims[row])
            if row == len(features) - 1:
                ax.set_xlabel("Wavelength [nm]")
            ax.set_ylabel(titles[row])

            if row == 0 and col == 1:
                ax.legend(title="Model", loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_shap_summary_and_bar(shap_values_class1, X_subset, save_prefix):
    """
    Generate SHAP beeswarm and bar plots and save them with the given prefix.

    Parameters:
    - shap_values_class1: SHAP values (np.array), shape = [n_samples, n_features]
    - X_subset: pandas DataFrame of input features (same order as shap values)
    - save_prefix: str, filename prefix (e.g. 'shap_full', 'shap_onefold')
    """
    # === Beeswarm Plot ===
    shap.summary_plot(
        shap_values_class1,
        X_subset,
        feature_names=X_subset.columns,
        plot_size=(18, 6),
        max_display=10,
        show=False
    )
    plt.xlabel("SHAP value (impact on model output)", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig(f"../plots/SHAP_plots/{save_prefix}_beeswarm.png", dpi=400)
    plt.show()

    # === Bar Plot ===
    mean_abs_shap = np.abs(shap_values_class1).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-10:][::-1]
    top_features = X_subset.columns[top_indices]
    top_values = mean_abs_shap[top_indices]

    plt.figure(figsize=(18, 6), dpi=400)
    plt.barh(range(len(top_features)), top_values, color="#d62728")
    plt.yticks(range(len(top_features)), top_features, fontsize=18)
    plt.xticks(fontsize=14)
    plt.xlabel("mean(|SHAP value|)", fontsize=18)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"../plots/SHAP_plots/{save_prefix}_bar.png", dpi=400)
    plt.show()


def plot_rm_corr_heatmaps(df_selected, features, subject_col='PatientName', save_path="../plots/misc/rm_corr_heatmaps.png"):
    """
    Plot repeated measures correlation and p-value heatmaps.

    Parameters:
    - df_selected: DataFrame with features + subject ID column
    - features: list of feature names to compute correlation on
    - subject_col: column name with subject IDs
    - save_path: where to save the output figure
    """

    n = len(features)
    corr_matrix = np.zeros((n, n))
    pval_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                corr_matrix[i, j] = 1
                pval_matrix[i, j] = 0
            else:
                res = pg.rm_corr(data=df_selected, x=features[i], y=features[j], subject=subject_col)
                corr_matrix[i, j] = res['r'].values[0]
                pval_matrix[i, j] = res['pval'].values[0]

    corr_df = pd.DataFrame(corr_matrix, index=range(1, n+1), columns=range(1, n+1))
    pval_df = pd.DataFrame(pval_matrix, index=range(1, n+1), columns=range(1, n+1))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(corr_df, ax=axes[0], cmap='Spectral', center=0, cbar_kws={'label': 'Correlation'})
    axes[0].set_title("Repeated Measures Correlation (r)")
    axes[0].set_xlabel("Feature ID")
    axes[0].set_ylabel("Feature ID")

    sns.heatmap(pval_df, ax=axes[1], cmap='viridis_r', vmin=0, vmax=1, cbar_kws={'label': 'p-value'})
    axes[1].set_title("p-values of rm_corr")
    axes[1].set_xlabel("Feature ID")
    axes[1].set_ylabel("")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()



def plot_kw_pvalues(kw_df, save_path="../plots/misc/kw_pvalues.png"):
    """
    Plot sorted Kruskal-Wallis p-values with FDR threshold line.
    """
    kw_df_plot = kw_df.reset_index()

    plt.figure(figsize=(10, 6), dpi=400)
    plt.scatter(range(len(kw_df_plot)), kw_df_plot["p_value"], color="black", s=60)
    plt.axhline(y=0.25, color="red", linestyle="--", linewidth=2, label="FDR threshold (0.25)")
    plt.ylim(0, 1.05)
    plt.xlim(-1, len(kw_df_plot))
    plt.ylabel("Kruskal-Wallis p-value", fontsize=14)
    plt.xlabel("Feature ID", fontsize=14)
    plt.xticks(np.arange(0, len(kw_df_plot)+1, 10), fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis="y", linestyle=":", color="grey", alpha=0.5)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=400)
    plt.show()
