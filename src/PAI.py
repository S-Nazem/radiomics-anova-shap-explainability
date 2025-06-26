import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

def extract_group_and_shortname(feature: str):
    """
    Extract the feature group and short name from a radiomic feature string.
    """
    if 'firstorder' in feature:
        group = 'FOS'
        short = feature.split('firstorder_')[-1]
    elif 'glcm' in feature:
        group = 'GLCM'
        short = feature.split('glcm_')[-1]
    elif 'gldm' in feature:
        group = 'GLDM'
        short = feature.split('gldm_')[-1]
    elif 'glrlm' in feature:
        group = 'GLRLM'
        short = feature.split('glrlm_')[-1]
    elif 'glszm' in feature:
        group = 'GLSZM'
        short = feature.split('glszm_')[-1]
    elif 'ngtdm' in feature:
        group = 'NGTDM'
        short = feature.split('ngtdm_')[-1]
    else:
        group = 'Other'
        short = feature
    return group, short



def get_group_boundaries(groups):
    """
    Compute index positions where feature group changes for vertical line plotting.
    """
    boundaries = []
    prev_group = groups[0]
    for idx, group in enumerate(groups):
        if group != prev_group:
            boundaries.append(idx)
            prev_group = group
    return boundaries


def plot_anova_full_factorial(df, save_path):
    """
    Plot the stacked bar chart of η² values from full factorial ANOVA output.

    Args:
        df (pd.DataFrame): DataFrame with η² values, indexed by feature name.
        save_path (str or Path): Output path for the saved figure.
    """

    # Assign feature group and shortname
    df['Group'], df['ShortName'] = zip(*df.index.map(extract_group_and_shortname))
    df_sorted = df.sort_values(by=['Group', 'ShortName'])

    # Plotting configuration
    colors = {
        "Model": "#2ca02c",
        "GLbins": "#ADD8E6",
        "Wavelength": "#FFD700",
        "Reconstruction": "#800080",
        "Residual": "#000000"
    }
    factors = list(colors.keys())
    group_names = ['FOS', 'GLCM', 'GLDM', 'GLRLM', 'GLSZM', 'NGTDM']

    x_labels = df_sorted['ShortName'].tolist()
    groups = df_sorted['Group'].tolist()
    group_boundaries = get_group_boundaries(groups)

    fig, ax = plt.subplots(figsize=(16, 8))
    bottom = pd.Series(0, index=df_sorted.index)

    for factor in factors:
        ax.bar(range(len(df_sorted)), df_sorted[factor], bottom=bottom, color=colors[factor], label=factor)
        bottom += df_sorted[factor]

    ax.set_xticks(range(len(df_sorted)))
    ax.set_xticklabels(x_labels, rotation=90, fontsize=10)

    # Add vertical lines
    for boundary in group_boundaries:
        ax.axvline(x=boundary - 0.5, color='grey', linestyle='--', linewidth=0.7)

    # Add group labels
    for group in group_names:
        indices = [i for i, g in enumerate(groups) if g == group]
        if indices:
            center_pos = (indices[0] + indices[-1]) / 2
            if group == 'NGTDM':
                center_pos += 1.8
            ax.text(center_pos, 1.02, group, ha='center', va='bottom', fontsize=16.5, fontweight='bold')

    ax.set_ylim(0, 1.1)
    ax.set_ylabel("η²", fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.legend(bbox_to_anchor=(1.0, 1), loc="upper left", fontsize=12, title="Factor", title_fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=400)
    plt.show()




def plot_anova_by_glbins(glbins_values, input_dir, save_path):
    """
    Plot η² stacked bar plots for each GL bin level.

    Args:
        glbins_values (List[int]): List of grey-level bin values (e.g., [8, 16, ...])
        input_dir (str or Path): Path to CSV directory
        save_path (str or Path): Output image path
    """
    colors = {
        "Model": "#2ca02c",
        "Wavelength": "#FFD700",
        "Reconstruction": "#800080",
        "Residual": "#000000"
    }
    factors = list(colors.keys())
    group_names = ['FOS', 'GLCM', 'GLDM', 'GLRLM', 'GLSZM', 'NGTDM']

    fig, axs = plt.subplots(3, 2, figsize=(25, 16), dpi=400)
    axs = axs.flatten()

    for i, gl in enumerate(glbins_values):
        df = pd.read_csv(f"{input_dir}/anova_eta2_glbins_{gl}.csv", index_col=0)
        df['Group'], df['ShortName'] = zip(*df.index.map(extract_group_and_shortname))
        df_sorted = df.sort_values(by=['Group', 'ShortName'])
        groups = df_sorted['Group'].tolist()
        group_boundaries = get_group_boundaries(groups)

        ax = axs[i]
        bottom = pd.Series(0, index=df_sorted.index)

        for factor in factors:
            ax.bar(range(len(df_sorted)), df_sorted[factor], bottom=bottom, color=colors[factor], label=factor)
            bottom += df_sorted[factor]

        ax.tick_params(axis='y', labelsize=16)
        for boundary in group_boundaries:
            ax.axvline(x=boundary - 0.5, color='grey', linestyle='--', linewidth=0.7)

        for group in group_names:
            indices = [j for j, g in enumerate(groups) if g == group]
            if indices:
                center_pos = (indices[0] + indices[-1]) / 2
                if group == 'NGTDM':
                    center_pos += 1.5
                ax.text(center_pos, 1.0, group, ha='center', va='bottom', fontsize=16, fontweight='bold')

        ax.set_ylim(0, 1.1)
        ax.set_ylabel("η²", fontsize=15)
        ax.set_title(f"GLbins = {gl}", fontsize=28)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.99, 0.95), fontsize=15, title="Factor", title_fontsize=12)
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(save_path, dpi=400)
    plt.show()


def plot_anova_by_reconstruction(recon_types, input_dir, save_path):
    """
    Plot η² stacked bar plots for each reconstruction method (e.g., BP, MB).

    Args:
        recon_types (List[str]): Reconstruction types to compare (e.g., ['BP', 'MB'])
        input_dir (str or Path): Path to CSV directory
        save_path (str or Path): Output image path
    """
    colors = {
        "Model": "#2ca02c",
        "GLbins": "#ADD8E6",
        "Wavelength": "#FFD700",
        "Residual": "#000000"
    }
    factors = list(colors.keys())
    group_names = ['FOS', 'GLCM', 'GLDM', 'GLRLM', 'GLSZM', 'NGTDM']

    fig, axs = plt.subplots(1, 2, figsize=(20, 5.3), dpi=400)

    for i, recon in enumerate(recon_types):
        df = pd.read_csv(f"{input_dir}/anova_eta2_reconstruction_{recon}.csv", index_col=0)
        df['Group'], df['ShortName'] = zip(*df.index.map(extract_group_and_shortname))
        df_sorted = df.sort_values(by=['Group', 'ShortName'])
        groups = df_sorted['Group'].tolist()
        group_boundaries = get_group_boundaries(groups)

        ax = axs[i]
        bottom = pd.Series(0, index=df_sorted.index)

        for factor in factors:
            ax.bar(range(len(df_sorted)), df_sorted[factor], bottom=bottom, color=colors[factor], label=factor)
            bottom += df_sorted[factor]

        ax.set_xticks(np.linspace(0, len(df_sorted)-1, 10, dtype=int))
        ax.set_xticklabels(np.linspace(10, 93, 10, dtype=int), fontsize=9)
        ax.tick_params(axis='y', labelsize=12)

        for boundary in group_boundaries:
            ax.axvline(x=boundary - 0.5, color='grey', linestyle='--', linewidth=0.7)

        for group in group_names:
            indices = [j for j, g in enumerate(groups) if g == group]
            if indices:
                center_pos = (indices[0] + indices[-1]) / 2
                if group == 'NGTDM':
                    center_pos += 2.2
                ax.text(center_pos, 1.02, group, ha='center', va='bottom', fontsize=14, fontweight='bold')

        ax.set_ylim(0, 1.1)
        ax.set_ylabel("η²", fontsize=14)
        ax.set_title(f"Reconstruction: {recon}", fontsize=26)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.975, 0.95), fontsize=12, title="Factor", title_fontsize=12)
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(save_path, dpi=400)
    plt.show()


def print_eta2_summary(feature_name: str, csv_dir: str):
    """
    Print per-fold η² values and summary statistics (Mean ± Std, CoV) for a given feature.

    Args:
        feature_name (str): Name of the feature (e.g., 'Skewness', 'Kurtosis')
        csv_dir (str): Path to the directory containing CSVs
    """
    summary_path = f"{csv_dir}/eta2_summary_{feature_name}.csv"
    folds_path = f"{csv_dir}/eta2_folds_{feature_name}.csv"

    df_summary = pd.read_csv(summary_path, index_col=0)
    df_folds = pd.read_csv(folds_path, index_col=0)

    print(f"\n--- η² per Fold for {feature_name} ---")
    print(tabulate(df_folds, headers="keys", tablefmt="grid", floatfmt=".2f"))

    df_summary_rounded = df_summary.copy()
    df_summary_rounded["Mean ± Std"] = (
        df_summary["Mean"].round(2).astype(str) + " ± " + df_summary["Std"].round(2).astype(str)
    )
    df_summary_rounded["CoV"] = df_summary["CoV"].round(2)

    print(f"\n--- Summary Statistics for {feature_name} η² ---")
    print(tabulate(df_summary_rounded[["Mean ± Std", "CoV"]], headers="keys", tablefmt="grid"))



