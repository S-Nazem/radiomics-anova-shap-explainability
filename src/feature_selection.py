import pandas as pd
from scipy.stats import kruskal
from statsmodels.stats.multitest import multipletests
import pingouin as pg
import warnings


def kruskal_fdr_corr_filter(df, target_col="Model", subject_col="PatientName", alpha=0.25, corr_thresh=0.9, verbose=True):
    """
    Perform Kruskal-Wallis test, FDR correction, and remove highly correlated features.

    Returns:
        final_features (list): Features that passed KW test and correlation filtering
        kw_df (pd.DataFrame): Full stats (p-values, FDR, keep/drop)
    """

    # Drop metadata and keep numeric columns only
    exclude = [subject_col, target_col, "Wavelength", "GLbins", "Reconstruction"]
    feature_cols = df.drop(columns=exclude, errors="ignore").select_dtypes("number").columns

    # Kruskal-Wallis test
    pvals = []
    for col in feature_cols:
        groups = [group[col].values for _, group in df.groupby(target_col)]
        _, p = kruskal(*groups)
        pvals.append(p)

    reject, pvals_corrected, _, _ = multipletests(pvals, alpha=alpha, method='fdr_bh')

    kw_df = pd.DataFrame({
        "Feature": feature_cols,
        "p_value": pvals,
        "p_adj": pvals_corrected,
        "Keep": reject
    }).set_index("Feature")

    selected = kw_df[kw_df["Keep"]].index.tolist()

    if verbose:
        print(f"✅ {len(selected)} features selected after BH correction (FDR < {alpha})")

    # Correlation filtering
    df_selected = df[[subject_col, target_col] + selected].copy()
    scores = {f: 1 - kw_df.loc[f, "p_adj"] for f in selected}

    selected_set = set(selected)
    dropped = set()

    for i, f1 in enumerate(selected):
        for f2 in selected[i + 1:]:
            if f1 in dropped or f2 in dropped:
                continue
            try:
                # 👉 suppress warnings only for this call
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")        # hide ALL warnings inside the block
                    r = pg.rm_corr(
                        data=df_selected,
                        x=f1,
                        y=f2,
                        subject=subject_col
                    )["r"].values[0]

                if r > corr_thresh:
                    # Drop lower-scoring feature
                    if scores[f2] > scores[f1]:
                        dropped.add(f1)
                    elif scores[f1] > scores[f2]:
                        dropped.add(f2)
                    else:
                        dropped.add(f2)  # tie-breaker

            except Exception:
                # silently skip correlation failures
                continue

    final_features = list(selected_set - dropped)

    if verbose:
        print(f"✅ {len(final_features)} features retained after correlation filtering")

    return final_features, kw_df
