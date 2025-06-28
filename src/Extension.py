import ast
import numpy as np
import pandas as pd
from skimage.draw import polygon
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.metrics import classification_report
from mpl_toolkits.mplot3d import Axes3D
import xml.etree.ElementTree as ET
import os, pickle, ast
import pydicom
import SimpleITK as sitk
from radiomics import featureextractor
from scipy.stats import kruskal
from statsmodels.stats.multitest import multipletests
import pingouin as pg
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import shap
from matplotlib_venn import venn2
import seaborn as sns


def assign_tumour_ids_by_overlap(df_good, image_shape=(512, 512)):
    all_rows = []
    tumour_id_counter = 0

    for patient_id, df_patient in tqdm(df_good.groupby("PatientID"), desc="Assigning TumourIDs"):
        # Build graph of overlapping nodules
        G = nx.Graph()
        nodules = df_patient["NoduleID"].unique()
        G.add_nodes_from(nodules)

        for z, group_z in df_patient.groupby("Z"):
            masks = {}

            for _, row in group_z.iterrows():
                nid = row["NoduleID"]
                x_coords = ast.literal_eval(row["X_coords"])
                y_coords = ast.literal_eval(row["Y_coords"])

                mask = np.zeros(image_shape, dtype=np.uint8)
                rr, cc = polygon(y_coords, x_coords, shape=image_shape)
                mask[rr, cc] = 1
                masks[nid] = mask

            for nid1 in masks:
                for nid2 in masks:
                    if nid1 >= nid2:
                        continue
                    # Check overlap
                    if np.logical_and(masks[nid1], masks[nid2]).any():
                        G.add_edge(nid1, nid2)

        # Connected components → TumourIDs
        components = list(nx.connected_components(G))
        nodule_to_tumour = {}
        for tumour_idx, component in enumerate(components):
            for nid in component:
                nodule_to_tumour[nid] = tumour_id_counter
            tumour_id_counter += 1

        # Assign TumourID and keep only largest-area contour per tumour per Z
        df_patient["TumourID"] = df_patient["NoduleID"].map(nodule_to_tumour)

        kept_rows = []
        for (tumour_id, z), group in df_patient.groupby(["TumourID", "Z"]):
            def area(row):
                x = ast.literal_eval(row["X_coords"])
                y = ast.literal_eval(row["Y_coords"])
                return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

            largest_row = group.loc[group.apply(area, axis=1).idxmax()]
            kept_rows.append(largest_row)

        all_rows.append(pd.DataFrame(kept_rows))

    # Final DataFrame
    df_final = pd.concat(all_rows, ignore_index=True)
    return df_final


def plot_patient_tumours_3d(df, patient_id, ax=None):
    df_p = df[df["PatientID"] == patient_id]
    tumour_ids = sorted(df_p["TumourID"].unique())
    n_tumours = len(tumour_ids)

    cmap = plt.cm.get_cmap("tab20", n_tumours)
    tid_to_colour = {tid: cmap(i) for i, tid in enumerate(tumour_ids)}

    # Create new figure if no axis is provided
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title(f"3D Tumour Contours – Patient {patient_id}", fontsize=14)

    for tid in tumour_ids:
        grp = df_p[df_p["TumourID"] == tid]
        xs, ys, zs = [], [], []

        for _, row in grp.iterrows():
            x = ast.literal_eval(row["X_coords"])
            y = ast.literal_eval(row["Y_coords"])
            z = [row["Z"]] * len(x)
            xs.extend(x)
            ys.extend(y)
            zs.extend(z)

        ax.scatter(xs, ys, zs, s=10, color=tid_to_colour[tid], label=f"Tumour {tid}")

    # Optional axis labels and view
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=20, azim=135)






def bootstrap_ci(X_train, y_train, X_test, y_test, pipeline, n_iterations=200, random_state=42):
    metrics_list = []

    for i in range(n_iterations):
        X_bs, y_bs = resample(X_train, y_train, stratify=y_train, random_state=random_state + i)
        pipeline.fit(X_bs, y_bs)
        y_pred = pipeline.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics_list.append([
            report["accuracy"],
            report["0"]["recall"],
            report["1"]["recall"],
            report["0"]["f1-score"],
            report["1"]["f1-score"],
        ])

    metrics_array = np.array(metrics_list)
    means = metrics_array.mean(axis=0)
    stds = metrics_array.std(axis=0)
    cis = 1.96 * stds / np.sqrt(n_iterations)
    return means, cis



def get_friendly_feature_names(feature_names):
    base_mapping = {
        'original_glcm_JointEntropy': 'GLCM Joint Entropy',
        'original_glszm_SizeZoneNonUniformity': 'GLSZM Size Zone Non-Uniform',
        'original_shape_MinorAxisLength': 'Shape Minor Axis Length',
        'original_gldm_LargeDependenceHighGrayLevelEmphasis': 'GLDM Large Dep High GLE',
        'original_ngtdm_Coarseness': 'NGTDM Coarseness',
        'original_gldm_DependenceEntropy': 'GLDM Dependence Entropy',
        'original_glszm_LowGrayLevelZoneEmphasis': 'GLSZM Low GLE Zone Emph.',
        'original_glcm_Autocorrelation': 'GLCM Autocorrelation',
        'original_shape_Maximum3DDiameter': 'Shape Maximum 3D Diameter',
        'original_glrlm_LongRunLowGrayLevelEmphasis': 'GLRLM Long Run Low GLE'
    }

    # Fallback: transform raw names to human-readable
    def fallback(name):
        name = name.replace("original_", "")
        name = name.replace("_", " ")
        name = name.replace("firstorder", "FO")
        name = name.replace("glcm", "GLCM")
        name = name.replace("glszm", "GLSZM")
        name = name.replace("gldm", "GLDM")
        name = name.replace("glrlm", "GLRLM")
        name = name.replace("ngtdm", "NGTDM")
        name = name.replace("shape", "Shape")
        return name.strip().title()

    return [base_mapping.get(f, fallback(f)) for f in feature_names]



def parse_xml_rois(xml_path, ns, patient_id):
    roi_data = []
    try:
        root = ET.parse(xml_path).getroot()
    except Exception as e:
        print(f"❌ Failed parsing XML for {patient_id}: {e}")
        return roi_data, False

    for reading_session in root.findall(".//lidc:readingSession", ns):
        for nodule in reading_session.findall("lidc:unblindedReadNodule", ns):
            nodule_id = nodule.findtext("lidc:noduleID", default="Unknown", namespaces=ns)

            for roi in nodule.findall("lidc:roi", ns):
                z_elem = roi.find("lidc:imageZposition", ns)
                if z_elem is None:
                    continue

                x_coords = [int(e.findtext("lidc:xCoord", default="0", namespaces=ns)) for e in roi.findall("lidc:edgeMap", ns)]
                y_coords = [int(e.findtext("lidc:yCoord", default="0", namespaces=ns)) for e in roi.findall("lidc:edgeMap", ns)]

                if not x_coords or not y_coords:
                    print(f"⚠️ Skipping empty ROI for {patient_id}, z={z_elem.text}")
                    return [], False

                roi_data.append({
                    "PatientID": patient_id,
                    "Z": float(z_elem.text),
                    "X_coords": x_coords,
                    "Y_coords": y_coords,
                    "NoduleID": nodule_id
                })
    return roi_data, bool(roi_data)





def build_3d_volumes_from_rois(df_good_clean, dicom_root, output_path="volume_results.pkl"):
    """
    Builds 3D CT volumes and binary ROI masks for each tumour from DICOM + polygon ROI.
    Saves the result as a list of dictionaries in a pickle file.

    Parameters:
    - df_good_clean: DataFrame with PatientID, TumourID, X/Y_coords, Z, etc.
    - dicom_root: Root folder of LIDC DICOM dataset.
    - output_path: Path to save pickle of volume_results.
    """

    volume_results = []
    grouped = df_good_clean.groupby(["PatientID", "TumourID"])

    for (patient_id, tumour_id), group in tqdm(grouped, desc="Building 3D Volumes"):
        try:
            group_sorted = group.sort_values("Z")
            unique_z = sorted(group_sorted["Z"].unique())
            z_to_index = {z: i for i, z in enumerate(unique_z)}

            vol_shape = (len(unique_z), 512, 512)
            image_3d = np.zeros(vol_shape, dtype=np.int16)
            mask_3d = np.zeros(vol_shape, dtype=np.uint8)

            dicom_slices = {}
            for root, _, files in os.walk(os.path.join(dicom_root, patient_id)):
                for file in files:
                    if file.endswith(".dcm"):
                        path = os.path.join(root, file)
                        try:
                            dcm = pydicom.dcmread(path)
                            if hasattr(dcm, "SliceLocation"):
                                dicom_slices[dcm.SliceLocation] = dcm
                        except Exception:
                            continue

            if not dicom_slices:
                print(f"⚠️ No DICOM slices found for {patient_id}")
                continue

            for _, row in group_sorted.iterrows():
                z = row["Z"]
                z_idx = z_to_index[z]
                x_coords = ast.literal_eval(row["X_coords"])
                y_coords = ast.literal_eval(row["Y_coords"])

                closest_z = min(dicom_slices.keys(), key=lambda zz: abs(zz - z))
                dcm = dicom_slices[closest_z]
                slice_array = dcm.pixel_array

                rr, cc = polygon(y_coords, x_coords, shape=slice_array.shape)
                image_3d[z_idx, rr, cc] = slice_array[rr, cc]
                mask_3d[z_idx, rr, cc] = 1

            sitk_image = sitk.GetImageFromArray(image_3d.astype(np.float32))
            sitk_mask = sitk.GetImageFromArray(mask_3d)
            sitk_mask.CopyInformation(sitk_image)

            volume_results.append({
                "PatientID": patient_id,
                "TumourID": tumour_id,
                "sitk_image": sitk_image,
                "sitk_mask": sitk_mask,
                "NumSlices": len(unique_z)
            })

        except Exception as e:
            print(f"❌ Error for {patient_id}, tumour {tumour_id}: {e}")

    with open(output_path, "wb") as f:
        pickle.dump(volume_results, f)

    print(f"✅ Saved {len(volume_results)} tumour volumes to {output_path}")



    
def extract_radiomic_features(volume_results, extractor=None, yaml_path=None):
    """
    Extract radiomic features from volume_results using PyRadiomics.

    Parameters:
    - volume_results: list of dicts with keys 'PatientID', 'TumourID', 'sitk_image', 'sitk_mask', 'NumSlices'
    - extractor: optional pre-initialized RadiomicsFeatureExtractor
    - yaml_path: optional path to .yaml file for configuration

    Returns:
    - DataFrame with radiomic features and metadata
    """

    if extractor is None:
        extractor = featureextractor.RadiomicsFeatureExtractor(yaml_path) if yaml_path else featureextractor.RadiomicsFeatureExtractor()

    features_list = []

    for item in tqdm(volume_results, desc="Extracting radiomics"):
        try:
            patient_id = item["PatientID"]
            tumour_id = item["TumourID"]
            sitk_image = item["sitk_image"]
            sitk_mask = item["sitk_mask"]
            num_slices = item["NumSlices"]

            features = extractor.execute(sitk_image, sitk_mask)
            features_clean = {k: v for k, v in features.items() if not k.startswith("diagnostics")}

            features_clean["PatientID"] = patient_id
            features_clean["TumourID"] = tumour_id
            features_clean["NumSlices"] = num_slices

            features_list.append(features_clean)

        except Exception as e:
            print(f"❌ Error for Patient {patient_id}, Tumour {tumour_id}: {e}")

    return pd.DataFrame(features_list)




def merge_radiomics_with_labels(df_radiomics, df_good_clean):
    """
    Merge extracted radiomics with tumour-level binary labels.
    Ensures no duplicate or conflicting labels and coerces features to numeric.

    Returns:
    - df_radiomics_labeled: Final merged DataFrame
    """

    # Step 1: One label per tumour
    df_labels = df_good_clean[["PatientID", "TumourID", "Label"]].drop_duplicates()

    # Step 2: Check for conflicting labels
    conflicts = df_labels.groupby(["PatientID", "TumourID"])["Label"].nunique()
    if (conflicts > 1).any():
        print("❌ Warning: Multiple labels found for some (PatientID, TumourID) combinations.")

    # Step 3: Remove true duplicates
    df_labels = df_labels.drop_duplicates(subset=["PatientID", "TumourID"])

    # Step 4: Merge
    df_radiomics_labeled = df_radiomics.merge(df_labels, on=["PatientID", "TumourID"], how="left")

    # Step 5: Check for missing labels
    missing = df_radiomics_labeled["Label"].isna().sum()
    if missing > 0:
        print(f"⚠️ {missing} tumours have missing labels!")

    # Step 6: Convert radiomic feature columns to numeric
    feature_cols = df_radiomics_labeled.columns.difference(["PatientID", "TumourID", "NumSlices", "Label"])
    df_radiomics_labeled[feature_cols] = df_radiomics_labeled[feature_cols].apply(pd.to_numeric, errors="coerce")

    return df_radiomics_labeled




def select_features_kw_fdr(df, alpha=0.25):

    X = df.drop(columns=["Label", "PatientID", "TumourID", "NumSlices"])
    y = df["Label"]

    p_values = [kruskal(X[col][y == 0], X[col][y == 1]).pvalue for col in X.columns]
    fdr_pass, pvals_corrected, _, _ = multipletests(p_values, alpha=alpha, method="fdr_bh")

    selected = X.columns[fdr_pass]
    X_selected = X[selected]

    print(f"✅ Selected {len(selected)} features out of {X.shape[1]} after FDR correction.")
    return X_selected, selected, pvals_corrected




def remove_correlated_features(df, selected_features, pvals_corrected, subject_col="PatientID", threshold=0.9):

    df_corr = df[[subject_col] + list(selected_features)].copy()
    scores = {feature: 1 - p for feature, p in zip(selected_features, pvals_corrected) if feature in selected_features}

    dropped = set()
    for i, key1 in enumerate(selected_features):
        for key2 in selected_features[i + 1:]:
            if key1 in dropped or key2 in dropped:
                continue

            try:
                result = pg.rm_corr(data=df_corr, x=key1, y=key2, subject=subject_col)
                r = result["r"].values[0]

                if r > threshold:
                    s1, s2 = scores.get(key1, 0), scores.get(key2, 0)
                    if s2 > s1:
                        dropped.add(key1)
                    else:
                        dropped.add(key2)

            except Exception as e:
                print(f"⚠️ Correlation failed for {key1} & {key2}: {e}")

    final_features = list(set(selected_features) - dropped)
    print(f"✅ {len(final_features)} features selected after rm_corr filtering")
    return final_features


def run_rfecv(X, y, output_path=None):
    rf = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        max_depth=None,
        class_weight='balanced',
        n_jobs=-1
    )

    rfecv = RFECV(
        estimator=rf,
        step=1,
        cv=StratifiedKFold(5),
        scoring="accuracy",
        n_jobs=-1
    )

    rfecv.fit(X, y)
    selected_features = X.columns[rfecv.support_].tolist()

    print("✅ Optimal number of features:", rfecv.n_features_)
    print("✅ Selected features:", selected_features)

    if output_path:
        plt.figure(figsize=(10, 5))
        plt.plot(
            range(1, len(rfecv.cv_results_['mean_test_score']) + 1),
            rfecv.cv_results_['mean_test_score'],
            marker='o'
        )
        plt.xlabel("Number of features selected", fontsize=16)
        plt.ylabel("CV Accuracy", fontsize=16)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

    return selected_features


def run_boruta(X, y, alpha=0.05, verbose=2, max_iter=100):
    rf = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        max_depth=None,
        class_weight='balanced',
        n_jobs=-1
    )

    selector = BorutaPy(
        estimator=rf,
        n_estimators=500,
        verbose=verbose,
        random_state=42,
        max_iter=max_iter,
        alpha=alpha,
        perc=50
    )

    selector.fit(X.values, y.values)

    selected = X.columns[selector.support_].tolist()

    print("✅ Selected features using Boruta:", selected)
    return selected


def train_test_with_smote(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test, X_train_resampled, y_train_resampled

def evaluate_pipeline(X_train, y_train, X_test, y_test, classifier=None, report_name=None):
    if classifier is None:
        classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("clf", classifier)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    auc_score = roc_auc_score(y_test, y_proba)
    clf_report = classification_report(y_test, y_pred, digits=3)
    clf_dict = classification_report(y_test, y_pred, output_dict=True)

    print(f"📊 {report_name or 'Model'} AUC:", auc_score)
    print(clf_report)

    return clf_dict

def cross_val_auc(X, y, classifier=None):
    if classifier is None:
        classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("clf", classifier)
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")
    print("Cross-validated AUCs:", scores)
    print("Mean AUC:", scores.mean())
    return scores



from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression


def fit_and_evaluate_model(X_train, y_train, X_test, y_test, model, label="Model"):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, digits=3, output_dict=True)

    print(f"📊 {label} AUC:", auc)
    print(classification_report(y_test, y_pred, digits=3))

    return report


def build_rf_pipeline():
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])


def build_xgb_model():
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
        n_estimators=300,
        max_depth=4,
        learning_rate=0.1
    )

def build_logreg_pipeline():
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ))
    ])

from sklearn.svm import SVC

def build_svm_pipeline():
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", SVC(
            probability=True,
            class_weight="balanced",
            random_state=42
        ))
    ])



def plot_bootstrap_metrics_bars(
    bar_sets: list,
    metrics: list,
    labels: list,
    colors: list,
    title: str,
    ax=None,
    legend=True,
    save_path=None,
    dpi=300
):
    """
    Plot grouped bar chart with error bars and SMOTE vs No-SMOTE background shading.

    Parameters
    ----------
    bar_sets : list of (means, errors)
        Each element is a tuple of metric means and corresponding CI/error bars.
    metrics : list of str
        Metric names for x-axis (e.g. ["Accuracy", "Recall (Class 0)", ...])
    labels : list of str
        Labels for each bar (must match length of bar_sets)
    colors : list of str
        Color for each bar
    title : str
        Title of the plot (shown above or optionally removed)
    ax : matplotlib.axes.Axes or None
        If provided, plots on this axis. If None, creates a new figure.
    legend : bool
        Whether to include a legend.
    save_path : str or None
        If set, saves the figure to this path.
    dpi : int
        Dots-per-inch for saving.
    """
    x = np.arange(len(metrics))
    width = 0.10
    group_width = width * len(bar_sets)

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6))

    # === Background shading for SMOTE vs No SMOTE ===
    for xi in x:
        left = xi - group_width / 2
        ax.axvspan(left, left + 4*width, color="lightgrey", alpha=0.7, zorder=0)
        ax.axvspan(left + 4*width, left + 8*width, color="lightblue", alpha=0.7, zorder=0)

    # === Plot bars ===
    for i, ((means, errors), color, label) in enumerate(zip(bar_sets, colors, labels)):
        offset = (i - (len(bar_sets)-1)/2) * width
        ax.bar(
            x + offset, means,
            width=width,
            yerr=errors,
            capsize=6,
            linewidth=1,
            edgecolor="black",
            color=color,
            label=label
        )

    # === Formatting ===
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=14)
    ax.set_ylabel("Score", fontsize=14)
    ax.set_ylim(0, 1.0)
    ax.set_title(title, fontsize=16)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.tick_params(axis="y", labelsize=12)

    if legend:
        ax.legend(ncol=4, fontsize=13, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.23))

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi)

    if ax is None:
        plt.show()




def plot_shap_summary_and_bar(X_train, y_train, save_prefix, get_friendly_feature_names):
    """
    Compute SHAP values and plot beeswarm and bar plots with friendly feature names.

    Parameters:
        X_train (pd.DataFrame): Feature matrix.
        y_train (pd.Series): Target vector.
        save_prefix (str): Filename prefix for saving plots.
        get_friendly_feature_names (callable): Maps column names to friendly names.
    """

    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])

    pipeline.fit(X_train, y_train)
    rf = pipeline.named_steps["clf"]
    scaler = pipeline.named_steps["scale"]
    X_scaled = scaler.transform(X_train)

    # SHAP explanation
    explainer = shap.TreeExplainer(rf)
    shap_vals = explainer.shap_values(X_scaled)
    shap_vals_class1 = shap_vals[:, :, 1]

    # Friendly names
    friendly_names = get_friendly_feature_names(X_train.columns)

    # Beeswarm plot
    shap.summary_plot(
        shap_vals_class1,
        X_scaled,
        feature_names=friendly_names,
        plot_size=(18, 8),
        max_display=10,
        show=False
    )
    plt.xlabel("SHAP value (impact on model output)", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(f"../plots_extension/shap_plots/shap_beeswarm_{save_prefix}.png", dpi=400)
    plt.show()

    # Bar plot (mean(|SHAP|))
    mean_abs_shap = np.abs(shap_vals_class1).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-10:][::-1]
    top_features = [friendly_names[i] for i in top_indices]
    top_values = mean_abs_shap[top_indices]

    plt.figure(figsize=(18, 8), dpi=400)
    plt.barh(range(len(top_features)), top_values, color="#d62728")
    plt.yticks(range(len(top_features)), top_features, fontsize=18)
    plt.xticks(fontsize=16)
    plt.xlabel("mean(|SHAP value|)", fontsize=18)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(f"../plots_extension/shap_plots/shap_bar_{save_prefix}.png", dpi=400)
    plt.show()




def plot_feature_selection_venn(rfe_features, boruta_features, save_path):

    rfe_set = set(rfe_features)
    boruta_set = set(boruta_features)

    plt.figure(figsize=(7, 7))
    v = venn2([rfe_set, boruta_set], set_labels=("RFE", "Boruta"), set_colors=("#56B4E9", "#009E73"), alpha=0.6)

    for text in v.set_labels:
        text.set_fontsize(14)
        text.set_fontweight("bold")

    for text in v.subset_labels:
        if text:
            text.set_fontsize(16)

    for patch in v.patches:
        if patch:
            patch.set_edgecolor("black")
            patch.set_linewidth(1.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=400)
    plt.show()



def compute_shap_rank_stability(X, y, friendly_names, n_seeds=1000, top_k=10):

    rank_dict = {}
    for seed in range(n_seeds):
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1
            ))
        ])
        pipe.fit(X, y)
        rf = pipe.named_steps["clf"]
        scaler = pipe.named_steps["scale"]
        X_scaled = scaler.transform(X)

        explainer = shap.TreeExplainer(rf)
        shap_vals = explainer.shap_values(X_scaled)
        shap_vals_class1 = shap_vals[:, :, 1]

        mean_abs_shap = np.abs(shap_vals_class1).mean(axis=0)
        ranks = pd.Series((-mean_abs_shap).argsort().argsort() + 1, index=X.columns)
        rank_dict[f"seed_{seed}"] = ranks

    shap_ranks = pd.concat(rank_dict.values(), axis=1)
    shap_ranks.columns = rank_dict.keys()

    friendly_names_dict = dict(zip(X.columns, friendly_names))
    shap_ranks_named = shap_ranks.rename(index=friendly_names_dict)

    top_features = shap_ranks_named.mean(axis=1).nsmallest(top_k).index
    shap_ranks_top10 = shap_ranks_named.loc[top_features[::-1]]

    return shap_ranks_top10



def plot_shap_rank_stability(shap_ranks_top10, save_path, xlim=(0, 25)):

    df_plot = shap_ranks_top10.T.melt(var_name="Feature", value_name="Rank")

    plt.figure(figsize=(12, 5.5))
    sns.set(style="whitegrid", font_scale=1.3)

    boxprops = dict(linewidth=1.5)
    medianprops = dict(linewidth=2, color='black')
    whiskerprops = dict(linewidth=1.5)
    capprops = dict(linewidth=1.5)
    palette = sns.color_palette("Set3", n_colors=len(shap_ranks_top10))

    sns.boxplot(
        data=df_plot,
        x="Rank", y="Feature",
        palette=palette,
        linewidth=1.5,
        boxprops=boxprops,
        medianprops=medianprops,
        whiskerprops=whiskerprops,
        capprops=capprops
    )

    plt.xlabel("SHAP Rank", fontsize=16)
    plt.ylabel("Feature", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(*xlim)

    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=400)
    plt.show()



def print_metric_table(selectors, sampling, models, metrics, model_names):
    for selector in selectors:
        print(f"\n==================== {selector.upper()} RESULTS ====================\n")
        for sample in sampling:
            print(f"--- With {sample.upper()} ---\n")
            for model in models:
                key_means = f"means_{model}_{selector}_{sample}"
                key_ci = f"ci_{model}_{selector}_{sample}"
                means = globals()[key_means]
                ci = globals()[key_ci]

                print(f"{model_names[model]}:")
                for metric, mean, err in zip(metrics, means, ci):
                    print(f"  {metric:15}: {mean:.3f} ± {err:.3f}")
                print("")