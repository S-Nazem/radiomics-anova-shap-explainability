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
