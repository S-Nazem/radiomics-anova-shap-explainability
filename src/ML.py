from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
import numpy as np
import pandas as pd



def compute_featurewise_model_scores(df, feature_cols, target_col="Model", cv=None):
    """
    Evaluate each feature individually using 5-fold CV across multiple classifiers.

    Returns a sorted DataFrame with accuracy scores.
    """

    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=42))
    }

    results = []
    for i, feature in enumerate(feature_cols, start=1):
        row = {"ID": i, "Feature": feature}
        for name, model in models.items():
            X = df[[feature]]
            y = df[target_col]
            scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
            row[name] = scores.mean()
        row["Average"] = np.mean([row[m] for m in models])
        results.append(row)

    df_scores = pd.DataFrame(results)
    return df_scores.sort_values("Average", ascending=False).reset_index(drop=True)
