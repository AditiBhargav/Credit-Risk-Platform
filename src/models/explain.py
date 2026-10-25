import json

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

MODEL_PATH = "models/xgb_model.json"
COLUMNS_PATH = "models/feature_columns.json"
META_PATH = "models/model_metadata.json"


def load_artifacts():
    """Load model, feature layout, metadata and a SHAP explainer."""
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    with open(COLUMNS_PATH) as f:
        columns = json.load(f)
    with open(META_PATH) as f:
        meta = json.load(f)
    explainer = shap.TreeExplainer(model)
    return model, columns, meta, explainer


def top_reasons(explainer, X_row, n=5, only_risk_increasing=False):
    """Return the n features that moved this applicant's score the most.

    X_row: a one-row DataFrame in the training column layout.
    only_risk_increasing: True gives adverse-action style reasons.
    """
    values = explainer(X_row).values[0]
    row = X_row.iloc[0]
    order = np.argsort(-np.abs(values))

    reasons = []
    for i in order:
        if only_risk_increasing and values[i] <= 0:
            continue
        reasons.append({
            "feature": X_row.columns[i],
            "value": None if pd.isna(row.iloc[i]) else float(row.iloc[i]),
            "shap": float(values[i]),
            "effect": "increases risk" if values[i] > 0 else "decreases risk",
        })
        if len(reasons) == n:
            break
    return reasons