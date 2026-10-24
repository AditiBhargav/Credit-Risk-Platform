import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
COST_FN = 5  # assumed cost of approving a loan that defaults
COST_FP = 1  # assumed cost of rejecting a loan that would have been repaid

# ---------- Load data ----------
train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")
X_full, y_full = train.drop(columns="target"), train["target"]
X_test, y_test = test.drop(columns="target"), test["target"].to_numpy()

# Hold out 20% of train for early stopping and threshold selection
X_tr, X_val, y_tr, y_val = train_test_split(
    X_full, y_full, test_size=0.2, stratify=y_full, random_state=SEED
)
y_val = y_val.to_numpy()


def ks_stat(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def report(name, y, p):
    print(f"{name:<18} AUC={roc_auc_score(y, p):.4f}  "
          f"PR-AUC={average_precision_score(y, p):.4f}  KS={ks_stat(y, p):.4f}")


# ---------- Baseline: logistic regression ----------
baseline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=1000))
baseline.fit(X_tr, y_tr)
report("LogReg baseline", y_test, baseline.predict_proba(X_test)[:, 1])

# ---------- XGBoost ----------
model = xgb.XGBClassifier(
    n_estimators=1000, learning_rate=0.05, max_depth=4,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
    eval_metric="auc", early_stopping_rounds=50,
    tree_method="hist", n_jobs=-1, random_state=SEED,
)
model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=50)
print(f"\nTrees used (best iteration): {model.best_iteration}")

val_proba = model.predict_proba(X_val)[:, 1]
test_proba = model.predict_proba(X_test)[:, 1]
report("XGBoost", y_test, test_proba)

# ---------- Calibration check ----------
print(f"\nMean predicted default prob (test): {test_proba.mean():.3f}")
print(f"Actual default rate (test):         {y_test.mean():.3f}")


# ---------- Pick threshold from business cost (on validation, not test) ----------
def cost_at(t, y, p):
    pred = p >= t
    fn = np.sum((~pred) & (y == 1))
    fp = np.sum(pred & (y == 0))
    return fn * COST_FN + fp * COST_FP


thresholds = np.arange(0.05, 0.60, 0.01)
best_t = float(min(thresholds, key=lambda t: cost_at(t, y_val, val_proba)))
print(f"\nBest threshold (cost FN={COST_FN}, FP={COST_FP}): {best_t:.2f}")

pred = test_proba >= best_t
tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
print(f"Recall (defaults caught):       {tp / (tp + fn):.2%}")
print(f"Precision (flagged are bad):    {tp / (tp + fp):.2%}")
print(f"Approval rate:                  {(tn + fn) / len(y_test):.2%}")
print(f"Default rate among approved:    {fn / (tn + fn):.2%}")
print(f"Default rate overall:           {y_test.mean():.2%}")

# ---------- Top features by gain ----------
imp = pd.Series(model.get_booster().get_score(importance_type="gain")).sort_values(ascending=False)
print("\nTop 10 features by gain:")
print(imp.head(10).round(1))

# ---------- Save ----------
Path("models").mkdir(exist_ok=True)
model.save_model("models/xgb_model.json")
metadata = {
    "threshold": best_t,
    "best_iteration": int(model.best_iteration),
    "test_auc": float(roc_auc_score(y_test, test_proba)),
    "test_pr_auc": float(average_precision_score(y_test, test_proba)),
    "test_ks": ks_stat(y_test, test_proba),
    "cost_fn": COST_FN,
    "cost_fp": COST_FP,
}
with open("models/model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
print("\nSaved models/xgb_model.json and models/model_metadata.json")