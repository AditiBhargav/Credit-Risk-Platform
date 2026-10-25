from datetime import date

import numpy as np
import pandas as pd

from src.features.build_features import (
    CATEGORICAL_COLS, EMP_MAP, NUMERIC_COLS, RAW_COLS, encode, engineer_features,
)

MODEL_VERSION = "1.0.0"
EMP_LABEL = {v: k for k, v in EMP_MAP.items()}  # 0 -> "< 1 year", 10 -> "10+ years"

FEATURE_LABELS = {
    "term": "Loan term (months)",
    "fico_avg": "FICO credit score",
    "loan_to_income": "Loan amount relative to income",
    "dti": "Debt-to-income ratio",
    "open_acc": "Number of open credit lines",
    "inq_last_6mths": "Credit inquiries in last 6 months",
    "annual_inc": "Annual income",
    "revol_bal": "Revolving balance",
    "revol_util": "Revolving credit utilization",
    "total_acc": "Total credit accounts",
    "loan_amnt": "Loan amount",
    "emp_length": "Employment length (years)",
    "emp_length_missing": "Employment length not provided",
    "credit_history_years": "Length of credit history (years)",
    "home_ownership_RENT": "Rents home",
    "home_ownership_MORTGAGE": "Has a mortgage",
    "purpose_credit_card": "Loan purpose: credit card",
    "purpose_small_business": "Loan purpose: small business",
}


def to_raw_frame(records):
    """Convert friendly API inputs into the raw layout used in training."""
    this_month = date.today().strftime("%b-%Y")
    skip = ("fico_range_low", "fico_range_high")
    passthrough = [c for c in NUMERIC_COLS + CATEGORICAL_COLS if c not in skip]

    rows = []
    for r in records:
        row = {c: r.get(c) for c in passthrough}
        row["fico_range_low"] = r["fico_score"] - 2   # so fico_avg == fico_score
        row["fico_range_high"] = r["fico_score"] + 2
        row["term"] = f"{r['term']} months"
        row["emp_length"] = EMP_LABEL.get(r["emp_length"]) if r.get("emp_length") is not None else None
        row["earliest_cr_line"] = r["earliest_cr_line"].strftime("%b-%Y")
        row["issue_d"] = this_month  # decision date
        rows.append(row)

    df = pd.DataFrame(rows, columns=RAW_COLS)
    df[NUMERIC_COLS] = df[NUMERIC_COLS].astype(float)  # None -> NaN
    return df


def risk_band(p, threshold):
    if p < 0.5 * threshold:
        return "low"
    if p < threshold:
        return "medium"
    if p < 2 * threshold:
        return "high"
    return "very_high"


def _factors(values, row, columns, n, risk_increasing):
    order = np.argsort(values)[::-1] if risk_increasing else np.argsort(values)
    picked = []
    for j in order:
        v = values[j]
        if (risk_increasing and v <= 0) or (not risk_increasing and v >= 0):
            break
        val = row.iloc[j]
        picked.append({
            "feature": columns[j],
            "label": FEATURE_LABELS.get(columns[j], columns[j].replace("_", " ")),
            "value": None if pd.isna(val) else round(float(val), 4),
            "shap": round(float(v), 4),
        })
        if len(picked) == n:
            break
    return picked


def score(records, artifacts, n_reasons=4):
    """Score a list of applications. Works for 1 or many (SHAP is vectorized)."""
    model, columns, meta, explainer = artifacts
    X = encode(engineer_features(to_raw_frame(records)), columns)

    proba = model.predict_proba(X)[:, 1]
    shap_matrix = explainer(X).values
    t = meta["threshold"]

    results = []
    for i, p in enumerate(proba):
        results.append({
            "default_probability": round(float(p), 4),
            "risk_band": risk_band(p, t),
            "recommendation": "decline" if p >= t else "approve",
            "threshold": t,
            "risk_drivers": _factors(shap_matrix[i], X.iloc[i], columns, n_reasons, True),
            "protective_factors": _factors(shap_matrix[i], X.iloc[i], columns, n_reasons, False),
            "model_version": MODEL_VERSION,
        })
    return results