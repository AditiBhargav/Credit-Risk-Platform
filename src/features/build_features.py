import numpy as np
import pandas as pd

GOOD = "Fully Paid"
BAD = "Charged Off"

# Columns a lender knows BEFORE approving the loan
NUMERIC_COLS = [
    "loan_amnt", "annual_inc", "dti", "delinq_2yrs",
    "fico_range_low", "fico_range_high", "inq_last_6mths", "open_acc",
    "pub_rec", "revol_bal", "revol_util", "total_acc", "mort_acc",
    "pub_rec_bankruptcies",
]
CATEGORICAL_COLS = ["home_ownership", "verification_status", "purpose", "application_type"]
RAW_COLS = NUMERIC_COLS + CATEGORICAL_COLS + ["term", "emp_length", "earliest_cr_line", "issue_d"]

EMP_MAP = {"< 1 year": 0, "10+ years": 10}
EMP_MAP.update({f"{i} year{'s' if i > 1 else ''}": i for i in range(1, 10)})


def load_and_filter(path, nrows=None):
    """Load only the columns we need, keep resolved loans, create the target."""
    df = pd.read_csv(path, usecols=RAW_COLS + ["loan_status"], nrows=nrows, low_memory=False)
    df = df[df["loan_status"].isin([GOOD, BAD])].copy()
    df["target"] = (df["loan_status"] == BAD).astype(int)
    return df.drop(columns="loan_status")


def engineer_features(df):
    """Clean columns and create new ones. Same function is used in training AND the API."""
    out = df.copy()

    # " 36 months" -> 36
    out["term"] = out["term"].astype(str).str.extract(r"(\d+)")[0].astype(float)

    # "10+ years" -> 10, missing stays missing (and we flag it)
    out["emp_length_missing"] = out["emp_length"].isna().astype(int)
    out["emp_length"] = out["emp_length"].map(EMP_MAP)

    # Dates -> credit history length in years
    out["issue_d"] = pd.to_datetime(out["issue_d"], format="%b-%Y")
    earliest = pd.to_datetime(out["earliest_cr_line"], format="%b-%Y")
    out["credit_history_years"] = (out["issue_d"] - earliest).dt.days / 365.25
    out = out.drop(columns="earliest_cr_line")

    # Two FICO columns say almost the same thing -> one average
    out["fico_avg"] = (out["fico_range_low"] + out["fico_range_high"]) / 2
    out = out.drop(columns=["fico_range_low", "fico_range_high"])

    # Ratios: how big is the loan relative to what the person earns?
    income = out["annual_inc"].replace(0, np.nan)
    out["loan_to_income"] = out["loan_amnt"] / income

    return out


def encode(df, feature_columns=None):
    """One-hot encode categories. Pass feature_columns to force the exact training layout."""
    X = df.drop(columns=[c for c in ["target", "issue_d"] if c in df.columns])
    X = pd.get_dummies(X, columns=CATEGORICAL_COLS, dtype=int)
    if feature_columns is not None:
        X = X.reindex(columns=feature_columns, fill_value=0)
    return X