import numpy as np
import pandas as pd

from src.features.build_features import encode, engineer_features


def sample_df():
    return pd.DataFrame({
        "loan_amnt": [10000, 5000], "installment": [300.0, 150.0],
        "annual_inc": [50000.0, 0.0], "dti": [15.0, 20.0],
        "delinq_2yrs": [0, 1], "fico_range_low": [700, 660],
        "fico_range_high": [704, 664], "inq_last_6mths": [0, 2],
        "open_acc": [8, 5], "pub_rec": [0, 0], "revol_bal": [5000, 2000],
        "revol_util": [30.0, 60.0], "total_acc": [20, 10],
        "mort_acc": [1, None], "pub_rec_bankruptcies": [0, 0],
        "home_ownership": ["RENT", "OWN"],
        "verification_status": ["Verified", "Not Verified"],
        "purpose": ["car", "other"],
        "application_type": ["Individual", "Individual"],
        "term": [" 36 months", " 60 months"],
        "emp_length": ["10+ years", None],
        "earliest_cr_line": ["Jan-2005", "Jan-2010"],
        "issue_d": ["Jan-2015", "Jan-2015"],
    })


def test_cleaning():
    out = engineer_features(sample_df())
    assert list(out["term"]) == [36.0, 60.0]
    assert out["emp_length"].iloc[0] == 10
    assert list(out["emp_length_missing"]) == [0, 1]
    assert abs(out["credit_history_years"].iloc[0] - 10) < 0.1


def test_zero_income_gives_nan_not_inf():
    out = engineer_features(sample_df())
    assert np.isnan(out["loan_to_income"].iloc[1])


def test_single_row_matches_training_layout():
    train_cols = list(encode(engineer_features(sample_df())).columns)
    one = engineer_features(sample_df().iloc[[1]])
    assert list(encode(one, train_cols).columns) == train_cols