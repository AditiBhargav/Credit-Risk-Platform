import json
from pathlib import Path

from src.features.build_features import encode, engineer_features, load_and_filter

RAW_PATH = "data/raw/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv"

# Load the WHOLE file (we only read ~25 columns, so it fits in memory)
df = load_and_filter(RAW_PATH)
df = engineer_features(df)

# Keep loans issued 2012-2015, where outcomes are mostly resolved
df = df[(df["issue_d"] >= "2012-01-01") & (df["issue_d"] <= "2015-12-31")]

# Explicit time split: train on 2012-2014, test on 2015
train_df = df[df["issue_d"] < "2015-01-01"]
test_df = df[df["issue_d"] >= "2015-01-01"]

assert len(train_df) > 0 and len(test_df) > 0, "Empty split - check the date ranges"

print(f"Resolved loans (2012-2015): {len(df)}")
print(f"Train: {len(train_df)} | default rate {train_df['target'].mean():.2%}")
print(f"Test:  {len(test_df)} | default rate {test_df['target'].mean():.2%}")

X_train = encode(train_df)
feature_columns = list(X_train.columns)
X_test = encode(test_df, feature_columns)

Path("models").mkdir(exist_ok=True)
with open("models/feature_columns.json", "w") as f:
    json.dump(feature_columns, f, indent=2)

X_train.assign(target=train_df["target"]).to_csv("data/processed/train.csv", index=False)
X_test.assign(target=test_df["target"]).to_csv("data/processed/test.csv", index=False)

print(f"Features: {len(feature_columns)}")
print("Saved train.csv, test.csv and models/feature_columns.json")