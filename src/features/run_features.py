import json
from pathlib import Path

from src.features.build_features import encode, engineer_features, load_and_filter

RAW_PATH = "data/raw/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv"

df = load_and_filter(RAW_PATH, nrows=200_000)
df = engineer_features(df)
print(f"Resolved loans: {len(df)} | default rate: {df['target'].mean():.2%}")

# Time-based split: train on the oldest 80%, test on the newest 20%
cutoff = df["issue_d"].quantile(0.8)
train_df = df[df["issue_d"] <= cutoff]
test_df = df[df["issue_d"] > cutoff]
print(f"Train: {len(train_df)} (up to {cutoff:%b %Y}) | Test: {len(test_df)}")
print(f"Train default rate: {train_df['target'].mean():.2%} | Test: {test_df['target'].mean():.2%}")

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