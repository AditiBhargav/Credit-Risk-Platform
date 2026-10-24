import pandas as pd

# Load a sample first - the full file is 648MB and ~2.2M rows,
# too heavy to explore comfortably on a laptop
df = pd.read_csv("data/raw/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv", nrows=200000, low_memory=False)

print(f"Shape: {df.shape}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
print("\n--- Column names (first 20) ---")
print(list(df.columns[:20]))

print("\n--- Data types ---")
print(df.dtypes.value_counts())

print("\n--- First 3 rows of key columns ---")
print(df[["loan_amnt", "term", "int_rate", "grade", "annual_inc", "loan_status"]].head(3))
print("\n--- Loan status value counts ---")
print(df["loan_status"].value_counts())

# Keep only loans with a known final outcome
resolved_statuses = ["Fully Paid", "Charged Off"]
df_clean = df[df["loan_status"].isin(resolved_statuses)].copy()

# Create the target column: 1 = defaulted, 0 = repaid
df_clean["target"] = (df_clean["loan_status"] == "Charged Off").astype(int)

print(f"\nRows before filtering: {len(df)}")
print(f"Rows after filtering: {len(df_clean)}")
print(f"\nDefault rate: {df_clean['target'].mean():.2%}")

print("\n--- Columns with most missing values (top 20) ---")
missing = df_clean.isnull().sum().sort_values(ascending=False)
missing_pct = (missing / len(df_clean) * 100).round(1)
print(pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct}).head(20))

# These columns only exist AFTER a loan's outcome is known -
# using them would let the model "cheat" by seeing the future
leaky_columns = [
    "total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int",
    "total_rec_late_fee", "recoveries", "collection_recovery_fee",
    "last_pymnt_amnt", "last_pymnt_d", "last_credit_pull_d",
    "next_pymnt_d", "loan_status"  # loan_status itself, since target is derived from it
]

leaky_present = [col for col in leaky_columns if col in df_clean.columns]
print(f"\nLeaky columns found in dataset: {leaky_present}")

print("\n--- Default rate by loan grade ---")
print(df_clean.groupby("grade")["target"].agg(["mean", "count"]).round(3))

print("\n--- Default rate by home ownership ---")
print(df_clean.groupby("home_ownership")["target"].agg(["mean", "count"]).round(3))

# Save the cleaned sample for the next phase
df_clean.to_csv("data/processed/eda_sample.csv", index=False)
print("\nSaved cleaned sample to data/processed/eda_sample.csv")