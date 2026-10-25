import pandas as pd

df = pd.read_csv(
    "data/raw/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv",
    usecols=["loan_amnt", "installment", "term", "int_rate"],
    nrows=300_000,
)
df["term_m"] = df["term"].str.extract(r"(\d+)")[0].astype(float)
df["repay_ratio"] = df["installment"] * df["term_m"] / df["loan_amnt"]

for months, g in df.groupby("term_m"):
    print(f"{int(months)} months: correlation(repay_ratio, int_rate) = {g['repay_ratio'].corr(g['int_rate']):.4f}")