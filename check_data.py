import pandas as pd

df = pd.read_csv(
	"data/raw/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv",
	nrows=1000,
)

print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)[:10]}")
print(df["loan_status"].value_counts())