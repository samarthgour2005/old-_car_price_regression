import pandas as pd
import os

data_path = r"C:\Users\samar\car_price\old-_car_price_regression\data\train.csv"

if not os.path.exists(data_path):
    print(f"Error: File not found at {data_path}")
    exit(1)

df = pd.read_csv(data_path)

print("--- Data Info ---")
print(df.info())
print("\n--- Numerical Description ---")
print(df.describe())
print("\n--- Categorical Description ---")
print(df.describe(include=['O']))

print("\n--- Unique Values Count ---")
for col in df.columns:
    print(f"{col}: {df[col].nunique()}")

print("\n--- Sample Values (First 5) ---")
print(df.head())

print("\n--- Value Counts for Low Cardinality Columns (<20 unique) ---")
for col in df.columns:
    if df[col].nunique() < 20:
        print(f"\nCol: {col}")
        print(df[col].value_counts())

print("\n--- 'engine' Sample Values (Top 20) ---")
print(df['engine'].value_counts().head(20))

print("\n--- 'transmission' Sample Values (Top 20) ---")
print(df['transmission'].value_counts().head(20))
