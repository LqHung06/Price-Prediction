"""EDA for Kaggle shivam2503/diamonds (diamonds.csv)."""
import numpy as np
import os
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(ROOT, "data", "diamonds.csv"))
print(f"SHAPE {df.shape}")
print(df.dtypes.to_string())
print(f"\nduplicated rows: {df.duplicated().sum()}")

print("\n--- MISSING ---")
print(df.isna().sum().to_string())
print(f"rows with any missing: {df.isna().any(axis=1).sum()} / {len(df)}")

print("\n--- NUMERIC describe ---")
print(df.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T.round(2).to_string())
print(f"\nPrice skew: {df['price'].skew():.2f} | log Price skew: {np.log(df['price']).skew():.2f}")
print(f"carat skew: {df['carat'].skew():.2f} | log carat skew: {np.log(df['carat']).skew():.2f}")

print("\n--- ZERO x/y/z rows (bad data) ---")
for c in ["x", "y", "z"]:
    print(f"{c} == 0: {(df[c] == 0).sum()}")
print(f"any of x/y/z == 0: {((df[['x', 'y', 'z']] == 0).any(axis=1)).sum()}")
print(f"depth == 0: {(df['depth'] == 0).sum()} | table == 0: {(df['table'] == 0).sum()}")

print("\n--- CATEGORICAL counts ---")
for c in ["cut", "color", "clarity"]:
    print(f"\n{c}:")
    print(df[c].value_counts(dropna=False).to_string())

print("\n--- CORR with price / log price ---")
num = df.select_dtypes("number")
print(num.corr()["price"].drop("price").round(3).to_string())
df["logPrice"] = np.log(df["price"])
print("\ncorr matrix numeric:")
print(num.corr().round(2).to_string())

print("\n--- IQR OUTLIERS ---")
for c in ["carat", "depth", "table", "x", "y", "z", "price"]:
    s = df[c].dropna()
    q1, q3 = s.quantile(.25), s.quantile(.75)
    iqr = q3 - q1
    n = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()
    print(f"{c:8s} outliers={n:5d} ({100*n/len(s):5.2f}%) range=[{s.min():.2f}, {s.max():.2f}]")

print("\n--- MEAN PRICE by category ---")
for c in ["cut", "color", "clarity"]:
    print(f"\n{c}:")
    print(df.groupby(c, dropna=False)["price"].agg(["mean", "median", "count"]).round(0).to_string())

print("\n--- volume check (x*y*z) ---")
df["volume"] = df["x"] * df["y"] * df["z"]
print("corr volume vs price:", round(df["volume"].corr(df["price"]), 3))
print("corr volume vs carat:", round(df["volume"].corr(df["carat"]), 3))
print("corr carat vs price:", round(df["carat"].corr(df["price"]), 3))
print("corr log carat vs log price:", round(np.log(df["carat"]).corr(df["logPrice"]), 3))
