"""EDA for Kaggle raviiyadavv/house-price-predecition (dataset_2.csv)."""
import numpy as np
import os
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(ROOT, "notebooks", "dataset_2.csv"))
print(f"SHAPE {df.shape}")
print(df.dtypes.to_string())
print(f"\nduplicated rows: {df.duplicated().sum()}")

print("\n--- MISSING ---")
print(df.isna().sum().to_string())
print(f"rows with any missing: {df.isna().any(axis=1).sum()} / {len(df)}")

print("\n--- NUMERIC describe ---")
print(df.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T.round(2).to_string())
print(f"\nPrice skew: {df['Price'].skew():.2f} | log Price skew: {np.log(df['Price']).skew():.2f}")
print(f"Area skew: {df['Area_SqFt'].skew():.2f} | log Area skew: {np.log(df['Area_SqFt']).skew():.2f}")
print(f"Rooms values: {sorted(df['Rooms'].dropna().unique())}")
print(f"Build_Year range: {df['Build_Year'].min()}-{df['Build_Year'].max()}")

print("\n--- CATEGORICAL counts ---")
for c in ["Location", "Street_Type", "Furnishing", "Property_Type", "Has_Pool"]:
    print(f"\n{c}:")
    print(df[c].value_counts(dropna=False).to_string())

print("\n--- CORR with Price / log Price ---")
num = df.select_dtypes("number")
print(num.corr()["Price"].drop("Price").round(3).to_string())
df["logPrice"] = np.log(df["Price"])
print("\ncorr with logPrice:")
print(num.corr()["Price"].drop("Price").round(3).to_string())
print("\ncorr matrix numeric:")
print(num.corr().round(2).to_string())

print("\n--- IQR OUTLIERS ---")
for c in ["Area_SqFt", "Rooms", "Build_Year", "Price"]:
    s = df[c].dropna()
    q1, q3 = s.quantile(.25), s.quantile(.75)
    iqr = q3 - q1
    n = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()
    print(f"{c:12s} outliers={n:4d} ({100*n/len(s):4.1f}%) range=[{s.min():.1f}, {s.max():.1f}]")

print("\n--- MEAN PRICE by category ---")
for c in ["Location", "Street_Type", "Furnishing", "Property_Type", "Has_Pool"]:
    print(f"\n{c}:")
    print(df.groupby(c, dropna=False)["Price"].agg(["mean", "median", "count"]).round(0).to_string())
print("\n--- MEAN PRICE by Rooms ---")
print(df.groupby("Rooms")["Price"].agg(["mean", "median", "count"]).round(0).to_string())
print("\n--- price per sqft stats ---")
df["pps"] = df["Price"] / df["Area_SqFt"]
print(df["pps"].describe(percentiles=[.01, .99]).round(1).to_string())
print("corr Area vs Price:", round(df["Area_SqFt"].corr(df["Price"]), 3))
