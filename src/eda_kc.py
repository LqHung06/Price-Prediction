"""EDA for Kaggle harlfoxem/housesalesprediction (kc_house_data.csv)."""
import numpy as np
import os
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(ROOT, "notebooks", "kc_house_data.csv"))
print(f"SHAPE {df.shape}")
print(df.dtypes.to_string())
print(f"\nduplicated rows: {df.duplicated().sum()} | duplicated id: {df['id'].duplicated().sum()}")

print("\n--- MISSING ---")
print(df.isna().sum().to_string())
print(f"rows with any missing: {df.isna().any(axis=1).sum()} / {len(df)}")

print("\n--- DATE range ---")
print(df["date"].min(), "->", df["date"].max())
print("sale year counts:"); print(pd.to_datetime(df["date"]).dt.year.value_counts().to_string())

print("\n--- NUMERIC describe ---")
print(df.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T.round(2).to_string())
print(f"\nPrice skew: {df['price'].skew():.2f} | log Price skew: {np.log(df['price']).skew():.2f}")
print(f"sqft_living skew: {df['sqft_living'].skew():.2f} | log skew: {np.log(df['sqft_living']).skew():.2f}")
print(f"sqft_lot skew: {df['sqft_lot'].skew():.2f} | log skew: {np.log(df['sqft_lot']).skew():.2f}")
print(f"yr_built range: {df['yr_built'].min()}-{df['yr_built'].max()}")
print(f"yr_renovated==0 (never renovated): {(df['yr_renovated']==0).sum()} / {len(df)}")
print(f"bedrooms values: {sorted(df['bedrooms'].unique())}")
print(f"waterfront values: {sorted(df['waterfront'].unique())} | waterfront=1: {(df['waterfront']==1).sum()}")
print(f"zipcode n_unique: {df['zipcode'].nunique()}")

print("\n--- CORR with price / log price ---")
num = df.select_dtypes("number")
print(num.corr()["price"].drop("price").sort_values(key=abs, ascending=False).round(3).to_string())
print("\ncorr matrix (key numerics):")
cols = ["price", "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
        "waterfront", "view", "condition", "grade", "sqft_above", "sqft_basement",
        "yr_built", "lat", "long", "sqft_living15", "sqft_lot15"]
print(num[cols].corr().round(2).to_string())

print("\n--- IQR OUTLIERS ---")
for c in ["price", "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "sqft_above",
          "sqft_basement", "yr_built", "lat", "long", "sqft_living15", "sqft_lot15"]:
    s = df[c].dropna()
    q1, q3 = s.quantile(.25), s.quantile(.75)
    iqr = q3 - q1
    n = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()
    print(f"{c:16s} outliers={n:5d} ({100*n/len(s):4.1f}%) range=[{s.min()}, {s.max()}]")

print("\n--- ORDINAL / BINARY distributions ---")
for c in ["bedrooms", "bathrooms", "floors", "view", "condition", "grade", "waterfront"]:
    print(f"\n{c}:")
    print(df[c].value_counts().sort_index().to_string())

print("\n--- MEAN PRICE by category ---")
for c in ["waterfront", "view", "condition", "grade", "floors"]:
    print(f"\n{c}:")
    print(df.groupby(c)["price"].agg(["mean", "median", "count"]).round(0).to_string())
print("\n--- MEAN PRICE by zipcode (top/bottom 5 by mean) ---")
z = df.groupby("zipcode")["price"].agg(["mean", "count"]).sort_values("mean")
print(z.head(5).round(0).to_string())
print(z.tail(5).round(0).to_string())
print("\n--- MEAN PRICE by bedrooms ---")
print(df.groupby("bedrooms")["price"].agg(["mean", "median", "count"]).round(0).to_string())
print("\n--- price per sqft stats ---")
df["pps"] = df["price"] / df["sqft_living"]
print(df["pps"].describe(percentiles=[.01, .99]).round(1).to_string())
print("corr sqft_living vs price:", round(df["sqft_living"].corr(df["price"]), 3))
print("corr grade vs price:", round(df["grade"].corr(df["price"]), 3))
print("corr sqft_living15 vs price:", round(df["sqft_living15"].corr(df["price"]), 3))
