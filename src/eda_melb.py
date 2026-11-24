"""EDA for Kaggle dansbecker/melbourne-housing-snapshot (melb_data.csv)."""
import numpy as np
import os
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(ROOT, "notebooks", "melb_data.csv"))
print(f"SHAPE {df.shape}")
print(df.dtypes.to_string())
print(f"\nduplicated rows: {df.duplicated().sum()}")
print(f"duplicated (Suburb,Address,Date,Price): {df.duplicated(subset=['Suburb','Address','Date','Price']).sum()}")
print(f"\nDate range: {df['Date'].min()} -> {df['Date'].max()} (n unique: {df['Date'].nunique()})")
print(f"Price missing: {df['Price'].isna().sum()} | Price==0: {(df['Price']==0).sum()}")

print("\n--- MISSING ---")
miss = df.isna().sum()
print(miss.to_string())
print(f"\nmissing %:")
print((miss/len(df)*100).round(1).to_string())
print(f"rows with any missing: {df.isna().any(axis=1).sum()} / {len(df)}")
print(f"rows with all 3 key numerics (Landsize/BuildingArea/YearBuilt) present: "
      f"{df[['Landsize','BuildingArea','YearBuilt']].notna().all(axis=1).sum()}")

print("\n--- NUMERIC describe ---")
print(df.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T.round(2).to_string())
print(f"\nPrice skew: {df['Price'].skew():.2f} | log Price skew: {np.log(df['Price']).skew():.2f}")
print(f"Landsize skew: {df['Landsize'].skew():.2f} | log Landsize skew: {np.log(df['Landsize'].replace(0,np.nan)).skew():.2f}")
print(f"BuildingArea skew: {df['BuildingArea'].skew():.2f} | log BuildingArea skew: {np.log(df['BuildingArea'].replace(0,np.nan)).skew():.2f}")

print("\n--- CORR with Price / log Price ---")
num = df.select_dtypes("number")
print(num.corr()["Price"].drop("Price").round(3).sort_values(key=abs, ascending=False).to_string())
df["logPrice"] = np.log(df["Price"])
print("\ncorr with logPrice (top):")
print(num.corr()["Price"].drop("Price").round(3).sort_values(key=abs, ascending=False).to_string())
print("\ncorr matrix (key numerics):")
print(df[["Price","Rooms","Bedroom2","Bathroom","Car","Distance","Landsize","BuildingArea","YearBuilt","Propertycount"]].corr().round(2).to_string())

print("\n--- IQR OUTLIERS ---")
for c in ["Price","Rooms","Bathroom","Car","Distance","Landsize","BuildingArea","YearBuilt","Propertycount"]:
    s = df[c].dropna()
    q1, q3 = s.quantile(.25), s.quantile(.75)
    iqr = q3 - q1
    n = ((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum()
    print(f"{c:14s} outliers={n:5d} ({100*n/len(s):5.1f}%) range=[{s.min():.1f}, {s.max():.1f}] p99={s.quantile(.99):.1f}")
print("\nLandsize==0:", (df["Landsize"]==0).sum(), "| BuildingArea==0:", (df["BuildingArea"]==0).sum())
print("YearBuilt < 1800:", (df["YearBuilt"]<1800).sum(), "| YearBuilt > 2017:", (df["YearBuilt"]>2017).sum())

print("\n--- CATEGORICAL counts ---")
for c in ["Type","Method","Regionname"]:
    print(f"\n{c}:")
    print(df[c].value_counts(dropna=False).to_string())
print(f"\nSuburb: {df['Suburb'].nunique()} unique; top 10:")
print(df["Suburb"].value_counts().head(10).to_string())
print(f"Suburb with <10 rows: {(df['Suburb'].value_counts()<10).sum()} of {df['Suburb'].nunique()}")
print(f"\nSellerG: {df['SellerG'].nunique()} unique; top 10:")
print(df["SellerG"].value_counts().head(10).to_string())
print(f"\nCouncilArea: {df['CouncilArea'].nunique()} unique (missing {df['CouncilArea'].isna().sum()})")
print(df["CouncilArea"].value_counts().head(10).to_string())
print(f"\nPostcode: {df['Postcode'].nunique()} unique")

print("\n--- MEAN PRICE by category ---")
for c in ["Type","Method","Regionname"]:
    print(f"\n{c}:")
    print(df.groupby(c, dropna=False)["Price"].agg(["mean","median","count"]).round(0).to_string())
print("\n--- MEAN PRICE by Rooms ---")
print(df.groupby("Rooms")["Price"].agg(["mean","median","count"]).round(0).to_string())
print("\n--- MEAN PRICE by Bathroom ---")
print(df.groupby("Bathroom")["Price"].agg(["mean","median","count"]).round(0).to_string())
print("\n--- MEAN PRICE by Car ---")
print(df.groupby("Car")["Price"].agg(["mean","median","count"]).round(0).to_string())
