"""Ridge vs Random Forest on Kaggle house-price-predecition (dataset_2.csv).
Design (documented):
- Target: log(Price). Raw skew 1.75 -> log skew -0.11. Area skew 3.50 -> log Area skew -0.16.
- Features: log_Area, Rooms, Age (=2025-Build_Year), Area_per_Room, Location,
  Street_Type, Furnishing, Property_Type, Has_Pool.
- No leakage columns exist in this file (no price-derived ratios); nothing excluded for leakage.
- Missing (33 rows each in Area/Rooms/Furnishing, 97 rows affected): median / most-frequent
  imputation from TRAIN only.
- Outliers: winsorise log_Area + numeric features at train 1%/99% (Area max 10,267 vs p99 3,730;
  Price max 2.07M vs p99 972k handled by log target, not deleted).
- Split: random 80/20 (no date column -> no time split possible), random_state=42.
  5-fold CV R2 reported for stability.
"""
import json, os, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "notebooks", "dataset_2.csv")
OUT = os.path.join(ROOT, "results", "indian-house")
CH = f"{OUT}/charts"
os.makedirs(CH, exist_ok=True)

df = pd.read_csv(DATA)
df["Age"] = 2025 - df["Build_Year"]
df["log_Area"] = np.log(df["Area_SqFt"])
df["Area_per_Room"] = df["Area_SqFt"] / df["Rooms"]
df["y"] = np.log(df["Price"])

NUM = ["log_Area", "Rooms", "Age", "Area_per_Room"]
CAT = ["Location", "Street_Type", "Furnishing", "Property_Type", "Has_Pool"]
FEATS = NUM + CAT

X = df[FEATS]
y = df["y"]
y_raw = df["Price"]
X_tr, X_te, y_tr, y_te, yraw_tr, yraw_te = train_test_split(
    X, y, y_raw, test_size=0.2, random_state=42)
print(f"train {len(X_tr)} | test {len(X_te)} | features {FEATS}")

# winsorise limits for numeric cols from TRAIN (as arrays: pipeline passes numpy)
lo = X_tr[NUM].quantile(0.01).to_numpy()
hi = X_tr[NUM].quantile(0.99).to_numpy()


from sklearn.base import BaseEstimator, TransformerMixin


class Winsoriser(BaseEstimator, TransformerMixin):
    """Clips each numeric column to its train 1%/99% percentiles. Input is a numpy array."""

    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        return np.clip(X, lo, hi)


def prep(scaler):
    num_pipe = make_pipeline(SimpleImputer(strategy="median"), Winsoriser(),
                             scaler) if scaler else make_pipeline(
                                 SimpleImputer(strategy="median"), Winsoriser())
    cat_pipe = make_pipeline(SimpleImputer(strategy="most_frequent"),
                             OneHotEncoder(handle_unknown="ignore"))
    return ColumnTransformer([("num", num_pipe, NUM), ("cat", cat_pipe, CAT)])


def metrics(name, y_true_log, pred_log, y_true_raw, secs=None):
    usd_pred = np.exp(pred_log)
    m = {
        "R2_log": round(float(r2_score(y_true_log, pred_log)), 4),
        "R2_raw": round(float(r2_score(y_true_raw, usd_pred)), 4),
        "MAE_log": round(float(mean_absolute_error(y_true_log, pred_log)), 4),
        "MAE_raw": round(float(mean_absolute_error(y_true_raw, usd_pred)), 1),
        "RMSE_raw": round(float(mean_squared_error(y_true_raw, usd_pred) ** 0.5), 1),
        "MedAPE_pct": round(float(np.median(np.abs(usd_pred / y_true_raw - 1)) * 100), 2),
    }
    if secs is not None:
        m["train_sec"] = round(secs, 1)
    print(name, m)
    return m


results = {}
# baseline: train mean (log)
pred_mean = np.full(len(X_te), y_tr.mean())
results["Mean baseline"] = metrics("Mean baseline", y_te, pred_mean, yraw_te)

# Ridge
t = time.time()
ridge = make_pipeline(prep(StandardScaler()), RidgeCV(alphas=[0.1, 1, 10, 100, 1000], cv=5))
ridge.fit(X_tr, y_tr)
alpha = ridge[-1].alpha_
pred_ridge = ridge.predict(X_te)
results["Ridge"] = metrics("Ridge", y_te, pred_ridge, yraw_te, time.time() - t)
results["Ridge"]["alpha"] = float(alpha)
results["Ridge"]["R2_train_log"] = round(float(r2_score(y_tr, ridge.predict(X_tr))), 4)
kf = KFold(5, shuffle=True, random_state=42)
cv_ridge = cross_val_score(ridge, X, y, cv=kf, scoring="r2")
results["Ridge"]["CV_R2_mean"] = round(float(cv_ridge.mean()), 4)
results["Ridge"]["CV_R2_std"] = round(float(cv_ridge.std()), 4)
print("Ridge alpha:", alpha, "| CV R2:", np.round(cv_ridge, 3))

# Random Forest
t = time.time()
rf = make_pipeline(prep(None),
                   RandomForestRegressor(n_estimators=500, min_samples_leaf=2,
                                         n_jobs=-1, random_state=42))
rf.fit(X_tr, y_tr)
pred_rf = rf.predict(X_te)
results["RandomForest"] = metrics("RandomForest", y_te, pred_rf, yraw_te, time.time() - t)
results["RandomForest"]["R2_train_log"] = round(float(r2_score(y_tr, rf.predict(X_tr))), 4)
cv_rf = cross_val_score(rf, X, y, cv=kf, scoring="r2")
results["RandomForest"]["CV_R2_mean"] = round(float(cv_rf.mean()), 4)
results["RandomForest"]["CV_R2_std"] = round(float(cv_rf.std()), 4)
print("RF CV R2:", np.round(cv_rf, 3))

# Ridge coefs (feature names after one-hot)
ohe = ridge[0].named_transformers_["cat"][1]
cat_names = list(ohe.get_feature_names_out(CAT))
names = NUM + cat_names
coefs = dict(zip(names, np.round(ridge[-1].coef_, 3)))
results["Ridge"]["coefs"] = {k: float(v) for k, v in coefs.items()}
print("Ridge coefs:", {k: v for k, v in sorted(coefs.items(), key=lambda kv: -abs(kv[1]))[:10]})

# RF permutation importance on test (grouped by original feature)
perm = permutation_importance(rf, X_te, y_te, n_repeats=5, random_state=42, n_jobs=-1)
perm_imp = dict(zip(FEATS, np.round(perm.importances_mean, 4)))
results["RandomForest"]["perm_importance"] = {k: float(v) for k, v in perm_imp.items()}
print("RF permutation importance:", perm_imp)

# residual diagnostics (RF)
resid = y_te.values - pred_rf
dec = pd.qcut(pred_rf, 10, labels=False)
diag = pd.DataFrame({"decile": dec, "resid": resid}).groupby("decile")["resid"].agg(["mean", "std", "count"])
print("RF residual by decile:\n", diag.round(3).to_string())

# example: median house
med = X_te.iloc[[0]]
print("example row actual/pred:",
      round(float(yraw_te.iloc[0]), 0),
      round(float(np.exp(pred_ridge[0])), 0),
      round(float(np.exp(pred_rf[0])), 0))

# save test predictions
pd.DataFrame({"actual": yraw_te.values, "ridge": np.exp(pred_ridge), "rf": np.exp(pred_rf),
              "actual_log": y_te.values, "ridge_log": pred_ridge, "rf_log": pred_rf}
             ).to_csv(f"{OUT}/test_predictions.csv", index=False)
with open(f"{OUT}/results.json", "w") as f:
    json.dump({"results": results, "n_train": len(X_tr), "n_test": len(X_te),
               "features": FEATS}, f, indent=2)
print("saved results.json")

# ================= CHARTS =================
plt.rcParams.update({"font.size": 11, "axes.edgecolor": "#333333",
                     "axes.labelcolor": "#101216", "text.color": "#101216",
                     "xtick.color": "#101216", "ytick.color": "#101216",
                     "figure.facecolor": "white", "axes.facecolor": "white"})
C_RIDGE, C_RF = "#1f77b4", "#2ca02c"

# 1. missing
miss = (df[FEATS].isna().mean() * 100).sort_values()
fig, ax = plt.subplots(figsize=(8, 4.4))
miss.plot.barh(ax=ax, color="#1f77b4")
ax.set_xlabel("Tỉ lệ thiếu (%)"); ax.set_title("Tỉ lệ dữ liệu thiếu theo cột")
for i, v in enumerate(miss.values):
    ax.text(v + 0.2, i, f"{v:.1f}%", va="center")
ax.set_xlim(0, 5); plt.tight_layout(); plt.savefig(f"{CH}/fig1_missing.png", dpi=150); plt.close()

# 2. target raw vs log
fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
ax[0].hist(df["Price"], bins=40, color="#1f77b4", edgecolor="white")
ax[0].set_title("Giá nhà (đơn vị tiền)"); ax[0].set_xlabel("Giá")
ax[1].hist(df["y"], bins=40, color="#2ca02c", edgecolor="white")
ax[1].set_title("log(Giá nhà)"); ax[1].set_xlabel("log giá")
plt.tight_layout(); plt.savefig(f"{CH}/fig2_target_dist.png", dpi=150); plt.close()

# 3. corr heatmap (numeric + log price)
corr = df[["Area_SqFt", "Rooms", "Build_Year", "Age", "Area_per_Room", "y"]].corr()
fig, ax = plt.subplots(figsize=(7, 5.5))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
labels = ["Diện tích", "Số phòng", "Năm xây", "Tuổi nhà", "Diện tích/phòng", "log(Giá)"]
ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=35, ha="right"); ax.set_yticklabels(labels)
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=10)
ax.set_title("Ma trận tương quan")
plt.colorbar(im, ax=ax, shrink=0.85); plt.tight_layout()
plt.savefig(f"{CH}/fig3_corr.png", dpi=150); plt.close()

# 4. actual vs predicted
fig, ax = plt.subplots(1, 2, figsize=(10, 4.6), sharex=True, sharey=True)
for a, pred, name, c in [(ax[0], np.exp(pred_ridge), "Ridge", C_RIDGE),
                         (ax[1], np.exp(pred_rf), "Random Forest", C_RF)]:
    a.scatter(yraw_te, pred, s=14, alpha=0.55, color=c)
    lims = [200000, 2100000]
    a.plot(lims, lims, "k--", lw=1)
    a.set_title(f"{name} (test)")
    a.set_xlabel("Thực tế"); a.set_ylabel("Dự báo")
plt.tight_layout(); plt.savefig(f"{CH}/fig4_actual_pred.png", dpi=150); plt.close()

# 5. residual vs fitted (RF)
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.scatter(pred_rf, resid, s=14, alpha=0.55, color=C_RF)
ax.axhline(0, color="black", lw=1)
ax.set_xlabel("Giá trị dự báo (log giá)"); ax.set_ylabel("Phần dư (thực tế - dự báo)")
ax.set_title("Random Forest: phần dư theo giá trị dự báo (test)")
plt.tight_layout(); plt.savefig(f"{CH}/fig5_residual.png", dpi=150); plt.close()

# 6. importance
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
pi = pd.Series(perm_imp).sort_values()
pi.plot.barh(ax=ax[0], color=C_RF)
ax[0].set_title("RF: permutation importance (test)")
ax[0].set_xlabel("Giảm R² khi xáo feature")
cf = pd.Series({k: v for k, v in coefs.items() if k in NUM or "Location" in k or "Property" in k})
cf = pd.Series(coefs).sort_values()
cf.plot.barh(ax=ax[1], color=C_RIDGE)
ax[1].axvline(0, color="black", lw=1)
ax[1].set_title("Ridge: hệ số chuẩn hoá")
ax[1].set_xlabel("Hệ số")
plt.tight_layout(); plt.savefig(f"{CH}/fig6_importance.png", dpi=150); plt.close()

print("charts done")
