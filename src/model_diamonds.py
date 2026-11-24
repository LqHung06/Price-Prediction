"""Ridge vs Random Forest on Kaggle shivam2503/diamonds (diamonds.csv).
Design (documented):
- Drop: Unnamed: 0 (row index, not a feature; its -0.31 corr with price is an ordering
  artefact), and 20 rows with any of x/y/z == 0 (impossible dimensions = data errors).
- Target: log(price). Raw skew 1.62 -> log skew 0.12.
- Features: log_carat (skew 1.12 -> 0.10), log_volume = log(x*y*z) (size+shape, from
  dimensions only - no leakage), depth, table, cut, color, clarity.
- Categoricals: OneHot (NOT ordinal) - raw mean price is NOT monotonic in cut
  (Premium 4,584 > Fair 4,359 > Ideal 3,458) because cut correlates with carat;
  imposing an order would be wrong. Conditional effect is left to the model.
- No leakage columns: volume comes only from x/y/z; nothing is derived from price.
- Missing: none in file, but pipeline still has median/mode imputation.
- Outliers: winsorise numeric features at train 1%/99% (y max 58.9mm, z max 31.8mm are
  data errors; price right tail handled by log target).
- Split: random 80/20, random_state=42 (no date column -> no time split).
"""
import json, os, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
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
DATA = os.path.join(ROOT, "data", "diamonds.csv")
OUT = os.path.join(ROOT, "results", "diamonds")
CH = f"{OUT}/charts"
os.makedirs(CH, exist_ok=True)

df = pd.read_csv(DATA)
n_raw = len(df)
bad = (df[["x", "y", "z"]] == 0).any(axis=1)
print(f"raw {n_raw} rows | bad x/y/z==0 rows dropped: {int(bad.sum())}")
df = df[~bad].copy()
df = df.drop(columns=["Unnamed: 0"])
df["log_carat"] = np.log(df["carat"])
df["volume"] = df["x"] * df["y"] * df["z"]
df["log_volume"] = np.log(df["volume"])
df["y"] = np.log(df["price"])

NUM = ["log_carat", "log_volume", "depth", "table"]
CAT = ["cut", "color", "clarity"]
FEATS = NUM + CAT

X = df[FEATS]
y = df["y"]
y_raw = df["price"]
X_tr, X_te, y_tr, y_te, yraw_tr, yraw_te = train_test_split(
    X, y, y_raw, test_size=0.2, random_state=42)
print(f"train {len(X_tr)} | test {len(X_te)} | features {FEATS}")

lo = X_tr[NUM].quantile(0.01).to_numpy()
hi = X_tr[NUM].quantile(0.99).to_numpy()


class Winsoriser(BaseEstimator, TransformerMixin):
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
                             OneHotEncoder(handle_unknown="ignore", sparse_output=False))
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


results = {"n_raw": n_raw, "n_bad_xyz0_dropped": int(bad.sum()), "n_used": len(df)}
pred_mean = np.full(len(X_te), y_tr.mean())
results["Mean baseline"] = metrics("Mean baseline", y_te, pred_mean, yraw_te)

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

t = time.time()
rf = make_pipeline(prep(None),
                   RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                         n_jobs=-1, random_state=42))
rf.fit(X_tr, y_tr)
pred_rf = rf.predict(X_te)
results["RandomForest"] = metrics("RandomForest", y_te, pred_rf, yraw_te, time.time() - t)
results["RandomForest"]["R2_train_log"] = round(float(r2_score(y_tr, rf.predict(X_tr))), 4)
cv_rf = cross_val_score(rf, X, y, cv=kf, scoring="r2")
results["RandomForest"]["CV_R2_mean"] = round(float(cv_rf.mean()), 4)
results["RandomForest"]["CV_R2_std"] = round(float(cv_rf.std()), 4)
print("RF CV R2:", np.round(cv_rf, 3))

ohe = ridge[0].named_transformers_["cat"][1]
cat_names = list(ohe.get_feature_names_out(CAT))
names = NUM + cat_names
coefs = dict(zip(names, np.round(ridge[-1].coef_, 3)))
results["Ridge"]["coefs"] = {k: float(v) for k, v in coefs.items()}
print("Ridge coefs top:", {k: v for k, v in sorted(coefs.items(), key=lambda kv: -abs(kv[1]))[:12]})

perm_idx = np.random.RandomState(42).choice(len(X_te), size=min(3000, len(X_te)), replace=False)
perm = permutation_importance(rf, X_te.iloc[perm_idx], y_te.iloc[perm_idx],
                              n_repeats=3, random_state=42, n_jobs=1)
perm_imp = dict(zip(FEATS, np.round(perm.importances_mean, 4)))
results["RandomForest"]["perm_importance"] = {k: float(v) for k, v in perm_imp.items()}
print("RF permutation importance:", perm_imp)

resid = y_te.values - pred_rf
dec = pd.qcut(pred_rf, 10, labels=False)
diag = pd.DataFrame({"decile": dec, "resid": resid}).groupby("decile")["resid"].agg(["mean", "std", "count"])
print("RF residual by decile:\n", diag.round(3).to_string())

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

fig, ax = plt.subplots(figsize=(8, 4.4))
miss = (df[FEATS].isna().mean() * 100).sort_values()
miss.plot.barh(ax=ax, color="#1f77b4")
ax.set_xlabel("Tỉ lệ thiếu (%)"); ax.set_title("Tỉ lệ dữ liệu thiếu theo cột (diamonds: không có ô thiếu nào)")
for i, v in enumerate(miss.values):
    ax.text(v + 0.2, i, f"{v:.1f}%", va="center")
ax.set_xlim(0, 5); plt.tight_layout(); plt.savefig(f"{CH}/fig1_missing.png", dpi=150); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
ax[0].hist(df["price"], bins=60, color="#1f77b4", edgecolor="white")
ax[0].set_title("Giá kim cương (USD)"); ax[0].set_xlabel("Giá (USD)")
ax[1].hist(df["y"], bins=60, color="#2ca02c", edgecolor="white")
ax[1].set_title("log(Giá kim cương)"); ax[1].set_xlabel("log giá")
plt.tight_layout(); plt.savefig(f"{CH}/fig2_target_dist.png", dpi=150); plt.close()

corr = df[["carat", "log_carat", "log_volume", "depth", "table", "y"]].corr()
fig, ax = plt.subplots(figsize=(7, 5.5))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
labels = ["carat", "log(carat)", "log(volume)", "depth", "table", "log(Giá)"]
ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=35, ha="right"); ax.set_yticklabels(labels)
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=10)
ax.set_title("Ma trận tương quan")
plt.colorbar(im, ax=ax, shrink=0.85); plt.tight_layout()
plt.savefig(f"{CH}/fig3_corr.png", dpi=150); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(10, 4.6), sharex=True, sharey=True)
for a, pred, name, c in [(ax[0], np.exp(pred_ridge), "Ridge", C_RIDGE),
                         (ax[1], np.exp(pred_rf), "Random Forest", C_RF)]:
    a.scatter(yraw_te, pred, s=6, alpha=0.25, color=c)
    lims = [300, 19000]
    a.plot(lims, lims, "k--", lw=1)
    a.set_title(f"{name} (test)")
    a.set_xlabel("Thực tế (USD)"); a.set_ylabel("Dự báo (USD)")
plt.tight_layout(); plt.savefig(f"{CH}/fig4_actual_pred.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(8, 4.6))
ax.scatter(pred_rf, resid, s=6, alpha=0.25, color=C_RF)
ax.axhline(0, color="black", lw=1)
ax.set_xlabel("Giá trị dự báo (log giá)"); ax.set_ylabel("Phần dư (thực tế - dự báo)")
ax.set_title("Random Forest: phần dư theo giá trị dự báo (test)")
plt.tight_layout(); plt.savefig(f"{CH}/fig5_residual.png", dpi=150); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
pi = pd.Series(perm_imp).sort_values()
pi.plot.barh(ax=ax[0], color=C_RF)
ax[0].set_title("RF: permutation importance (test)")
ax[0].set_xlabel("Giảm R² khi xáo feature")
cf = pd.Series(coefs).sort_values()
cf.plot.barh(ax=ax[1], color=C_RIDGE)
ax[1].axvline(0, color="black", lw=1)
ax[1].set_title("Ridge: hệ số chuẩn hoá")
ax[1].set_xlabel("Hệ số")
plt.tight_layout(); plt.savefig(f"{CH}/fig6_importance.png", dpi=150); plt.close()

print("charts done")
