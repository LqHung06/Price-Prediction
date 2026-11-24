"""Melbourne preprocessing optimisation (opt_melb.py).
Goal: try preprocessing variants for Ridge / RF, compare ONLY by 5-fold CV on TRAIN,
evaluate the winner ONCE on test.

Protocol notes (leakage control):
- Stage A reproduces the ORIGINAL pipeline exactly (preprocessing learned on full
  train, as in model_melb.py) and checks test R2_log vs 0.7936 (Ridge) / 0.8685 (RF)
  within +/-0.002. If it fails, the script stops.
- Stage B compares variants with a leak-free prep transformer: suburb grouping,
  winsorise limits, medians/modes, scalers and target-encoding mappings are ALL
  learned inside each outer CV fold (prep is a Pipeline step). Target encoding uses
  OUT-OF-FOLD means inside fit (fit_transform returns OOF-encoded train rows;
  transform uses the full-fold mapping) with smoothing m=10.
- RF CV uses 150 trees (not 300) purely to fit the 20-min/variant budget: one
  300-tree fit takes ~270s on this VM, so a 5-fold CV at 300 trees would take
  ~22 min. The final test evaluation of the RF winner still uses 300 trees,
  exactly as in the original run. This is disclosed in the report.
- RidgeCV's inner 5-fold alpha search runs on the already-prepped outer-train
  fold (mild, standard, and disclosed); the outer CV score itself is clean.
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OPT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "notebooks", "melb_data.csv")
os.makedirs(OPT, exist_ok=True)
RES_PATH = f"{OPT}/opt_results.json"

ORIG_RIDGE_R2 = 0.7936
ORIG_RF_R2 = 0.8685
TOL = 0.002

# ---------------------------------------------------------------- load
df = pd.read_csv(DATA)
df["log_Landsize"] = np.log(df["Landsize"].replace(0, np.nan))
df["log_BuildingArea"] = np.log(df["BuildingArea"].replace(0, np.nan))
df["Age"] = 2017 - df["YearBuilt"]
df["BuildingArea_missing"] = df["BuildingArea"].isna().astype(int)
df["YearBuilt_missing"] = df["YearBuilt"].isna().astype(int)
df["y"] = np.log(df["Price"])

NUM = ["Rooms", "Bathroom", "Car", "Distance", "log_Landsize", "log_BuildingArea",
       "Age", "Lattitude", "Longtitude", "Propertycount",
       "BuildingArea_missing", "YearBuilt_missing"]
CAT = ["Type", "Method", "Regionname", "CouncilArea", "Suburb_grouped"]
FEATS = NUM + CAT
CONT_NUM = ["Rooms", "Bathroom", "Car", "Distance", "log_Landsize", "log_BuildingArea",
            "Age", "Lattitude", "Longtitude", "Propertycount"]
KNN_COLS = ["log_Landsize", "log_BuildingArea", "Age"]
RAW_COLS = NUM + ["Suburb", "Type", "Method", "Regionname", "CouncilArea"]

X = df[RAW_COLS]
y = df["y"]
y_raw = df["Price"]
X_tr, X_te, y_tr, y_te, yraw_tr, yraw_te = train_test_split(
    X, y, y_raw, test_size=0.2, random_state=42)
print(f"train {len(X_tr)} | test {len(X_te)}", flush=True)

results = {"baseline_reproduction": {}, "cv": {"ridge": {}, "rf_150trees": {}},
           "test_winner": {}}


def save():
    with open(RES_PATH, "w") as f:
        json.dump(results, f, indent=2)


def metrics(name, y_true_log, pred_log, y_true_raw):
    usd_pred = np.exp(pred_log)
    m = {
        "R2_log": round(float(r2_score(y_true_log, pred_log)), 4),
        "R2_raw": round(float(r2_score(y_true_raw, usd_pred)), 4),
        "MAE_log": round(float(mean_absolute_error(y_true_log, pred_log)), 4),
        "MAE_raw": round(float(mean_absolute_error(y_true_raw, usd_pred)), 1),
        "RMSE_raw": round(float(mean_squared_error(y_true_raw, usd_pred) ** 0.5), 1),
        "MedAPE_pct": round(float(np.median(np.abs(usd_pred / y_true_raw - 1)) * 100), 2),
    }
    print(name, m, flush=True)
    return m


# ================================================================ STAGE A
# Exact copy of the original pipeline (model_melb.py): preprocessing learned
# on the FULL train, then fit once and evaluate on test.
print("\n=== STAGE A: reproduce original baseline ===", flush=True)
vc = X_tr["Suburb"].value_counts()
keep_suburbs = set(vc[vc >= 30].index)


def add_suburb_group(Xdf):
    Xdf = Xdf.copy()
    Xdf["Suburb_grouped"] = np.where(Xdf["Suburb"].isin(keep_suburbs),
                                     Xdf["Suburb"], "Other")
    return Xdf[FEATS]


X_tr_a = add_suburb_group(X_tr)
X_te_a = add_suburb_group(X_te)
lo = X_tr_a[NUM].quantile(0.01).to_numpy()
hi = X_tr_a[NUM].quantile(0.99).to_numpy()


class Winsoriser(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        return np.clip(X, lo, hi)


def prep_orig(scaler):
    num_pipe = make_pipeline(SimpleImputer(strategy="median"), Winsoriser(), scaler) \
        if scaler else make_pipeline(SimpleImputer(strategy="median"), Winsoriser())
    cat_pipe = make_pipeline(SimpleImputer(strategy="most_frequent"),
                             OneHotEncoder(handle_unknown="ignore"))
    return ColumnTransformer([("num", num_pipe, NUM), ("cat", cat_pipe, CAT)])


t = time.time()
ridge_a = make_pipeline(prep_orig(StandardScaler()),
                        RidgeCV(alphas=[0.1, 1, 10, 100, 1000], cv=5))
ridge_a.fit(X_tr_a, y_tr)
pred_ridge_a = ridge_a.predict(X_te_a)
m_ridge_a = metrics("Ridge baseline", y_te, pred_ridge_a, yraw_te)
print(f"  Ridge fit took {time.time() - t:.1f}s, alpha={ridge_a[-1].alpha_}", flush=True)

t = time.time()
rf_a = make_pipeline(prep_orig(None),
                     RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                           n_jobs=-1, random_state=42))
rf_a.fit(X_tr_a, y_tr)
pred_rf_a = rf_a.predict(X_te_a)
m_rf_a = metrics("RF baseline", y_te, pred_rf_a, yraw_te)
print(f"  RF fit took {time.time() - t:.1f}s", flush=True)

ok_ridge = abs(m_ridge_a["R2_log"] - ORIG_RIDGE_R2) <= TOL
ok_rf = abs(m_rf_a["R2_log"] - ORIG_RF_R2) <= TOL
results["baseline_reproduction"] = {
    "ridge_R2_log": m_ridge_a["R2_log"], "ridge_expected": ORIG_RIDGE_R2,
    "ridge_within_tol": bool(ok_ridge),
    "rf_R2_log": m_rf_a["R2_log"], "rf_expected": ORIG_RF_R2,
    "rf_within_tol": bool(ok_rf)}
save()
print(f"Baseline check: Ridge {'PASS' if ok_ridge else 'FAIL'} | "
      f"RF {'PASS' if ok_rf else 'FAIL'}", flush=True)
if not (ok_ridge and ok_rf):
    print("BASELINE MISMATCH > 0.002 -- STOPPING, no variants tried.", flush=True)
    sys.exit(1)

# ================================================================ STAGE B
# Leak-free prep transformer: everything learned inside each CV fold.
print("\n=== STAGE B: variant comparison by 5-fold CV on TRAIN only ===", flush=True)


def out_columns(target_enc=False, interactions=False, use_flags=True):
    num = list(CONT_NUM)
    if use_flags:
        num += ["BuildingArea_missing", "YearBuilt_missing"]
    if interactions:
        num += ["lat_long", "dist_Type_h", "dist_Type_t", "dist_Type_u",
                "logland_rooms"]
    if target_enc:
        num += ["Suburb_te"]
    cat = ["Type", "Method", "Regionname", "CouncilArea"]
    if not target_enc:
        cat += ["Suburb_grouped"]
    return num, cat


class MelbPrep(BaseEstimator, TransformerMixin):
    """All preprocessing inside the fold. Input: raw X (RAW_COLS). Output:
    DataFrame with imputed/winsorised numerics (+ optional interactions /
    target-encoded suburb) and categoricals as strings."""

    def __init__(self, suburb_min=30, knn=False, target_enc=False,
                 interactions=False, use_flags=True):
        self.suburb_min = suburb_min
        self.knn = knn
        self.target_enc = target_enc
        self.interactions = interactions
        self.use_flags = use_flags

    def fit(self, X, y=None):
        X = X.reset_index(drop=True)
        vc = X["Suburb"].value_counts()
        self.keep_suburbs_ = set(vc[vc >= self.suburb_min].index)
        num = X[CONT_NUM]
        self.medians_ = num.median()
        self.lo_ = num.quantile(0.01)
        self.hi_ = num.quantile(0.99)
        self.modes_ = {c: X[c].mode(dropna=True)[0]
                       for c in ["Type", "Method", "Regionname", "CouncilArea"]}
        if self.knn:
            # scale stats from winsorised non-NaN values (for KNN distances)
            tmp = self._prep_numeric(X)
            self.knn_mean_ = tmp[CONT_NUM].mean()
            self.knn_std_ = tmp[CONT_NUM].std(ddof=0).replace(0, 1.0)
        if self.target_enc:
            yv = np.asarray(y, dtype=float)
            keys = X["Suburb"].astype(str).to_numpy()
            gmean = float(yv.mean())
            self.te_global_ = gmean
            d = pd.DataFrame({"k": keys, "y": yv})
            a = d.groupby("k")["y"].agg(["mean", "count"])
            self.te_map_ = ((a["count"] * a["mean"] + 10 * gmean)
                            / (a["count"] + 10)).to_dict()
            kf = KFold(5, shuffle=True, random_state=42)
            oof = np.empty(len(yv))
            for tr_idx, va_idx in kf.split(keys):
                dd = pd.DataFrame({"k": keys[tr_idx], "y": yv[tr_idx]})
                aa = dd.groupby("k")["y"].agg(["mean", "count"])
                gm = float(dd["y"].mean())
                m = ((aa["count"] * aa["mean"] + 10 * gm)
                     / (aa["count"] + 10)).to_dict()
                oof[va_idx] = np.array([m.get(k, gm) for k in keys[va_idx]])
            self.oof_te_ = oof
        return self

    def _prep_numeric(self, X):
        """Median-impute non-KNN numerics, winsorise. KNN cols keep NaN."""
        num = X[CONT_NUM].copy()
        if self.knn:
            fill = {c: self.medians_[c] for c in CONT_NUM if c not in KNN_COLS}
            num = num.fillna(fill)
        else:
            num = num.fillna(self.medians_)
        return num.clip(lower=self.lo_, upper=self.hi_, axis=1)

    def _transform(self, X, oof=False):
        X = X.reset_index(drop=True)
        num = self._prep_numeric(X)
        if self.knn:
            sc = (num[CONT_NUM] - self.knn_mean_) / self.knn_std_
            imp = KNNImputer(n_neighbors=5)
            sc_imp = pd.DataFrame(imp.fit_transform(sc), columns=CONT_NUM,
                                  index=sc.index)
            num = sc_imp * self.knn_std_ + self.knn_mean_
        out = pd.DataFrame(index=X.index)
        for c in CONT_NUM:
            out[c] = num[c].to_numpy()
        if self.use_flags:
            out["BuildingArea_missing"] = X["BuildingArea_missing"].to_numpy()
            out["YearBuilt_missing"] = X["YearBuilt_missing"].to_numpy()
        if self.interactions:
            out["lat_long"] = num["Lattitude"] * num["Longtitude"]
            for lv in ["h", "t", "u"]:
                out[f"dist_Type_{lv}"] = num["Distance"] * (X["Type"] == lv).astype(int)
            out["logland_rooms"] = num["log_Landsize"] * num["Rooms"]
        if self.target_enc:
            keys = X["Suburb"].astype(str)
            if oof:
                out["Suburb_te"] = self.oof_te_
            else:
                out["Suburb_te"] = keys.map(self.te_map_).fillna(self.te_global_).to_numpy()
        for c in ["Type", "Method", "Regionname", "CouncilArea"]:
            out[c] = X[c].fillna(self.modes_[c]).astype(str).to_numpy()
        if not self.target_enc:
            out["Suburb_grouped"] = np.where(
                X["Suburb"].isin(self.keep_suburbs_), X["Suburb"], "Other")
        return out

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self._transform(X, oof=True)

    def transform(self, X):
        return self._transform(X, oof=False)


def make_pipe(kind, variant, rf_trees=300):
    num_cols, cat_cols = out_columns(
        target_enc=variant.get("target_enc", False),
        interactions=variant.get("interactions", False),
        use_flags=variant.get("use_flags", True))
    prep = MelbPrep(**variant)
    if kind == "ridge":
        ct = ColumnTransformer(
            [("num", StandardScaler(), num_cols),
             ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)])
        model = RidgeCV(alphas=[0.1, 1, 10, 100, 1000], cv=5)
    else:
        ct = ColumnTransformer(
            [("num", "passthrough", num_cols),
             ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)])
        model = RandomForestRegressor(n_estimators=rf_trees, min_samples_leaf=2,
                                      n_jobs=-1, random_state=42)
    return Pipeline([("prep", prep), ("ct", ct), ("model", model)])


KF = KFold(5, shuffle=True, random_state=42)
RIDGE_VARIANTS = {
    "base": {},
    "knn": {"knn": True},
    "target_enc": {"target_enc": True},
    "interactions": {"interactions": True},
    "suburb10": {"suburb_min": 10},
    "no_flags": {"use_flags": False},
    "te_inter": {"target_enc": True, "interactions": True},
}
RF_VARIANTS = {
    "base": {},
    "target_enc": {"target_enc": True},
    "suburb10": {"suburb_min": 10},
}

for name, var in RIDGE_VARIANTS.items():
    t = time.time()
    pipe = make_pipe("ridge", var)
    scores = cross_val_score(pipe, X_tr, y_tr, cv=KF, scoring="r2")
    el = time.time() - t
    results["cv"]["ridge"][name] = {
        "mean": round(float(scores.mean()), 4),
        "std": round(float(scores.std()), 4),
        "folds": [round(float(s), 4) for s in scores],
        "seconds": round(el, 1)}
    save()
    print(f"Ridge CV [{name}]: {scores.mean():.4f} ± {scores.std():.4f} "
          f"folds={np.round(scores, 3)} ({el:.0f}s)", flush=True)

for name, var in RF_VARIANTS.items():
    t = time.time()
    pipe = make_pipe("rf", var, rf_trees=150)
    scores = cross_val_score(pipe, X_tr, y_tr, cv=KF, scoring="r2")
    el = time.time() - t
    results["cv"]["rf_150trees"][name] = {
        "mean": round(float(scores.mean()), 4),
        "std": round(float(scores.std()), 4),
        "folds": [round(float(s), 4) for s in scores],
        "seconds": round(el, 1)}
    save()
    print(f"RF(150 trees) CV [{name}]: {scores.mean():.4f} ± {scores.std():.4f} "
          f"folds={np.round(scores, 3)} ({el:.0f}s)", flush=True)

# ================================================================ STAGE C
# Pick winners by CV R2_log on train, evaluate ONCE on test.
print("\n=== STAGE C: single test evaluation of CV winners ===", flush=True)
ridge_cv = results["cv"]["ridge"]
rf_cv = results["cv"]["rf_150trees"]
best_ridge = max(ridge_cv, key=lambda k: ridge_cv[k]["mean"])
best_rf = max(rf_cv, key=lambda k: rf_cv[k]["mean"])
print(f"Ridge winner by CV: {best_ridge} "
      f"({ridge_cv[best_ridge]['mean']:.4f} vs base {ridge_cv['base']['mean']:.4f})",
      flush=True)
print(f"RF winner by CV: {best_rf} "
      f"({rf_cv[best_rf]['mean']:.4f} vs base {rf_cv['base']['mean']:.4f})",
      flush=True)

t = time.time()
pipe_r = make_pipe("ridge", RIDGE_VARIANTS[best_ridge])
pipe_r.fit(X_tr, y_tr)
pred_r = pipe_r.predict(X_te)
m = metrics(f"Ridge winner [{best_ridge}] test", y_te, pred_r, yraw_te)
m["variant"] = best_ridge
m["alpha"] = float(pipe_r[-1].alpha_)
m["R2_train_log"] = round(float(r2_score(y_tr, pipe_r.predict(X_tr))), 4)
m["CV_R2_mean"] = ridge_cv[best_ridge]["mean"]
m["CV_R2_std"] = ridge_cv[best_ridge]["std"]
m["train_sec"] = round(time.time() - t, 1)
results["test_winner"]["ridge"] = m
save()

t = time.time()
pipe_f = make_pipe("rf", RF_VARIANTS[best_rf], rf_trees=300)
pipe_f.fit(X_tr, y_tr)
pred_f = pipe_f.predict(X_te)
m = metrics(f"RF winner [{best_rf}] test (300 trees)", y_te, pred_f, yraw_te)
m["variant"] = best_rf
m["R2_train_log"] = round(float(r2_score(y_tr, pipe_f.predict(X_tr))), 4)
m["CV_R2_mean"] = rf_cv[best_rf]["mean"]
m["CV_R2_std"] = rf_cv[best_rf]["std"]
m["note"] = "CV used 150 trees; final test fit uses 300 trees as in original run"
m["train_sec"] = round(time.time() - t, 1)
results["test_winner"]["rf"] = m
save()

print("\nDONE. Full results in", RES_PATH, flush=True)
print(json.dumps(results, indent=2), flush=True)
