"""Melbourne preprocessing COMBO round (combo_melb.py).
Question: is the TRIPLE combo (interactions + KNN impute + suburb>=10) the best
for Ridge, or does stacking the three individually-helpful variants fail to add up?

Protocol (same as opt_melb.py):
- Stage 1: reproduce Ridge + interactions CV on TRAIN only; expect ~0.7959
  (+/-0.005). If off, STOP.
- Stage 2: 5-fold CV (KFold shuffle, random_state=42) on TRAIN only for
  triple / pair1 (interactions+KNN) / pair2 (interactions+suburb10), Ridge only.
  All preprocessing learned inside each fold (MelbPrep is a Pipeline step).
- Stage 3: pick the CV winner among {triple, pair1, pair2, interactions-single}
  and evaluate it ONCE on test. If triple does not win CV, say so plainly and
  do NOT evaluate triple on test.
- Never touches opt_results.json / opt_melb.py / results.json / model_melb.py.
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
RES_PATH = f"{OPT}/combo_results.json"

EXPECTED_INTER_CV = 0.7959
TOL = 0.005

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

results = {"reproduction": {}, "cv": {"ridge": {}},
           "cv_reference_from_opt_results": {
               "base": 0.7811, "knn": 0.7914, "suburb10": 0.7927,
               "interactions": 0.7959, "target_enc": 0.7766, "no_flags": 0.7805,
               "te_inter": 0.7895},
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


# ------------------------------------------------- prep (copied from opt_melb.py)
def out_columns(interactions=False, use_flags=True):
    num = list(CONT_NUM)
    if use_flags:
        num += ["BuildingArea_missing", "YearBuilt_missing"]
    if interactions:
        num += ["lat_long", "dist_Type_h", "dist_Type_t", "dist_Type_u",
                "logland_rooms"]
    cat = ["Type", "Method", "Regionname", "CouncilArea", "Suburb_grouped"]
    return num, cat


class MelbPrep(BaseEstimator, TransformerMixin):
    def __init__(self, suburb_min=30, knn=False, interactions=False, use_flags=True):
        self.suburb_min = suburb_min
        self.knn = knn
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
            tmp = self._prep_numeric(X)
            self.knn_mean_ = tmp[CONT_NUM].mean()
            self.knn_std_ = tmp[CONT_NUM].std(ddof=0).replace(0, 1.0)
        return self

    def _prep_numeric(self, X):
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
        for c in ["Type", "Method", "Regionname", "CouncilArea"]:
            out[c] = X[c].fillna(self.modes_[c]).astype(str).to_numpy()
        out["Suburb_grouped"] = np.where(
            X["Suburb"].isin(self.keep_suburbs_), X["Suburb"], "Other")
        return out

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self._transform(X, oof=True)

    def transform(self, X):
        return self._transform(X, oof=False)


def make_pipe(variant):
    num_cols, cat_cols = out_columns(
        interactions=variant.get("interactions", False),
        use_flags=variant.get("use_flags", True))
    prep = MelbPrep(**variant)
    ct = ColumnTransformer(
        [("num", StandardScaler(), num_cols),
         ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)])
    model = RidgeCV(alphas=[0.1, 1, 10, 100, 1000], cv=5)
    return Pipeline([("prep", prep), ("ct", ct), ("model", model)])


KF = KFold(5, shuffle=True, random_state=42)
VARIANTS = {
    "interactions_single": {"interactions": True},
    "pair1_inter_knn": {"interactions": True, "knn": True},
    "pair2_inter_suburb10": {"interactions": True, "suburb_min": 10},
    "triple": {"interactions": True, "knn": True, "suburb_min": 10},
}


def cv_variant(name, var):
    t = time.time()
    pipe = make_pipe(var)
    scores = cross_val_score(pipe, X_tr, y_tr, cv=KF, scoring="r2")
    el = time.time() - t
    entry = {"mean": round(float(scores.mean()), 4),
             "std": round(float(scores.std()), 4),
             "folds": [round(float(s), 4) for s in scores],
             "seconds": round(el, 1)}
    results["cv"]["ridge"][name] = entry
    save()
    print(f"Ridge CV [{name}]: {scores.mean():.4f} ± {scores.std():.4f} "
          f"folds={np.round(scores, 3)} ({el:.0f}s)", flush=True)
    return entry


# ================================================================ STAGE 1
print("\n=== STAGE 1: reproduce Ridge + interactions CV ===", flush=True)
rep = cv_variant("interactions_single", VARIANTS["interactions_single"])
ok = abs(rep["mean"] - EXPECTED_INTER_CV) <= TOL
results["reproduction"] = {
    "interactions_cv_mean": rep["mean"], "expected": EXPECTED_INTER_CV,
    "within_tol": bool(ok)}
save()
print(f"Reproduction check: {'PASS' if ok else 'FAIL'} "
      f"({rep['mean']:.4f} vs expected {EXPECTED_INTER_CV})", flush=True)
if not ok:
    print("REPRODUCTION MISMATCH > 0.005 -- STOPPING.", flush=True)
    sys.exit(1)

# ================================================================ STAGE 2
print("\n=== STAGE 2: combo variants, 5-fold CV on TRAIN only ===", flush=True)
for name in ["pair1_inter_knn", "pair2_inter_suburb10", "triple"]:
    cv_variant(name, VARIANTS[name])

# ================================================================ STAGE 3
print("\n=== STAGE 3: single test evaluation of CV winner ===", flush=True)
ridge_cv = results["cv"]["ridge"]
best = max(ridge_cv, key=lambda k: ridge_cv[k]["mean"])
print(f"CV ranking (Ridge): " +
      ", ".join(f"{k}={ridge_cv[k]['mean']:.4f}"
                for k in sorted(ridge_cv, key=lambda k: -ridge_cv[k]["mean"])),
      flush=True)
print(f"Winner by CV: {best} ({ridge_cv[best]['mean']:.4f})", flush=True)
if best != "triple":
    print(f"TRIPLE is NOT the best: triple CV {ridge_cv['triple']['mean']:.4f} "
          f"vs winner {best} CV {ridge_cv[best]['mean']:.4f} -- "
          f"triple will NOT be evaluated on test.", flush=True)

t = time.time()
pipe = make_pipe(VARIANTS[best])
pipe.fit(X_tr, y_tr)
pred = pipe.predict(X_te)
m = metrics(f"Ridge winner [{best}] test", y_te, pred, yraw_te)
m["variant"] = best
m["alpha"] = float(pipe[-1].alpha_)
m["R2_train_log"] = round(float(r2_score(y_tr, pipe.predict(X_tr))), 4)
m["CV_R2_mean"] = ridge_cv[best]["mean"]
m["CV_R2_std"] = ridge_cv[best]["std"]
m["train_sec"] = round(time.time() - t, 1)
results["test_winner"]["ridge"] = m
save()

print("\nDONE. Full results in", RES_PATH, flush=True)
print(json.dumps(results, indent=2), flush=True)
