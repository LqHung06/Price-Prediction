"""Indian house preprocessing COMBO round (combo_house.py).

Question: is the TRIPLE combo (interactions + KNN impute + log_Area^2) the best
for Ridge / RF on this small dataset, or does stacking fail to add up?

Context: Ridge already 0.9427 R2_log test / 0.9176 CV; RF 0.8900 / 0.8474.
Headroom is small, so any gain is expected to be tiny; a null result is a
real result here.

Protocol (same spirit as Melbourne combo_melb.py):
- Stage 1: reproduce ORIGINAL pipeline test R2 exactly (Ridge 0.9427, RF 0.8900,
  tol +/-0.002). If off, STOP.
- Stage 2: 5-fold CV (KFold shuffle, random_state=42) on TRAIN only (899 rows)
  for base / inter / knn / quad / pair1 (inter+knn) / pair2 (inter+quad) /
  triple, Ridge AND RF. All preprocessing learned inside each fold.
- Stage 3: per-model CV winner evaluated ONCE on test.
- Never touches results.json / REPORT_DATA.md / model_house.py.

Third-intervention choice (recorded):
- Location has only 8 cities, each 127-156 rows, so a suburb>=10-style grouping
  threshold is a no-op and is NOT counted as an intervention.
- Target encoding for Location was REJECTED: OneHot already gives each of the
  8 cities its own coefficient (cheap and exact); collapsing them into one
  noisy number is exactly what HURT Ridge in Melbourne (-0.0045 CV).
- Chosen third intervention: log_Area^2 (quadratic term in log area). The one
  remaining nonlinear suspect is curvature in the dominant area-price relation
  (log_Area alone explains most of the price; a straight line in log-log space
  may miss diminishing/premium returns at the extremes).

Interactions (2, with reasons):
- log_Area x Rooms: the value of an extra room may depend on total area
  (a 5th room in a 700 sqft house vs a 3,000 sqft house).
- log_Area x Age: the area premium may decay with age (old large houses vs
  new large houses).

One deliberate deviation from the original pipeline: Area_per_Room is
recomputed from the IMPUTED Area and Rooms in every experiment variant
(instead of median-filling Area_per_Room independently as model_house.py did),
so that all variants share one coherent derivation. Stage 1 still reproduces
the original pipeline exactly, so the deviation is isolated to the experiment.
Note also: the original CV (0.9176) was computed over the FULL data with
winsorise limits from the 80% train; the base CV in this round is stricter
(fold-internal preprocessing, train only), so the two CV numbers are not
directly comparable -- the comparison that matters is between variants inside
this round, all measured the same strict way.
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
DATA = os.path.join(ROOT, "notebooks", "dataset_2.csv")
os.makedirs(OPT, exist_ok=True)
RES_PATH = f"{OPT}/combo_results.json"

EXPECTED = {"ridge": 0.9427, "rf": 0.8900}
TOL = 0.002

# ---------------------------------------------------------------- load
df = pd.read_csv(DATA)
RAW = ["Area_SqFt", "Rooms", "Build_Year", "Location", "Street_Type",
       "Furnishing", "Property_Type", "Has_Pool"]
CAT = ["Location", "Street_Type", "Furnishing", "Property_Type", "Has_Pool"]
NUM_BASE = ["log_Area", "Rooms", "Age", "Area_per_Room"]
KNN_COLS = ["log_Area", "Rooms", "Age"]

y = np.log(df["Price"])
y_raw = df["Price"]
X_raw = df[RAW]

X_tr, X_te, y_tr, y_te, yraw_tr, yraw_te = train_test_split(
    X_raw, y, y_raw, test_size=0.2, random_state=42)
print(f"train {len(X_tr)} | test {len(X_te)}", flush=True)

results = {"reproduction": {}, "cv": {"ridge": {}, "rf": {}},
           "cv_reference_original": {"ridge": 0.9176, "rf": 0.8474,
                                     "note": "original CV used full data + "
                                     "train winsorise limits; this round's "
                                     "base CV is stricter (fold-internal, "
                                     "train only) -- not directly comparable"},
           "test_winner": {},
           "notes": {
               "third_intervention": "log_Area^2 (quadratic in log area); "
               "target encoding for Location REJECTED (8 cities, OneHot "
               "already exact; Melbourne showed target enc hurts Ridge); "
               "suburb>=10-style grouping is a no-op here (all cities "
               ">100 rows)",
               "interactions": ["log_Area x Rooms", "log_Area x Age"],
               "area_per_room": "recomputed from imputed Area/Rooms in all "
               "experiment variants (original median-filled it independently)"},
           }


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


# ==================================================== STAGE 1: reproduce original
print("\n=== STAGE 1: reproduce ORIGINAL pipeline test R2 ===", flush=True)

df2 = df.copy()
df2["Age"] = 2025 - df2["Build_Year"]
df2["log_Area"] = np.log(df2["Area_SqFt"])
df2["Area_per_Room"] = df2["Area_SqFt"] / df2["Rooms"]
df2["y"] = np.log(df2["Price"])
FEATS = NUM_BASE + CAT
Xo = df2[FEATS]
yo = df2["y"]
yraw_o = df2["Price"]
Xo_tr, Xo_te, yo_tr, yo_te, yraw_tr_o, yraw_te_o = train_test_split(
    Xo, yo, yraw_o, test_size=0.2, random_state=42)
lo = Xo_tr[NUM_BASE].quantile(0.01).to_numpy()
hi = Xo_tr[NUM_BASE].quantile(0.99).to_numpy()


class Winsoriser(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        return np.clip(X, lo, hi)


def prep_orig(scaler):
    num_pipe = make_pipeline(SimpleImputer(strategy="median"), Winsoriser(),
                             scaler) if scaler else make_pipeline(
                                 SimpleImputer(strategy="median"), Winsoriser())
    cat_pipe = make_pipeline(SimpleImputer(strategy="most_frequent"),
                             OneHotEncoder(handle_unknown="ignore"))
    return ColumnTransformer([("num", num_pipe, NUM_BASE), ("cat", cat_pipe, CAT)])


ridge0 = make_pipeline(prep_orig(StandardScaler()),
                       RidgeCV(alphas=[0.1, 1, 10, 100, 1000], cv=5))
ridge0.fit(Xo_tr, yo_tr)
pred0 = ridge0.predict(Xo_te)
r_ridge = metrics("ORIGINAL Ridge test", yo_te, pred0, yraw_te_o)

rf0 = make_pipeline(prep_orig(None),
                    RandomForestRegressor(n_estimators=500, min_samples_leaf=2,
                                          n_jobs=-1, random_state=42))
rf0.fit(Xo_tr, yo_tr)
pred0_rf = rf0.predict(Xo_te)
r_rf = metrics("ORIGINAL RF test", yo_te, pred0_rf, yraw_te_o)

ok_r = abs(r_ridge["R2_log"] - EXPECTED["ridge"]) <= TOL
ok_f = abs(r_rf["R2_log"] - EXPECTED["rf"]) <= TOL
results["reproduction"] = {
    "ridge_R2_log": r_ridge["R2_log"], "ridge_expected": EXPECTED["ridge"],
    "ridge_within_tol": bool(ok_r),
    "rf_R2_log": r_rf["R2_log"], "rf_expected": EXPECTED["rf"],
    "rf_within_tol": bool(ok_f)}
save()
print(f"Reproduction check: Ridge {'PASS' if ok_r else 'FAIL'} "
      f"({r_ridge['R2_log']:.4f} vs {EXPECTED['ridge']}); "
      f"RF {'PASS' if ok_f else 'FAIL'} "
      f"({r_rf['R2_log']:.4f} vs {EXPECTED['rf']})", flush=True)
if not (ok_r and ok_f):
    print("REPRODUCTION MISMATCH > 0.002 -- STOPPING.", flush=True)
    sys.exit(1)


# ==================================================== experiment prep
def num_cols(variant):
    cols = list(NUM_BASE)
    if variant.get("interactions"):
        cols += ["inter_area_rooms", "inter_area_age"]
    if variant.get("quad"):
        cols += ["log_area_sq"]
    return cols


class HousePrep(BaseEstimator, TransformerMixin):
    """Fold-internal prep: derive -> impute (median or KNN, fitted on the
    fold-train only) -> recompute Area_per_Room -> winsorise -> add
    interactions / quadratic term -> mode-fill cats."""

    def __init__(self, knn=False, interactions=False, quad=False):
        self.knn = knn
        self.interactions = interactions
        self.quad = quad

    def _derive(self, X):
        X = X.reset_index(drop=True)
        D = pd.DataFrame(index=X.index)
        D["log_Area"] = np.log(X["Area_SqFt"])
        D["Rooms"] = X["Rooms"].astype(float)
        D["Age"] = 2025 - X["Build_Year"]
        return D

    def fit(self, X, y=None):
        X = X.reset_index(drop=True)
        D = self._derive(X)
        self.med_ = D.median()
        if self.knn:
            filled = D.fillna(self.med_)
            self.knn_mean_ = filled[KNN_COLS].mean()
            self.knn_std_ = filled[KNN_COLS].std(ddof=0).replace(0, 1.0)
            sc = (filled[KNN_COLS] - self.knn_mean_) / self.knn_std_
            self.knn_imp_ = KNNImputer(n_neighbors=5)
            # fit on the standardized frame WITH its NaNs (imputer handles them)
            raw_sc = (D[KNN_COLS] - self.knn_mean_) / self.knn_std_
            self.knn_imp_.fit(raw_sc)
        num = self._impute(D)
        num["Area_per_Room"] = np.exp(num["log_Area"]) / num["Rooms"]
        self.lo_ = num.quantile(0.01)
        self.hi_ = num.quantile(0.99)
        self.modes_ = {c: X[c].mode(dropna=True)[0] for c in CAT}
        return self

    def _impute(self, D):
        if self.knn:
            sc = (D[KNN_COLS] - self.knn_mean_) / self.knn_std_
            imp = pd.DataFrame(self.knn_imp_.transform(sc), columns=KNN_COLS,
                               index=D.index)
            out = imp * self.knn_std_ + self.knn_mean_
            return out
        return D.fillna(self.med_)

    def _transform(self, X):
        X = X.reset_index(drop=True)
        num = self._impute(self._derive(X))
        num["Area_per_Room"] = np.exp(num["log_Area"]) / num["Rooms"]
        num = num.clip(lower=self.lo_, upper=self.hi_, axis=1)
        if self.interactions:
            num["inter_area_rooms"] = num["log_Area"] * num["Rooms"]
            num["inter_area_age"] = num["log_Area"] * num["Age"]
        if self.quad:
            num["log_area_sq"] = num["log_Area"] ** 2
        out = num.copy()
        for c in CAT:
            out[c] = X[c].fillna(self.modes_[c]).astype(str).to_numpy()
        return out

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self._transform(X)

    def transform(self, X):
        return self._transform(X)


def make_pipe(variant, model_kind):
    prep = HousePrep(knn=variant.get("knn", False),
                     interactions=variant.get("interactions", False),
                     quad=variant.get("quad", False))
    if model_kind == "ridge":
        ct = ColumnTransformer(
            [("num", StandardScaler(), num_cols(variant)),
             ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
        model = RidgeCV(alphas=[0.1, 1, 10, 100, 1000], cv=5)
    else:
        ct = ColumnTransformer(
            [("num", "passthrough", num_cols(variant)),
             ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
        model = RandomForestRegressor(n_estimators=500, min_samples_leaf=2,
                                      n_jobs=-1, random_state=42)
    return Pipeline([("prep", prep), ("ct", ct), ("model", model)])


KF = KFold(5, shuffle=True, random_state=42)
VARIANTS = {
    "base": {},
    "inter": {"interactions": True},
    "knn": {"knn": True},
    "quad": {"quad": True},
    "pair1_inter_knn": {"interactions": True, "knn": True},
    "pair2_inter_quad": {"interactions": True, "quad": True},
    "triple": {"interactions": True, "knn": True, "quad": True},
}


def cv_variant(model_kind, name, var):
    t = time.time()
    pipe = make_pipe(var, model_kind)
    scores = cross_val_score(pipe, X_tr, y_tr, cv=KF, scoring="r2")
    el = time.time() - t
    entry = {"mean": round(float(scores.mean()), 4),
             "std": round(float(scores.std()), 4),
             "folds": [round(float(s), 4) for s in scores],
             "seconds": round(el, 1)}
    results["cv"][model_kind][name] = entry
    save()
    print(f"{model_kind} CV [{name}]: {scores.mean():.4f} ± {scores.std():.4f} "
          f"folds={np.round(scores, 3)} ({el:.0f}s)", flush=True)
    return entry


# ================================================================ STAGE 2
print("\n=== STAGE 2: combo variants, 5-fold CV on TRAIN only ===", flush=True)
for model_kind in ["ridge", "rf"]:
    for name, var in VARIANTS.items():
        cv_variant(model_kind, name, var)

# ================================================================ STAGE 3
print("\n=== STAGE 3: single test evaluation of each model's CV winner ===",
      flush=True)
for model_kind in ["ridge", "rf"]:
    cvd = results["cv"][model_kind]
    best = max(cvd, key=lambda k: cvd[k]["mean"])
    ranking = ", ".join(f"{k}={cvd[k]['mean']:.4f}"
                        for k in sorted(cvd, key=lambda k: -cvd[k]["mean"]))
    print(f"CV ranking ({model_kind}): {ranking}", flush=True)
    print(f"Winner by CV ({model_kind}): {best} ({cvd[best]['mean']:.4f})",
          flush=True)
    if best != "triple":
        print(f"TRIPLE is NOT the best for {model_kind}: triple CV "
              f"{cvd['triple']['mean']:.4f} vs winner {best} CV "
              f"{cvd[best]['mean']:.4f} -- triple will NOT be evaluated on "
              f"test for {model_kind}.", flush=True)
    t = time.time()
    pipe = make_pipe(VARIANTS[best], model_kind)
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    m = metrics(f"{model_kind} winner [{best}] test", y_te, pred, yraw_te)
    m["variant"] = best
    if model_kind == "ridge":
        m["alpha"] = float(pipe[-1].alpha_)
    m["R2_train_log"] = round(float(r2_score(y_tr, pipe.predict(X_tr))), 4)
    m["CV_R2_mean"] = cvd[best]["mean"]
    m["CV_R2_std"] = cvd[best]["std"]
    m["train_sec"] = round(time.time() - t, 1)
    results["test_winner"][model_kind] = m
    save()

print("\nDONE. Full results in", RES_PATH, flush=True)
print(json.dumps(results, indent=2), flush=True)
