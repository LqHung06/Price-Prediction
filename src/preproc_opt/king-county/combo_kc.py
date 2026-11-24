"""King County preprocessing COMBO round (combo_kc.py).
Question: is the TRIPLE combo the best for King County, like it was for
Melbourne -- and does stacking help at all here, given both models already
pass 0.80 and RF already captures interactions on its own?

ADJUSTMENTS vs Melbourne (recorded explicitly, because the datasets differ):
- KNN impute is a NO-OP here: the file has ZERO missing cells, so there is
  nothing to impute. It is NOT counted as a real intervention. Instead the
  third intervention is split into TWO real ones for this dataset:
  (b) zipcode grouping: zipcodes with <100 train rows -> "Other".
      The Melbourne <30 threshold would be a no-op here (smallest zipcode
      has 39 train rows); <100 groups 8 zips / 593 train rows (3.4%), whose
      coefficients are the noisiest (e.g. 98039: 39 rows, yet the largest
      Ridge coef +0.79).
  (c) quadratic term log_sqft_living^2: diminishing marginal returns to
      living area (each extra sqft adds less log-price).
- Components: (a) interactions, (b) zip grouping, (c) quadratic.
  Variants: base, inter (a only), pair1 (a+b), pair2 (a+c), triple (a+b+c).
- Interactions for Ridge (3, each with a reason):
  lat*long -- the price gradient over Seattle is not separable in lat/long
    (the expensive core sits north-east; the product captures the diagonal).
  log_sqft_living*grade -- the size premium depends on quality grade
    (a big low-grade house != a big high-grade house).
  waterfront*view -- a waterfront home WITH a good view commands a premium
    beyond the sum of the two (only 163 waterfront homes).
- RF CV uses 150 trees (like Melbourne); final test fits use 300 trees.

Protocol (same as Melbourne rounds):
- Stage 1: reproduce the original pipeline on test; expect Ridge R2_log
  0.8793 and RF R2_log 0.8867 (+/-0.002). If off, STOP.
- Stage 2: 5-fold CV (KFold shuffle, random_state=42) on TRAIN only for all
  variants, Ridge + RF. All preprocessing learned inside each fold.
- Stage 3: pick the CV winner PER MODEL (Ridge by Ridge CV, RF by RF CV)
  and evaluate each ONCE on test. If triple does not win CV for a model,
  say so plainly and do NOT evaluate triple on test for that model.
- Never touches results.json / REPORT_DATA.md / model_kc.py.
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
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OPT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "notebooks", "kc_house_data.csv")
os.makedirs(OPT, exist_ok=True)
RES_PATH = f"{OPT}/combo_results.json"

EXPECTED_RIDGE_TEST = 0.8793
EXPECTED_RF_TEST = 0.8867
TOL = 0.002
RF_CV_TREES = 150
RF_TEST_TREES = 300

# ---------------------------------------------------------------- load
df = pd.read_csv(DATA)
df["Age"] = 2015 - df["yr_built"]
df["renovated"] = (df["yr_renovated"] > 0).astype(int)
df["yrs_since_renov"] = np.where(df["renovated"] == 1, 2015 - df["yr_renovated"], 0)
df["total_rooms"] = df["bedrooms"] + df["bathrooms"]
df["sqft_per_room"] = df["sqft_living"] / df["total_rooms"].clip(lower=1)
for c in ["sqft_living", "sqft_lot", "sqft_above", "sqft_living15", "sqft_lot15"]:
    df[f"log_{c}"] = np.log(df[c])
df["y"] = np.log(df["price"])

NUM = ["log_sqft_living", "log_sqft_lot", "log_sqft_above", "log_sqft_living15",
       "log_sqft_lot15", "bedrooms", "bathrooms", "floors", "waterfront", "view",
       "condition", "grade", "sqft_basement", "Age", "renovated", "yrs_since_renov",
       "total_rooms", "sqft_per_room", "lat", "long"]
CAT = ["zipcode"]
FEATS = NUM + CAT

X = df[FEATS]
y = df["y"]
y_raw = df["price"]
X_tr, X_te, y_tr, y_te, yraw_tr, yraw_te = train_test_split(
    X, y, y_raw, test_size=0.2, random_state=42)
print(f"train {len(X_tr)} | test {len(X_te)}", flush=True)

results = {
    "design_notes": {
        "knn_impute": "NO-OP on this dataset (0 missing cells) -- NOT counted as an intervention",
        "component_a_interactions": ["lat*long", "log_sqft_living*grade", "waterfront*view"],
        "component_b_zip_grouping": "zipcodes with <100 train rows -> 'Other' (8 zips, 593 train rows, 3.4%); Melbourne <30 threshold would group 0 zips here (smallest zip has 39 train rows)",
        "component_c_quadratic": "log_sqft_living^2 (diminishing returns to living area)",
        "variants": {
            "base": "original pipeline",
            "interactions_single": "a only",
            "pair1_inter_zipgroup": "a + b",
            "pair2_inter_quad": "a + c",
            "triple": "a + b + c",
        },
        "rf_cv_trees": RF_CV_TREES,
        "rf_test_trees": RF_TEST_TREES,
        "baseline_expected": {"ridge_R2_log": EXPECTED_RIDGE_TEST, "rf_R2_log": EXPECTED_RF_TEST},
    },
    "reproduction": {},
    "cv": {"ridge": {}, "rf": {}},
    "test_winner": {},
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


def out_columns(interactions=False, quad=False):
    num = list(NUM)
    if interactions:
        num += ["lat_long", "logliving_grade", "waterfront_view"]
    if quad:
        num += ["log_sqft_living_sq"]
    return num, CAT


class KCPrep(BaseEstimator, TransformerMixin):
    """Fold-safe prep: median impute + winsorise (bounds learned in fit),
    optional interactions / quadratic / rare-zipcode grouping."""

    def __init__(self, interactions=False, zip_min=0, quad=False):
        self.interactions = interactions
        self.zip_min = zip_min
        self.quad = quad

    def fit(self, X, y=None):
        X = X.reset_index(drop=True)
        num = X[NUM]
        self.medians_ = num.median()
        self.lo_ = num.quantile(0.01)
        self.hi_ = num.quantile(0.99)
        if self.zip_min and self.zip_min > 0:
            vc = X["zipcode"].value_counts()
            self.keep_zips_ = set(vc[vc >= self.zip_min].index)
            self.n_grouped_zips_ = int((vc < self.zip_min).sum())
            self.n_grouped_rows_ = int(vc[vc < self.zip_min].sum())
        else:
            self.keep_zips_ = None
            self.n_grouped_zips_ = 0
            self.n_grouped_rows_ = 0
        return self

    def _transform(self, X):
        X = X.reset_index(drop=True)
        num = X[NUM].fillna(self.medians_).clip(
            lower=self.lo_, upper=self.hi_, axis=1)
        out = pd.DataFrame(index=X.index)
        for c in NUM:
            out[c] = num[c].to_numpy()
        if self.interactions:
            out["lat_long"] = num["lat"] * num["long"]
            out["logliving_grade"] = num["log_sqft_living"] * num["grade"]
            out["waterfront_view"] = num["waterfront"] * num["view"]
        if self.quad:
            out["log_sqft_living_sq"] = num["log_sqft_living"] ** 2
        if self.keep_zips_ is not None:
            out["zipcode"] = np.where(
                X["zipcode"].isin(self.keep_zips_), X["zipcode"], "Other")
        else:
            out["zipcode"] = X["zipcode"].to_numpy()
        return out

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self._transform(X)

    def transform(self, X):
        return self._transform(X)


def make_ridge(variant):
    num_cols, cat_cols = out_columns(
        interactions=variant.get("interactions", False),
        quad=variant.get("quad", False))
    prep = KCPrep(**variant)
    ct = ColumnTransformer(
        [("num", StandardScaler(), num_cols),
         ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)])
    return Pipeline([("prep", prep), ("ct", ct),
                     ("model", RidgeCV(alphas=[0.1, 1, 10, 100, 1000], cv=5))])


def make_rf(variant, n_trees):
    num_cols, cat_cols = out_columns(
        interactions=variant.get("interactions", False),
        quad=variant.get("quad", False))
    prep = KCPrep(**variant)
    ct = ColumnTransformer(
        [("num", "passthrough", num_cols),
         ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)])
    return Pipeline([("prep", prep), ("ct", ct),
                     ("model", RandomForestRegressor(
                         n_estimators=n_trees, min_samples_leaf=2,
                         n_jobs=-1, random_state=42))])


KF = KFold(5, shuffle=True, random_state=42)
VARIANTS = {
    "base": {},
    "interactions_single": {"interactions": True},
    "pair1_inter_zipgroup": {"interactions": True, "zip_min": 100},
    "pair2_inter_quad": {"interactions": True, "quad": True},
    "triple": {"interactions": True, "zip_min": 100, "quad": True},
}
ORDER = ["base", "interactions_single", "pair1_inter_zipgroup",
         "pair2_inter_quad", "triple"]


def cv_variant(model, name, var):
    t = time.time()
    pipe = make_ridge(var) if model == "ridge" else make_rf(var, RF_CV_TREES)
    scores = cross_val_score(pipe, X_tr, y_tr, cv=KF, scoring="r2")
    el = time.time() - t
    entry = {"mean": round(float(scores.mean()), 4),
             "std": round(float(scores.std()), 4),
             "folds": [round(float(s), 4) for s in scores],
             "seconds": round(el, 1)}
    results["cv"][model][name] = entry
    save()
    print(f"{model} CV [{name}]: {scores.mean():.4f} ± {scores.std():.4f} "
          f"folds={np.round(scores, 3)} ({el:.0f}s)", flush=True)
    return entry


# ================================================================ STAGE 1
print("\n=== STAGE 1: reproduce original pipeline on test ===", flush=True)
t = time.time()
ridge0 = make_ridge(VARIANTS["base"])
ridge0.fit(X_tr, y_tr)
pred_ridge0 = ridge0.predict(X_te)
m_ridge0 = metrics("Ridge base test", y_te, pred_ridge0, yraw_te)
m_ridge0["alpha"] = float(ridge0[-1].alpha_)
m_ridge0["R2_train_log"] = round(float(r2_score(y_tr, ridge0.predict(X_tr))), 4)
m_ridge0["train_sec"] = round(time.time() - t, 1)

t = time.time()
rf0 = make_rf(VARIANTS["base"], RF_TEST_TREES)
rf0.fit(X_tr, y_tr)
pred_rf0 = rf0.predict(X_te)
m_rf0 = metrics("RF base test", y_te, pred_rf0, yraw_te)
m_rf0["R2_train_log"] = round(float(r2_score(y_tr, rf0.predict(X_tr))), 4)
m_rf0["train_sec"] = round(time.time() - t, 1)

ok_r = abs(m_ridge0["R2_log"] - EXPECTED_RIDGE_TEST) <= TOL
ok_f = abs(m_rf0["R2_log"] - EXPECTED_RF_TEST) <= TOL
results["reproduction"] = {
    "ridge_R2_log": m_ridge0["R2_log"], "ridge_expected": EXPECTED_RIDGE_TEST,
    "ridge_within_tol": bool(ok_r),
    "rf_R2_log": m_rf0["R2_log"], "rf_expected": EXPECTED_RF_TEST,
    "rf_within_tol": bool(ok_f),
    "ridge_base_test": m_ridge0, "rf_base_test": m_rf0,
}
save()
print(f"Reproduction: Ridge {'PASS' if ok_r else 'FAIL'} "
      f"({m_ridge0['R2_log']:.4f} vs {EXPECTED_RIDGE_TEST}); "
      f"RF {'PASS' if ok_f else 'FAIL'} "
      f"({m_rf0['R2_log']:.4f} vs {EXPECTED_RF_TEST})", flush=True)
if not (ok_r and ok_f):
    print("REPRODUCTION MISMATCH > 0.002 -- STOPPING.", flush=True)
    sys.exit(1)

# zip grouping stats on full train (for the report)
tmp = KCPrep(zip_min=100).fit(X_tr)
print(f"zip grouping <100: {tmp.n_grouped_zips_} zips, "
      f"{tmp.n_grouped_rows_} train rows grouped to 'Other'", flush=True)
results["design_notes"]["zip_grouping_stats"] = {
    "threshold": 100, "n_grouped_zips": tmp.n_grouped_zips_,
    "n_grouped_rows_train": tmp.n_grouped_rows_}
save()

# ================================================================ STAGE 2
print("\n=== STAGE 2a: Ridge CV on TRAIN only ===", flush=True)
for name in ORDER:
    cv_variant("ridge", name, VARIANTS[name])

print("\n=== STAGE 2b: RF CV (150 trees) on TRAIN only ===", flush=True)
for name in ORDER:
    cv_variant("rf", name, VARIANTS[name])

# ================================================================ STAGE 3
print("\n=== STAGE 3: single test evaluation of CV winners ===", flush=True)
for model in ["ridge", "rf"]:
    table = results["cv"][model]
    best = max(table, key=lambda k: table[k]["mean"])
    ranking = ", ".join(f"{k}={table[k]['mean']:.4f}"
                        for k in sorted(table, key=lambda k: -table[k]["mean"]))
    print(f"CV ranking ({model}): {ranking}", flush=True)
    print(f"Winner by CV ({model}): {best} ({table[best]['mean']:.4f})", flush=True)
    if best != "triple":
        print(f"TRIPLE is NOT the best for {model}: triple CV "
              f"{table['triple']['mean']:.4f} vs winner {best} CV "
              f"{table[best]['mean']:.4f} -- triple will NOT be evaluated "
              f"on test for {model}.", flush=True)
    t = time.time()
    pipe = make_ridge(VARIANTS[best]) if model == "ridge" \
        else make_rf(VARIANTS[best], RF_TEST_TREES)
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    m = metrics(f"{model} winner [{best}] test", y_te, pred, yraw_te)
    m["variant"] = best
    if model == "ridge":
        m["alpha"] = float(pipe[-1].alpha_)
    m["R2_train_log"] = round(float(r2_score(y_tr, pipe.predict(X_tr))), 4)
    m["CV_R2_mean"] = table[best]["mean"]
    m["CV_R2_std"] = table[best]["std"]
    m["train_sec"] = round(time.time() - t, 1)
    results["test_winner"][model] = m
    save()

print("\nDONE. Full results in", RES_PATH, flush=True)
print(json.dumps(results, indent=2), flush=True)
