"""Melbourne preprocessing optimisation ROUND 2 (opt2_melb.py).
Question: can anything else on top of the TRIPLE (interactions + KNN k=5 +
suburb>=10) push Ridge R2_raw past 0.80? Triple stands at CV 0.8144, test
R2_log 0.8183 / R2_raw 0.7902, MAE 184,621 AUD.

Variants, ALL built on triple, Ridge only in the CV stage:
- V1: KNN k=10 (was 5)
- V2: KNN k=15
- V3: IterativeImputer (max_iter=5, BayesianRidge) instead of KNN.
  Pre-run probe: one fit_transform on 8,691 rows took 0.4s, so no skip
  expected; the >25 min guard below still applies.
- V4: suburb>=20 (was 10)
- V5: suburb>=50
- V6: triple + extra interactions Distance x Rooms and Lattitude x Rooms
- V7: triple + target encoding of Regionname (OOF means, smoothing m=10,
  exactly like the Suburb encoding in round 1)
- V8: triple with zero-drop: drop TRAIN rows where Landsize==0 or
  BuildingArea==0. Drop happens inside each CV fold (fold-train only) and on
  the final full train; TEST is untouched so test numbers stay comparable.
  Implemented as a manual CV loop (a Pipeline step cannot drop rows without
  breaking X/y alignment).
- V9: triple with RobustScaler replacing StandardScaler (Ridge only)
- V10: triple + PolynomialFeatures degree 2 on the 10 continuous numerics
  only, after imputation and before scaling; the 3 manual interactions stay
  as separate features.

Protocol (same as opt_melb.py / combo_melb.py):
- Stage 1: reproduce triple Ridge 5-fold CV on TRAIN; expect 0.8144 +/-0.002.
  If off, STOP.
- Stage 2: 5-fold CV (KFold shuffle, random_state=42) on TRAIN only, Ridge.
  All preprocessing (grouping, winsorise, medians, scalers, TE mappings,
  imputation, zero-drop) refit inside each fold. Target encoding strictly
  out-of-fold.
- Stage 3: winner = highest Ridge CV R2_log. If best beats triple by < 0.005,
  declare a tie / no meaningful improvement (test eval still runs once).
- Stage 4: touch TEST exactly once: CV winner (Ridge) on test, then RF on the
  winning preprocessing only: 150-tree 5-fold CV + one 300-tree test fit.
- Never touches results.json / REPORT_DATA.md / model_melb.py /
  opt_results.json / combo_results.json.
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
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import BayesianRidge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, RobustScaler, StandardScaler

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OPT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "notebooks", "melb_data.csv")
os.makedirs(OPT, exist_ok=True)
RES_PATH = f"{OPT}/opt2_results.json"

EXPECTED_TRIPLE_CV = 0.8144
TOL = 0.002
TIE_MARGIN = 0.005
V3_TIME_LIMIT = 1500  # seconds; skip V3 if its CV exceeds this

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
# Raw Landsize/BuildingArea are kept only so V8 can build its drop mask.
RAW_COLS = NUM + ["Landsize", "BuildingArea", "Suburb", "Type", "Method",
                  "Regionname", "CouncilArea"]

X = df[RAW_COLS]
y = df["y"]
y_raw = df["Price"]
X_tr, X_te, y_tr, y_te, yraw_tr, yraw_te = train_test_split(
    X, y, y_raw, test_size=0.2, random_state=42)
print(f"train {len(X_tr)} | test {len(X_te)}", flush=True)
print(f"V8 zero counts on train: Landsize==0: {(X_tr['Landsize'] == 0).sum()}, "
      f"BuildingArea==0: {(X_tr['BuildingArea'] == 0).sum()}, "
      f"either: {((X_tr['Landsize'] == 0) | (X_tr['BuildingArea'] == 0)).sum()}",
      flush=True)

results = {"reproduction": {}, "cv": {"ridge": {}, "rf_150trees": {}},
           "test_winner": {}, "verdict": {}}


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


# ------------------------------------------------- prep (extends combo MelbPrep)
def out_columns(prep, poly=False):
    """Return (poly_cols, plain_num_cols, cat_cols) for the ColumnTransformer."""
    plain = []
    if prep.get("use_flags", True):
        plain += ["BuildingArea_missing", "YearBuilt_missing"]
    if prep.get("interactions", False):
        plain += ["lat_long", "dist_Type_h", "dist_Type_t", "dist_Type_u",
                  "logland_rooms"]
    if prep.get("extra_interactions", False):
        plain += ["dist_rooms", "lat_rooms"]
    if prep.get("te_region", False):
        plain += ["Regionname_te"]
    cat = ["Type", "Method", "CouncilArea"]
    if not prep.get("te_region", False):
        cat += ["Regionname"]
    cat += ["Suburb_grouped"]
    if poly:
        return list(CONT_NUM), plain, cat
    return list(CONT_NUM) + plain, [], cat


class MelbPrep2(BaseEstimator, TransformerMixin):
    """Triple prep + round-2 options. Input: raw X (RAW_COLS)."""

    def __init__(self, suburb_min=10, imputer="knn", knn_k=5,
                 interactions=True, extra_interactions=False,
                 te_region=False, use_flags=True):
        self.suburb_min = suburb_min
        self.imputer = imputer
        self.knn_k = knn_k
        self.interactions = interactions
        self.extra_interactions = extra_interactions
        self.te_region = te_region
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
        if self.imputer is not None:
            tmp = self._prep_numeric(X)
            self.imp_mean_ = tmp[CONT_NUM].mean()
            self.imp_std_ = tmp[CONT_NUM].std(ddof=0).replace(0, 1.0)
        if self.te_region:
            yv = np.asarray(y, dtype=float)
            keys = X["Regionname"].astype(str).to_numpy()
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
        num = X[CONT_NUM].copy()
        if self.imputer is not None:
            fill = {c: self.medians_[c] for c in CONT_NUM if c not in KNN_COLS}
            num = num.fillna(fill)
        else:
            num = num.fillna(self.medians_)
        return num.clip(lower=self.lo_, upper=self.hi_, axis=1)

    def _transform(self, X, oof=False):
        X = X.reset_index(drop=True)
        num = self._prep_numeric(X)
        if self.imputer == "knn":
            sc = (num[CONT_NUM] - self.imp_mean_) / self.imp_std_
            imp = KNNImputer(n_neighbors=self.knn_k)
            sc_imp = pd.DataFrame(imp.fit_transform(sc), columns=CONT_NUM,
                                  index=sc.index)
            num = sc_imp * self.imp_std_ + self.imp_mean_
        elif self.imputer == "iterative":
            sc = (num[CONT_NUM] - self.imp_mean_) / self.imp_std_
            imp = IterativeImputer(max_iter=5, estimator=BayesianRidge(),
                                   random_state=42)
            sc_imp = pd.DataFrame(imp.fit_transform(sc), columns=CONT_NUM,
                                  index=sc.index)
            num = sc_imp * self.imp_std_ + self.imp_mean_
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
        if self.extra_interactions:
            out["dist_rooms"] = num["Distance"] * num["Rooms"]
            out["lat_rooms"] = num["Lattitude"] * num["Rooms"]
        if self.te_region:
            keys = X["Regionname"].astype(str)
            if oof:
                out["Regionname_te"] = self.oof_te_
            else:
                out["Regionname_te"] = keys.map(self.te_map_).fillna(
                    self.te_global_).to_numpy()
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


def make_pipe(kind, prep_params, scaler="standard", poly=False, rf_trees=300):
    poly_cols, plain_cols, cat_cols = out_columns(prep_params, poly=poly)
    prep = MelbPrep2(**prep_params)
    if kind == "ridge":
        sc = RobustScaler() if scaler == "robust" else StandardScaler()
        branches = []
        if poly:
            branches.append(("num_poly", Pipeline(
                [("poly", PolynomialFeatures(degree=2, include_bias=False)),
                 ("scaler", sc)]), poly_cols))
            branches.append(("num_plain", StandardScaler()
                             if scaler != "robust" else RobustScaler(),
                             plain_cols))
        else:
            branches.append(("num", sc, poly_cols + plain_cols))
        branches.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))
        model = RidgeCV(alphas=[0.1, 1, 10, 100, 1000], cv=5)
    else:
        branches = []
        if poly:
            branches.append(("num_poly", PolynomialFeatures(
                degree=2, include_bias=False), poly_cols))
            branches.append(("num_plain", "passthrough", plain_cols))
        else:
            branches.append(("num", "passthrough", poly_cols + plain_cols))
        branches.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))
        model = RandomForestRegressor(n_estimators=rf_trees, min_samples_leaf=2,
                                      n_jobs=-1, random_state=42)
    ct = ColumnTransformer(branches)
    return Pipeline([("prep", prep), ("ct", ct), ("model", model)])


KF = KFold(5, shuffle=True, random_state=42)
TRIPLE_PREP = {"interactions": True, "imputer": "knn", "knn_k": 5, "suburb_min": 10}
# variant -> (prep_params, scaler, poly, zero_drop)
VARIANTS = {
    "V1_knn10": (dict(TRIPLE_PREP, knn_k=10), "standard", False, False),
    "V2_knn15": (dict(TRIPLE_PREP, knn_k=15), "standard", False, False),
    "V3_iterative": (dict(TRIPLE_PREP, imputer="iterative"), "standard", False, False),
    "V4_suburb20": (dict(TRIPLE_PREP, suburb_min=20), "standard", False, False),
    "V5_suburb50": (dict(TRIPLE_PREP, suburb_min=50), "standard", False, False),
    "V6_extra_inter": (dict(TRIPLE_PREP, extra_interactions=True), "standard", False, False),
    "V7_te_region": (dict(TRIPLE_PREP, te_region=True), "standard", False, False),
    "V8_zero_drop": (dict(TRIPLE_PREP), "standard", False, True),
    "V9_robust": (dict(TRIPLE_PREP), "robust", False, False),
    "V10_poly2": (dict(TRIPLE_PREP), "standard", True, False),
}


def zero_mask(Xdf):
    return (Xdf["Landsize"] == 0) | (Xdf["BuildingArea"] == 0)


def cv_variant(name, spec):
    prep_params, scaler, poly, zero_drop = spec
    t = time.time()
    if zero_drop:
        # Manual loop: a Pipeline step cannot drop rows without breaking
        # X/y alignment, so the drop is applied to each fold-train here.
        scores = []
        for tr_idx, va_idx in KF.split(X_tr):
            Xf, yf = X_tr.iloc[tr_idx], y_tr.iloc[tr_idx]
            keep = ~zero_mask(Xf)
            pipe = make_pipe("ridge", prep_params, scaler=scaler, poly=poly)
            pipe.fit(Xf[keep], yf[keep])
            scores.append(r2_score(y_tr.iloc[va_idx],
                                   pipe.predict(X_tr.iloc[va_idx])))
        scores = np.array(scores)
    else:
        pipe = make_pipe("ridge", prep_params, scaler=scaler, poly=poly)
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
print("\n=== STAGE 1: reproduce triple Ridge CV ===", flush=True)
rep = cv_variant("triple", (TRIPLE_PREP, "standard", False, False))
ok = abs(rep["mean"] - EXPECTED_TRIPLE_CV) <= TOL
results["reproduction"] = {
    "triple_cv_mean": rep["mean"], "expected": EXPECTED_TRIPLE_CV,
    "within_tol": bool(ok)}
save()
print(f"Reproduction check: {'PASS' if ok else 'FAIL'} "
      f"({rep['mean']:.4f} vs expected {EXPECTED_TRIPLE_CV})", flush=True)
if not ok:
    print("REPRODUCTION MISMATCH > 0.002 -- STOPPING.", flush=True)
    sys.exit(1)

# ================================================================ STAGE 2
print("\n=== STAGE 2: round-2 variants, 5-fold CV on TRAIN only ===", flush=True)
for name, spec in VARIANTS.items():
    t0 = time.time()
    cv_variant(name, spec)
    if name == "V3_iterative" and time.time() - t0 > V3_TIME_LIMIT:
        # Guard only (probe showed ~0.4s per imputer fit, so not expected):
        # the time is already spent, but the variant is marked skipped so the
        # winner selection ignores it.
        results["cv"]["ridge"]["V3_iterative"] = {
            "mean": None, "std": None, "folds": [],
            "seconds": round(time.time() - t0, 1),
            "skipped": "exceeded 25 min time limit"}
        save()
        print("V3_iterative exceeded 25 min -- marked SKIPPED.", flush=True)

# ================================================================ STAGE 3
print("\n=== STAGE 3: pick CV winner ===", flush=True)
ridge_cv = {k: v for k, v in results["cv"]["ridge"].items()
            if v.get("mean") is not None}
ranking = sorted(ridge_cv, key=lambda k: -ridge_cv[k]["mean"])
print("CV ranking (Ridge): " +
      ", ".join(f"{k}={ridge_cv[k]['mean']:.4f}" for k in ranking), flush=True)
best = ranking[0]
gain = ridge_cv[best]["mean"] - ridge_cv["triple"]["mean"]
tie = gain < TIE_MARGIN
print(f"Winner by CV: {best} ({ridge_cv[best]['mean']:.4f}), "
      f"gain over triple: {gain:+.4f} -> "
      f"{'TIE / no meaningful improvement' if tie else 'meaningful improvement'}",
      flush=True)
results["verdict"] = {
    "winner": best,
    "winner_cv": ridge_cv[best]["mean"],
    "triple_cv": ridge_cv["triple"]["mean"],
    "gain_over_triple": round(float(gain), 4),
    "tie": bool(tie),
    "tie_margin": TIE_MARGIN}

# ================================================================ STAGE 4
print("\n=== STAGE 4: single test evaluation of CV winner ===", flush=True)
prep_params, scaler, poly, zero_drop = VARIANTS[best]

t = time.time()
pipe_r = make_pipe("ridge", prep_params, scaler=scaler, poly=poly)
if zero_drop:
    keep = ~zero_mask(X_tr)
    print(f"V8 final fit: dropped {int((~keep).sum())} zero rows from full train",
          flush=True)
    pipe_r.fit(X_tr[keep], y_tr[keep])
else:
    pipe_r.fit(X_tr, y_tr)
pred_r = pipe_r.predict(X_te)
m = metrics(f"Ridge winner [{best}] test", y_te, pred_r, yraw_te)
m["variant"] = best
m["alpha"] = float(pipe_r[-1].alpha_)
m["R2_train_log"] = round(float(r2_score(
    y_tr[~zero_mask(X_tr)] if zero_drop else y_tr,
    pipe_r.predict(X_tr[~zero_mask(X_tr)] if zero_drop else X_tr))), 4)
m["CV_R2_mean"] = ridge_cv[best]["mean"]
m["CV_R2_std"] = ridge_cv[best]["std"]
m["train_sec"] = round(time.time() - t, 1)
results["test_winner"]["ridge"] = m
save()

# RF on the winning preprocessing only: 150-tree CV, then 300-tree test fit.
print(f"\nRF on winner [{best}]: 150-tree 5-fold CV on TRAIN...", flush=True)
t = time.time()
pipe_rf_cv = make_pipe("rf", prep_params, scaler=scaler, poly=poly, rf_trees=150)
if zero_drop:
    scores_rf = []
    for tr_idx, va_idx in KF.split(X_tr):
        Xf, yf = X_tr.iloc[tr_idx], y_tr.iloc[tr_idx]
        keep = ~zero_mask(Xf)
        p = make_pipe("rf", prep_params, scaler=scaler, poly=poly, rf_trees=150)
        p.fit(Xf[keep], yf[keep])
        scores_rf.append(r2_score(y_tr.iloc[va_idx], p.predict(X_tr.iloc[va_idx])))
    scores_rf = np.array(scores_rf)
else:
    scores_rf = cross_val_score(pipe_rf_cv, X_tr, y_tr, cv=KF, scoring="r2")
el = time.time() - t
results["cv"]["rf_150trees"][best] = {
    "mean": round(float(scores_rf.mean()), 4),
    "std": round(float(scores_rf.std()), 4),
    "folds": [round(float(s), 4) for s in scores_rf],
    "seconds": round(el, 1)}
save()
print(f"RF(150 trees) CV [{best}]: {scores_rf.mean():.4f} ± {scores_rf.std():.4f} "
      f"({el:.0f}s)", flush=True)

t = time.time()
pipe_f = make_pipe("rf", prep_params, scaler=scaler, poly=poly, rf_trees=300)
if zero_drop:
    keep = ~zero_mask(X_tr)
    pipe_f.fit(X_tr[keep], y_tr[keep])
else:
    pipe_f.fit(X_tr, y_tr)
pred_f = pipe_f.predict(X_te)
m = metrics(f"RF winner [{best}] test (300 trees)", y_te, pred_f, yraw_te)
m["variant"] = best
m["R2_train_log"] = round(float(r2_score(
    y_tr[~zero_mask(X_tr)] if zero_drop else y_tr,
    pipe_f.predict(X_tr[~zero_mask(X_tr)] if zero_drop else X_tr))), 4)
m["CV_R2_mean"] = results["cv"]["rf_150trees"][best]["mean"]
m["CV_R2_std"] = results["cv"]["rf_150trees"][best]["std"]
m["note"] = "CV used 150 trees; final test fit uses 300 trees as in original run"
m["train_sec"] = round(time.time() - t, 1)
results["test_winner"]["rf"] = m
save()

print("\nDONE. Full results in", RES_PATH, flush=True)
print(json.dumps(results, indent=2), flush=True)
