"""Resume for opt2_melb.py after a service restart killed it mid-run.
What already finished and is saved in opt2_results.json:
- Stage 1 reproduction (triple CV 0.8144, PASS)
- Stage 2 all 10 Ridge variant CVs, winner V10_poly2
- Stage 4 Ridge test eval (V10_poly2, R2_log 0.8419 / R2_raw 0.8175)
- RF 150-tree CV on the winner (0.8528)
What is missing: ONLY the RF 300-tree single test fit.
This script runs exactly that one missing step and nothing else. It does NOT
re-run any Ridge evaluation and does NOT touch the test set for anything else.
It reuses the pipeline code from opt2_melb.py verbatim by exec'ing the file's
definition head (everything before STAGE 1), so the winner pipeline is
identical to the one the main script would have built.
"""
import json
import os
import time

OPT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(OPT, "opt2_melb.py")
RES_PATH = os.path.join(OPT, "opt2_results.json")

# Reuse opt2_melb.py definitions + data load/split verbatim.
with open(SRC) as f:
    src = f.read()
head = src.split("=" * 64 + " STAGE 1")[0]
ns = {"__name__": "opt2_defs"}
exec(compile(head, SRC, "exec"), ns)

results = json.load(open(RES_PATH))
best = results["verdict"]["winner"]
assert best == "V10_poly2", best
prep_params, scaler, poly, zero_drop = ns["VARIANTS"][best]
assert not zero_drop, "winner is not a zero-drop variant"

print(f"Resuming: RF 300-tree test fit for winner [{best}] (single evaluation)",
      flush=True)
t = time.time()
pipe_f = ns["make_pipe"]("rf", prep_params, scaler=scaler, poly=poly, rf_trees=300)
pipe_f.fit(ns["X_tr"], ns["y_tr"])
pred_f = pipe_f.predict(ns["X_te"])
m = ns["metrics"](f"RF winner [{best}] test (300 trees)",
                  ns["y_te"], pred_f, ns["yraw_te"])
m["variant"] = best
from sklearn.metrics import r2_score
m["R2_train_log"] = round(float(r2_score(ns["y_tr"], pipe_f.predict(ns["X_tr"]))), 4)
m["CV_R2_mean"] = results["cv"]["rf_150trees"][best]["mean"]
m["CV_R2_std"] = results["cv"]["rf_150trees"][best]["std"]
m["note"] = "CV used 150 trees; final test fit uses 300 trees as in original run"
m["train_sec"] = round(time.time() - t, 1)
results["test_winner"]["rf"] = m
with open(RES_PATH, "w") as f:
    json.dump(results, f, indent=2)

print("\nRESUME DONE. Full results in", RES_PATH, flush=True)
print(json.dumps(results, indent=2), flush=True)
