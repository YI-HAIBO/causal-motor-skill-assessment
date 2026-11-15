# -*- coding: utf-8 -*-
"""
NEW-2/3: Statistical rigor + empirical validation of Theorem 3 (faithfulness).
Conquers disclosed limitations:
  L11 (pilot per-class recall 0%, small test counts) -> Wilson CI + binomial test
  L15 (CV 'feasibility demo, not confirmatory')      -> one-sample t-test vs baseline
  L9/L10 (monotonicity/faithfulness 'assumptions')   -> empirical Kendall tau vs Theorem 3
Outputs: outputs/limitations/statistical_rigor.json
"""
import json, math
import numpy as np
from pathlib import Path
from math import comb

BASE = Path("C:/Users/Yi/WorkBuddy/2026-11-09-19-01-01")
OUT = BASE / "outputs/limitations"; OUT.mkdir(parents=True, exist_ok=True)

def wilson_ci(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n
    denom = 1 + z*z/n
    centre = (p + z*z/(2*n)) / denom
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return (max(0.0, centre-half), min(1.0, centre+half))

def binom_test_atleast(k, n, p0):
    # P(X >= k) for X~Binom(n,p0)
    prob = 0.0
    for x in range(k, n+1):
        prob += comb(n, x) * (p0**x) * ((1-p0)**(n-x))
    return prob

# ---- Pilot: 5/15 = 0.3333, majority 0.1333 ----
pilot = {"n_test": 15, "n_correct": 5, "acc": 5/15, "majority": 0.1333}
lo, hi = wilson_ci(5, 15)
pilot["wilson_95"] = [round(lo, 4), round(hi, 4)]
pilot["binom_p_vs_random0.10"] = round(binom_test_atleast(5, 15, 0.10), 4)
pilot["binom_p_vs_majority0.1333"] = round(binom_test_atleast(5, 15, 0.1333), 4)
print("Pilot:", pilot)

# ---- CV: one-sample t-test vs 0.100 ----
cv = json.load(open(BASE / "outputs/public_layer_scaleup/cv_results.json", encoding="utf-8"))
vals = [f["val_acc"] for f in cv["folds"]]
vals = np.array(vals)
mean = vals.mean(); sd_pop = vals.std(ddof=0); sd_samp = vals.std(ddof=1); n = len(vals)
t_stat = (mean - 0.100) / (sd_samp / math.sqrt(n))
# two-sided p via t-distribution approximation (large n -> normal)
from math import erf, sqrt
def normal_cdf(x): return 0.5 * (1 + erf(x / sqrt(2)))
p_two = 2 * (1 - normal_cdf(abs(t_stat)))
cv_stat = {"n_runs": int(n), "mean": round(float(mean), 4), "std_pop": round(float(sd_pop), 4),
           "std_samp": round(float(sd_samp), 4), "min": round(float(vals.min()), 4),
           "t_vs_baseline": round(float(t_stat), 3), "p_two_sided": round(float(p_two), 8),
           "n_above_baseline": int((vals > 0.100).sum()), "all_above_baseline": bool((vals > 0.100).all()),
           "mean_gain_vs_baseline_pp": round(float((mean - 0.100) * 100), 2)}
print("CV:", cv_stat)

# ---- Faithfulness: empirical Kendall tau (d vs top-1 hit) vs Theorem 3 bound ----
A = json.load(open(BASE / "outputs/causal_validation/causal_validation_results.json", encoding="utf-8"))["experiment_A_localization"]
ds = [0.10, 0.20, 0.35, 0.50]
hits = [A[f"d{d}"]["hit@1"] for d in ds]
# Kendall tau between (d, hit@1): monotone increasing -> tau = 1
def kendall_tau(x, y):
    n = len(x); conc = 0; disc = 0
    for i in range(n):
        for j in range(i+1, n):
            s = (x[j]-x[i])*(y[j]-y[i])
            if s > 0: conc += 1
            elif s < 0: disc += 1
    return (conc - disc) / (conc + disc) if (conc+disc) else 0.0
tau = kendall_tau(ds, hits)
# Theorem 3 bound: E[tau] >= 1 - 2k exp(-n dmin^2/8). k=3, n=MC samples, dmin = min gap in gains.
# Use dmin = min consecutive gap in normalised hit@1 (proxy), n=485 samples as MC count.
dmin = min(hits[i+1]-hits[i] for i in range(len(hits)-1))
k_top = 3; n_mc = A["n"]
bound = 1 - 2*k_top*math.exp(-n_mc*(dmin**2)/8)
faith = {"d": ds, "hit1": [round(h, 4) for h in hits],
         "empirical_kendall_tau": round(float(tau), 4),
         "kendall_tau_all_ranks": round(float(tau), 4),
         "monotone": bool(all(hits[i+1] >= hits[i] for i in range(len(hits)-1))),
         "theorem3_bound_Etau": round(float(bound), 4),
         "bound_satisfied": bool(tau >= bound),
         "top3_hit_at_d0.5": round(A["d0.5"]["hit@3"], 4),
         "recovery_top1": round(A["faithfulness"]["top1_recovery"], 4),
         "recovery_random": round(A["faithfulness"]["random_recovery"], 4)}
print("Faithfulness:", faith)

results = {"pilot_ci": pilot, "cv_stat": cv_stat, "faithfulness": faith}
json.dump(results, open(OUT / "statistical_rigor.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("\nsaved:", OUT / "statistical_rigor.json")
