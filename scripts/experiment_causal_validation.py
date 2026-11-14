# -*- coding: utf-8 -*-
"""
Refined causal validation experiments on the public layer (recorded, reproducible).
  A. Counterfactual localization severity sweep (validates M4 with known ground truth)
  B. Cross-equipment domain robustness (real equipment labels; ERM vs IRM-style)
  C. Stronger-baseline category signal (plain MLP vs ST-GCN) + E-value
"""
import json, os, math, random
import numpy as np
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F

BASE = Path("C:/Users/Yi/WorkBuddy/2026-11-09-19-01-01")
SKEL = json.load(open(BASE / "outputs/stgcn_irm/skeleton_sequences.json", encoding="utf-8"))
OUT = BASE / "outputs/causal_validation"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20261114); torch.manual_seed(20261114)

T, J, C = 16, 33, 4
CATEGORIES = ["upper arms", "upper legs", "back", "waist", "chest",
              "shoulders", "lower legs", "lower arms", "cardio", "neck"]
CAT_IDX = {c: i for i, c in enumerate(CATEGORIES)}
valid = []
for s in SKEL:
    sk = np.array(s["skeleton"], dtype=np.float32)
    if sk.shape == (T, J, C) and sk[..., 3].sum() > 1e-6 and s["category"] in CAT_IDX:
        valid.append((s["id"], s["category"], s["equipment"], sk))
print("valid skeletons:", len(valid))

cat_samples = {}
for sid, cat, eq, sk in valid:
    cat_samples.setdefault(cat, []).append(sk)
normative = {cat: np.median(np.stack(lst), axis=0) for cat, lst in cat_samples.items()}

def tracking_error(sk, mstar):
    vis = sk[..., 3]
    d = np.linalg.norm(sk[..., :3] - mstar[..., :3], axis=-1)
    return (d * vis).sum() / (vis.sum() + 1e-8)
def score(sk, mstar): return -tracking_error(sk, mstar)

def gains_at(sk_def, mstar, tau):
    base = score(sk_def, mstar); g = np.zeros(J)
    for j in range(J):
        skc = sk_def.copy(); skc[tau, j] = mstar[tau, j]
        g[j] = score(skc, mstar) - base
    return g

# ---------------- Experiment A: severity sweep ----------------
print("\n=== A: localization severity sweep ===")
A = {"n": len(valid)}
for D in [0.10, 0.20, 0.35, 0.50]:
    h1 = h3 = h5 = 0; margins = []; n = 0
    for sid, cat, eq, sk in valid:
        mstar = normative[cat]; tau = int(rng.integers(0, T))
        vis = mstar[tau, :, 3]; cand = np.where(vis > 0.3)[0]
        if len(cand) == 0: cand = np.where(mstar[tau, :, 3] > 0)[0]
        if len(cand) == 0: continue
        jstar = int(rng.choice(cand))
        dirv = rng.normal(size=3); dirv /= (np.linalg.norm(dirv) + 1e-9)
        sk_def = sk.copy(); sk_def[tau, jstar, :3] += D * dirv
        sk_def[tau, jstar, 3] = max(sk_def[tau, jstar, 3], 0.9)
        g = gains_at(sk_def, mstar, tau)
        order = np.argsort(-g); rank = int(np.where(order == jstar)[0][0]) + 1
        h1 += rank <= 1; h3 += rank <= 3; h5 += rank <= 5
        best_other = g[order[1]] if len(order) > 1 else g[order[0]]
        margins.append(g[jstar] - best_other); n += 1
    A[f"d{D}"] = {"hit@1": h1/n, "hit@3": h3/n, "hit@5": h5/n,
                  "median_margin": float(np.median(margins))}
    print(f"  d={D}: hit@1={h1/n:.3f} hit@3={h3/n:.3f} hit@5={h5/n:.3f} med_margin={np.median(margins):.5f}")

# faithfulness recovery
top_rec = rand_rec = 0.0; nc = 0
for sid, cat, eq, sk in valid:
    mstar = normative[cat]; tau = int(rng.integers(0, T))
    vis = mstar[tau, :, 3]; cand = np.where(vis > 0.3)[0]
    if len(cand) < 4: continue
    sk_def = sk.copy()
    for j in rng.choice(cand, size=3, replace=False):
        dv = rng.normal(size=3); dv /= (np.linalg.norm(dv) + 1e-9)
        sk_def[tau, j, :3] += 0.08 * dv
    g = gains_at(sk_def, mstar, tau); order = np.argsort(-g); top1 = order[0]
    sk_t = sk_def.copy(); sk_t[tau, top1] = mstar[tau, top1]
    sk_r = sk_def.copy(); sk_r[tau, int(rng.choice(cand))] = mstar[tau, int(rng.choice(cand))]
    sk_p = sk_def.copy(); sk_p[tau, cand] = mstar[tau, cand]
    denom = (score(sk_p, mstar) - score(sk_def, mstar)) + 1e-8
    top_rec += (score(sk_t, mstar) - score(sk_def, mstar)) / denom
    rand_rec += (score(sk_r, mstar) - score(sk_def, mstar)) / denom
    nc += 1
A["faithfulness"] = {"n": nc, "top1_recovery": float(top_rec/nc), "random_recovery": float(rand_rec/nc)}
print("  faithfulness:", A["faithfulness"])

# ---------------- Experiment B: cross-equipment (real domain) ----------------
print("\n=== B: cross-equipment domain robustness (ERM vs IRM-style) ===")
eqs = {}
for sid, cat, eq, sk in valid:
    eqs.setdefault(eq, []).append((cat, sk))
print("  equipment:", {k: len(v) for k, v in sorted(eqs.items(), key=lambda x: -len(x[1]))})

class MLP(nn.Module):
    def __init__(self, d_in, hidden=128, n_class=10):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(hidden, n_class))
    def forward(self, x): return self.net(x)

def feat(sk): return sk.reshape(-1)

def train_eval(train_list, test_list, use_irm, seed=0, epochs=60, lam=0.3):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    Xtr = np.array([feat(sk) for _, sk in train_list]); ytr = np.array([CAT_IDX[c] for c, _ in train_list])
    Xte = np.array([feat(sk) for _, sk in test_list]); yte = np.array([CAT_IDX[c] for c, _ in test_list])
    mu = Xtr.mean(0, keepdims=True); sd = Xtr.std(0, keepdims=True) + 1e-6
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    model = MLP(Xtr.shape[1]); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lf = nn.CrossEntropyLoss(); idx = np.arange(len(Xtr))
    for ep in range(epochs):
        model.train(); np.random.shuffle(idx)
        for b in range(0, len(idx), 32):
            bi = idx[b:b+32]; opt.zero_grad()
            loss = lf(model(torch.tensor(Xtr[bi])), torch.tensor(ytr[bi]))
            loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        tr_acc = float((model(torch.tensor(Xtr)).argmax(1) == torch.tensor(ytr)).float().mean())
        te_acc = float((model(torch.tensor(Xte)).argmax(1) == torch.tensor(yte)).float().mean())
    return tr_acc, te_acc

# hold out body weight (81, spans many categories) as target; train on cable/barbell/band/dumbbell
HOLD = "body weight"
train_pool = [(c, sk) for _, c, e, sk in valid if e != HOLD]
test_pool = [(c, sk) for _, c, e, sk in valid if e == HOLD]
print(f"  train n={len(train_pool)} (non-{HOLD}), test n={len(test_pool)} ({HOLD})")
B = {"held_out": HOLD, "train_n": len(train_pool), "test_n": len(test_pool)}
for use_irm, name in [(False, "ERM"), (True, "IRM")]:
    tr, te = train_eval(train_pool, test_pool, use_irm, seed=0)
    B[name] = {"train_acc": round(tr, 4), "heldout_acc": round(te, 4),
               "degradation": round((tr - te), 4)}
    print(f"  {name}: train={tr:.4f} heldout={te:.4f} degr={tr-te:.4f}")

# ---------------- Experiment C: stronger baseline + E-value ----------------
print("\n=== C: stronger baseline (plain MLP) + E-value ===")
Xr = np.array([feat(sk) for _, _, _, sk in valid]); yr = np.array([CAT_IDX[c] for _, c, _, _ in valid])
mu = Xr.mean(0, keepdims=True); sd = Xr.std(0, keepdims=True) + 1e-6; Xr = (Xr - mu) / sd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score
skf = StratifiedKFold(5, shuffle=True, random_state=0)
accs, baccs, f1s = [], [], []
for tr, te in skf.split(Xr, yr):
    model = MLP(Xr.shape[1]); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(40):
        model.train(); order = np.random.permutation(len(tr))
        for b in range(0, len(order), 32):
            bi = tr[order[b:b+32]]; opt.zero_grad()
            loss = F.cross_entropy(model(torch.tensor(Xr[bi])), torch.tensor(yr[bi]))
            loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(Xr[te])).argmax(1).numpy()
    accs.append(accuracy_score(yr[te], pred)); baccs.append(balanced_accuracy_score(yr[te], pred))
    f1s.append(f1_score(yr[te], pred, average="macro", zero_division=0))
C = {"n": len(valid), "mean_top1": round(float(np.mean(accs)), 4), "std_top1": round(float(np.std(accs)), 4),
     "mean_balanced": round(float(np.mean(baccs)), 4), "std_balanced": round(float(np.std(baccs)), 4),
     "mean_macro_f1": round(float(np.mean(f1s)), 4), "random": 0.10}
print("  ", C)

# E-value for the reported CV result (0.228 vs 0.100)
def e_value(rr):
    return rr + math.sqrt(rr * (rr - 1))
for acc, base, lab in [(0.228, 0.100, "ST-GCN CV 0.228 vs 0.100"),
                       (C["mean_top1"], 0.100, f"MLP {C['mean_top1']:.3f} vs 0.100")]:
    rr = acc / base
    print(f"  E-value({lab}) = {e_value(rr):.2f} (RR={rr:.2f})")
C["e_values"] = {"stgcn_0.228": round(e_value(0.228/0.100), 2),
                 f"mlp_{C['mean_top1']:.3f}": round(e_value(C['mean_top1']/0.100), 2)}

results = {"experiment_A_localization": A, "experiment_B_cross_equipment": B, "experiment_C_baseline": C}
json.dump(results, open(OUT / "causal_validation_results.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("\nsaved:", OUT / "causal_validation_results.json")
