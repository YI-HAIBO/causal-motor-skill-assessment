# -*- coding: utf-8 -*-
"""Same-setup uncontrolled contrast (pooled features, LOEO, ERM vs GRL, 1 seed).
Makes the controlled-vs-uncontrolled comparison rigorous within one feature space."""
import json, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from collections import defaultdict
from pathlib import Path

BASE = Path("C:/Users/Yi/WorkBuddy/2026-11-09-19-01-01")
SKEL = json.load(open(BASE / "outputs/stgcn_irm/skeleton_sequences.json", encoding="utf-8"))
OUT = BASE / "outputs/limitations"; OUT.mkdir(parents=True, exist_ok=True)
T, J, C = 16, 33, 4
CATEGORIES = ["upper arms","upper legs","back","waist","chest","shoulders","lower legs","lower arms","cardio","neck"]
CAT_IDX = {c: i for i, c in enumerate(CATEGORIES)}
rows = []
for s in SKEL:
    sk = np.array(s["skeleton"], dtype=np.float32)
    if sk.shape == (T, J, C) and sk[..., 3].sum() > 1e-6 and s["category"] in CAT_IDX:
        rows.append((s["id"], s["category"], s["equipment"], sk))
def feats(sk):
    return np.concatenate([sk.mean(0).reshape(-1), sk.std(0).reshape(-1)]).astype(np.float32)
EQ_GROUPS = ["cable","barbell","body weight","band","dumbbell"]
EQ_IDX = {e: i for i, e in enumerate(EQ_GROUPS)}
data = defaultdict(list)
for sid, cat, eq, sk in rows:
    if eq in EQ_IDX: data[(eq, cat)].append(feats(sk))

class Enc(nn.Module):
    def __init__(s, d, h=64):
        super().__init__(); s.fc1 = nn.Linear(d, h)
    def forward(s, x): return F.relu(s.fc1(x))
class Head(nn.Module):
    def __init__(s, h=64, nc=10):
        super().__init__(); s.drop = nn.Dropout(0.3); s.fc = nn.Linear(h, nc)
    def forward(s, h): return s.fc(s.drop(h))
class Disc(nn.Module):
    def __init__(s, h=64, ne=5): super().__init__(); s.fc = nn.Linear(h, ne)
    def forward(s, h): return s.fc(h)

def train_one(train, test, variant, seed, epochs=60, lam=0.3):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    Xtr = np.array([t[0] for t in train]); ytr = np.array([t[1] for t in train])
    etr = np.array([t[2] for t in train])
    Xte = np.array([t[0] for t in test]); yte = np.array([t[1] for t in test])
    mu = Xtr.mean(0, keepdims=True); sd = Xtr.std(0, keepdims=True) + 1e-6
    Xtr = (Xtr-mu)/sd; Xte = (Xte-mu)/sd
    enc = Enc(Xtr.shape[1]); head = Head(); disc = Disc() if variant == "grl" else None
    params = list(enc.parameters()) + list(head.parameters()) + (list(disc.parameters()) if disc else [])
    opt = torch.optim.Adam(params, lr=1e-3); lf = nn.CrossEntropyLoss()
    idx = np.arange(len(Xtr))
    for ep in range(epochs):
        enc.train(); head.train()
        if disc: disc.train()
        np.random.shuffle(idx)
        for b in range(0, len(idx), 32):
            bi = idx[b:b+32]; opt.zero_grad()
            h = enc(torch.tensor(Xtr[bi])); logits = head(h)
            loss = lf(logits, torch.tensor(ytr[bi]))
            if disc: loss = loss - lam * lf(disc(h), torch.tensor(etr[bi]))
            loss.backward(); opt.step()
    enc.eval(); head.eval()
    with torch.no_grad():
        ptr = head(enc(torch.tensor(Xtr))).argmax(1).numpy()
        pte = head(enc(torch.tensor(Xte))).argmax(1).numpy()
    return float((ptr == ytr).mean()), float((pte == yte).mean())

def run_uncontrolled(variant, seed=0):
    degs = []; trs = []; tes = []
    for held in EQ_GROUPS:
        src = [e for e in EQ_GROUPS if e != held]
        shared = [c for c in CATEGORIES if len(data.get((held,c),[]))>0 and sum(len(data.get((e,c),[])) for e in src)>0]
        if not shared: continue
        train = []; test = []
        for c in shared:
            test.extend((f, CAT_IDX[c], EQ_IDX[held]) for f in data[(held,c)])
            train.extend((f, CAT_IDX[c], EQ_IDX[e]) for e in src for f in data[(e,c)])
        if len(train) < 20 or len(test) < 10: continue
        tr, te = train_one(train, test, variant, seed)
        degs.append(tr-te); trs.append(tr); tes.append(te)
    return {"mean_tr": round(float(np.mean(trs)),4), "mean_te": round(float(np.mean(tes)),4),
            "mean_degradation": round(float(np.mean(degs)),4), "n_folds": len(degs)}

res = {}
for v in ["erm", "grl"]:
    res[f"uncontrolled_{v}"] = run_uncontrolled(v, seed=0)
    print(v, res[f"uncontrolled_{v}"])
json.dump(res, open(OUT / "cross_equipment_uncontrolled.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("saved")
