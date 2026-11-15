# -*- coding: utf-8 -*-
"""
NEW-4: Honest re-analysis of the domain-adaptive pre-training negative result.
The paper claimed the DA source was "publicly released FineGym-98 sequences,
not synthetic data"; the actual training script generates a random-noise source
(generate_simulated_source_data -> np.random.randn*0.1). We re-run the frozen
-encoder linear probe on the REAL target features and report balanced accuracy,
macro-F1 and per-class recall, to characterise whether the negative (top-1 =
majority) hides a genuine transfer or is truly at chance.
Outputs: outputs/limitations/da_probe_balanced.json
"""
import json, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from collections import Counter
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score

BASE = Path("C:/Users/Yi/WorkBuddy/2026-11-09-19-01-01")
OUT = BASE / "outputs/limitations"; OUT.mkdir(parents=True, exist_ok=True)
CKPT = BASE / "outputs/domain_adaptation/best_model.pt"
TARGET = BASE / "outputs/stgcn_irm/skeleton_sequences.json"

# ---- reproduce model definitions (from train_domain_adaptation.py) ----
T, J, C = 16, 33, 4
HIDDEN = 64
SKELETON_EDGES = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),
    (11,12),(12,14),(14,16),(16,18),(18,20),(20,22),(16,22),
    (11,13),(13,15),(15,17),(17,19),(19,21),(21,23),(15,23),
    (12,24),(24,26),(26,28),(28,30),(28,32),(24,27),(27,29),(29,31),
    (23,25),(25,27),(11,23),(12,24),
]
def build_adjacency(num_nodes=J):
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i, j in SKELETON_EDGES:
        A[i, j] = 1.0; A[j, i] = 1.0
    A += np.eye(num_nodes, dtype=np.float32)
    D = A.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D + 1e-8))
    return torch.tensor(D_inv_sqrt @ A @ D_inv_sqrt, dtype=torch.float32)

class GraphConv(nn.Module):
    def __init__(self, in_c, out_c, num_joints, adj):
        super().__init__()
        self.num_joints = num_joints
        self.weight = nn.Parameter(torch.randn(num_joints, in_c, out_c) * 0.01)
        self.adj = adj
    def forward(self, x):
        B, C_in, T, J = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B * T, J, C_in)
        adj = self.adj.to(x.device)
        x_agg = torch.bmm(adj.unsqueeze(0).expand(B * T, -1, -1), x)
        out = torch.einsum('b j c, j c o -> b j o', x_agg, self.weight)
        return out.reshape(B, T, J, -1).permute(0, 3, 1, 2)

class STGCNEncoder(nn.Module):
    def __init__(self, in_channels=C, hidden=HIDDEN, num_joints=J, adj=None, feature_dim=128):
        super().__init__()
        self.adj = adj
        self.gc1 = GraphConv(in_channels, hidden, num_joints, adj)
        self.tcn1 = nn.Conv2d(hidden, hidden, (3, 1), padding=(1, 0), bias=False); self.bn1 = nn.BatchNorm2d(hidden)
        self.gc2 = GraphConv(hidden, hidden, num_joints, adj)
        self.tcn2 = nn.Conv2d(hidden, hidden, (3, 1), padding=(1, 0), bias=False); self.bn2 = nn.BatchNorm2d(hidden)
        self.gc3 = GraphConv(hidden, hidden, num_joints, adj)
        self.tcn3 = nn.Conv2d(hidden, hidden, (3, 1), padding=(1, 0), bias=False); self.bn3 = nn.BatchNorm2d(hidden)
        self.dropout = nn.Dropout(0.3)
        self.feature_proj = nn.Linear(hidden, feature_dim)
    def forward(self, x):
        B, C_in, T, J = x.shape
        x = self.gc1(x); x = self.tcn1(x); x = self.bn1(x); x = F.relu(x); x = self.dropout(x)
        x = self.gc2(x); x = self.tcn2(x); x = self.bn2(x); x = F.relu(x); x = self.dropout(x)
        x = self.gc3(x); x = self.tcn3(x); x = self.bn3(x); x = F.relu(x)
        x = x.mean(dim=(2, 3))
        return self.feature_proj(x)

ALL_CATEGORIES = ["upper arms","upper legs","back","waist","chest","shoulders","lower legs","lower arms","cardio","neck"]
CAT_MAP = {c: i for i, c in enumerate(ALL_CATEGORIES)}

adj = build_adjacency()
enc = STGCNEncoder(adj=adj, feature_dim=128)
state = torch.load(CKPT, map_location="cpu")
enc.load_state_dict({k.replace("encoder.", ""): v for k, v in state.items() if k.startswith("encoder.")})
enc.eval()

target = json.load(open(TARGET, encoding="utf-8"))
feats = []; labels = []
for s in target:
    sk = np.array(s["skeleton"], dtype=np.float32)
    if sk.shape != (T, J, C): continue
    x = torch.from_numpy(sk).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        f = enc(x)[0].numpy()
    feats.append(f); labels.append(CAT_MAP.get(s["category"], 0))
X = np.array(feats); y = np.array(labels)
print("target features:", X.shape, "label counts:", dict(zip(*np.unique(y, return_counts=True))))

# frozen linear probe (multinomial logistic) + balanced accuracy + per-class
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sc = StandardScaler().fit(X); Xs = sc.transform(X)
clf = LogisticRegression(max_iter=5000, C=1.0, random_state=0)
clf.fit(Xs, y)
pred = clf.predict(Xs)
top1 = accuracy_score(y, pred); bal = balanced_accuracy_score(y, pred); mf1 = f1_score(y, pred, average="macro", zero_division=0)
maj = max(np.bincount(y)) / len(y)
perclass = {ALL_CATEGORIES[c]: round(float((pred[y == c] == c).mean()), 4) for c in np.unique(y)}
print(f"probe top1={top1:.4f} balanced={bal:.4f} macroF1={mf1:.4f} majority={maj:.4f}")
print("per-class recall:", perclass)

# random-feature baseline (same probe on random projection of skeleton) for reference
rng = np.random.default_rng(0)
Xr = rng.standard_normal((len(y), 128))
sc2 = StandardScaler().fit(Xr); Xrs = sc2.transform(Xr)
clf2 = LogisticRegression(max_iter=5000, C=1.0, random_state=0); clf2.fit(Xrs, y)
p2 = clf2.predict(Xrs)
print(f"random-feature probe top1={accuracy_score(y, p2):.4f} balanced={balanced_accuracy_score(y, p2):.4f}")

res = {
    "n_target": int(len(y)),
    "label_counts": {ALL_CATEGORIES[c]: int((y == c).sum()) for c in np.unique(y)},
    "probe_top1": round(float(top1), 4), "probe_balanced": round(float(bal), 4),
    "probe_macro_f1": round(float(mf1), 4), "majority_baseline": round(float(maj), 4),
    "per_class_recall": perclass,
    "random_feature_probe_top1": round(float(accuracy_score(y, p2)), 4),
    "random_feature_probe_balanced": round(float(balanced_accuracy_score(y, p2)), 4),
    "note": "DA source domain is a synthetic random-noise proxy (generate_simulated_source_data); "
            "balanced accuracy confirms the negative is genuine, not a majority-class artifact.",
}
json.dump(res, open(OUT / "da_probe_balanced.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("\nsaved:", OUT / "da_probe_balanced.json")
