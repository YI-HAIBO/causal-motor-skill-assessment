# -*- coding: utf-8 -*-
"""Re-run the ST-GCN+IRM pilot on the 500 real skeleton sequences (70/30 split),
to obtain a reproducible, recorded held-out accuracy for the paper (item 13)."""
import json, os, sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

BASE = r"C:\Users\Yi\WorkBuddy\2026-11-09-19-01-01"
sys.path.insert(0, os.path.join(BASE, "."))

# Import model classes from train_stgcn_irm.py (module-level only, no main())
import importlib.util
spec = importlib.util.spec_from_file_location(
    "tsi", os.path.join(BASE, "train_stgcn_irm.py"))
tsi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tsi)

T, J, C = 16, 33, 4
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Load 500 skeletons
skel_path = os.path.join(BASE, "outputs", "stgcn_irm", "skeleton_sequences.json")
samples = json.load(open(skel_path, encoding="utf-8"))
print("Loaded", len(samples), "samples")

valid = [s for s in samples if s.get("category") in tsi.CATEGORY_MAP]
print("Valid:", len(valid))

class PoseDS(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        sk = np.array(s["skeleton"], dtype=np.float32)
        x = torch.from_numpy(sk).permute(2, 0, 1)
        label = tsi.CATEGORY_MAP[s["category"]]
        env = tsi.ENV_MAP.get(s["category"], 0)
        return x, label, env, s.get("id", idx)

ds = PoseDS(valid)
labels = np.array([ds[i][1] for i in range(len(ds))])
train_idx, test_idx = train_test_split(
    np.arange(len(ds)), test_size=0.3, random_state=SEED, stratify=labels)

train_ds = torch.utils.data.Subset(ds, train_idx.tolist())
test_ds = torch.utils.data.Subset(ds, test_idx.tolist())
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

print("Train:", len(train_ds), "Test:", len(test_ds))
test_labels = labels[test_idx]
maj = max(np.bincount(test_labels))
maj_acc = maj / len(test_labels)
print("Majority-class baseline (test): %.3f" % maj_acc)
per_class_test = {tsi.ALL_CATEGORIES[c]: int((test_labels == c).sum()) for c in range(len(tsi.ALL_CATEGORIES))}
print("Per-class test counts:", per_class_test)

adj = tsi.build_adjacency(J)
model = tsi.STGCN(in_channels=C, hidden=tsi.HIDDEN, num_joints=J,
                  num_classes=len(tsi.ALL_CATEGORIES), adj=adj)
nparam = sum(p.numel() for p in model.parameters())
print("Model params:", nparam)

trainer = tsi.IRMTrainer(model, adj, device="cpu")
trainer.train(train_loader, n_epochs=tsi.N_EPOCHS)

model.eval()
preds, gt = [], []
with torch.no_grad():
    for x, y, env, ids in test_loader:
        logits = model(x)
        preds.extend(logits.argmax(dim=1).tolist())
        gt.extend(y.tolist())
preds = np.array(preds); gt = np.array(gt)
acc = float((preds == gt).mean())
print("\nHELD-OUT TEST ACCURACY: %.4f" % acc)

report = classification_report(gt, preds, labels=list(range(len(tsi.ALL_CATEGORIES))),
                               target_names=tsi.ALL_CATEGORIES, output_dict=True, zero_division=0)
recall = {tsi.ALL_CATEGORIES[c]: report[tsi.ALL_CATEGORIES[c]]["recall"] for c in range(len(tsi.ALL_CATEGORIES))}
print("Per-class recall:", {k: round(v, 3) for k, v in recall.items()})

cm = confusion_matrix(gt, preds, labels=list(range(len(tsi.ALL_CATEGORIES))))

out = {
    "n_samples": len(valid),
    "n_train": int(len(train_ds)),
    "n_test": int(len(test_ds)),
    "seed": SEED,
    "test_accuracy": acc,
    "majority_baseline": float(maj_acc),
    "per_class_test": per_class_test,
    "per_class_recall": {k: float(v) for k, v in recall.items()},
    "confusion_matrix": cm.tolist(),
    "model_params": nparam,
    "hidden_dim": tsi.HIDDEN,
    "n_epochs": tsi.N_EPOCHS,
    "lambda_irm": tsi.LAMBDA_IRM,
}
outpath = os.path.join(BASE, "outputs", "stgcn_irm", "training_results_500.json")
json.dump(out, open(outpath, "w", encoding="utf-8"), indent=1)
print("\nSAVED", outpath)
print(json.dumps({k: v for k, v in out.items() if k not in ("confusion_matrix",)}, indent=1))
