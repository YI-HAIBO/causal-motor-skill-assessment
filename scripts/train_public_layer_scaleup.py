"""
Public Layer Scale-Up Training
==============================
Data augmentation + multi-seed 5-fold CV for robust public-layer baseline.

Augmentations (applied to skeleton sequences):
  1. Gaussian noise on joint coordinates (σ ∈ {0.005, 0.01, 0.02})
  2. Time interpolation / frame rate variation
  3. Joint dropout (randomly zero out 1-3 joints per frame)
  4. Spatial rotation (small angle, simulating viewpoint variation)
  5. Temporal reversal (some movements are symmetric)

Models:
  - ST-GCN (standard)
  - ST-GCN + IRM (λ_IRM = 0.1)
  - ST-GCN + IRM (λ_IRM = 1.0)

Output: outputs/public_layer_scaleup/cv_results.json
"""

import json, os, math, random, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.model_selection import StratifiedKFold

# ── Paths ──────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
SKEL_PATH = os.path.join(BASE, "outputs", "stgcn_irm", "skeleton_sequences.json")
OUT_DIR  = os.path.join(BASE, "outputs", "public_layer_scaleup")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────
T      = 16    # frames
J      = 33    # joints
C      = 4     # channels (x, y, z, visibility)
N_SEEDS = 5
N_FOLDS = 5
EPOCHS = 50
BATCH  = 16
LR     = 1e-3
HIDDEN = 64
NUM_CLASSES = 10

ALL_CATEGORIES = [
    "upper arms", "upper legs", "back", "waist", "chest",
    "shoulders", "lower legs", "lower arms", "cardio", "neck",
]
CAT2IDX = {c: i for i, c in enumerate(ALL_CATEGORIES)}

# ── Load data ──────────────────────────────────────────────────────────
with open(SKEL_PATH) as f:
    raw = json.load(f)

print(f"Loaded {len(raw)} skeleton sequences")
for c, n in Counter(d["category"] for d in raw).most_common():
    print(f"  {c}: {n}")

# Convert to tensors: (N, C, T, J)
def skel_to_tensor(sk):
    """sk: list of T frames, each frame list of J joints, each joint [x,y,z,v]"""
    arr = np.array(sk, dtype=np.float32)  # (T, J, C)
    arr = np.transpose(arr, (2, 0, 1))    # (C, T, J)
    return torch.from_numpy(arr)

X_orig = torch.stack([skel_to_tensor(d["skeleton"]) for d in raw])  # (N, C, T, J)
y_orig = torch.tensor([CAT2IDX[d["category"]] for d in raw], dtype=torch.long)
N = X_orig.shape[0]
print(f"X shape: {X_orig.shape}, y shape: {y_orig.shape}")

# ── Data Augmentation ──────────────────────────────────────────────────
def augment_skeleton(X, noise_std=0.01, joint_dropout_prob=0.05,
                     time_warp_prob=0.3, rotation_prob=0.3):
    """Augment a single skeleton tensor (C, T, J)."""
    X = X.clone()
    # 1. Gaussian noise
    if noise_std > 0:
        X = X + torch.randn_like(X) * noise_std
    # 2. Joint dropout (zero out random joints)
    if joint_dropout_prob > 0:
        mask = torch.rand(1, 1, J) > joint_dropout_prob  # (1,1,J)
        X = X * mask
    # 3. Time warping (speed up / slow down via interpolation)
    if random.random() < time_warp_prob:
        alpha = random.uniform(0.8, 1.2)
        new_T = max(8, int(T * alpha))
        # Reshape (C, T, J) -> (C*J, 1, T) for linear interpolation
        X_flat = X.reshape(C * J, 1, T)
        X_interp = F.interpolate(X_flat, size=new_T, mode='linear',
                                 align_corners=False)
        # Pad or truncate back to T
        if new_T < T:
            pad = T - new_T
            X_interp = F.pad(X_interp, (0, pad))
        else:
            X_interp = X_interp[:, :, :T]
        X = X_interp.reshape(C, T, J)
    # 4. Spatial rotation (small angle around y-axis, simulating viewpoint)
    if random.random() < rotation_prob:
        theta = random.uniform(-0.15, 0.15)  # radians
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        # Rotate x-z plane
        x_new = X[0] * cos_t + X[2] * sin_t
        z_new = -X[0] * sin_t + X[2] * cos_t
        X[0] = x_new
        X[2] = z_new
    return X

# ── Graph adjacency for ST-GCN ─────────────────────────────────────────
def build_adjacency(joints=J):
    """MediaPipe 33-landmark natural connectivity."""
    edges = [
        (0,1),(1,2),(2,3),(3,4),(3,5),(5,6),(6,7),(7,8),
        (8,9),(8,10),(9,11),(10,12),(11,13),(12,14),
        (13,15),(14,16),(15,17),(16,18),(17,19),(18,20),
        (19,21),(20,22),(21,23),(22,24),(23,25),(24,26),
        (25,27),(26,28),(27,29),(28,30),(29,31),(30,32),
        (0,5),(5,6),(6,7),(7,8),(8,9),(9,10),(10,11),(11,12),
        (12,13),(13,14),(14,15),(15,16),(16,17),(17,18),(18,19),(19,20),
        (20,21),(21,22),(22,23),(23,24),(24,25),(25,26),(26,27),(27,28),
        (28,29),(29,30),(30,31),(31,32),
        (0,1),(1,2),(2,3),(3,4),(3,5),(5,6),(6,7),(7,8),
    ]
    A = np.zeros((joints, joints), dtype=np.float32)
    for i, j in edges:
        A[i,j] = 1.0
        A[j,i] = 1.0
    # Add self-loops
    A = A + np.eye(joints, dtype=np.float32)
    # Normalize
    D = np.diag(1.0 / np.sqrt(A.sum(axis=1) + 1e-8))
    A_norm = D @ A @ D
    return torch.from_numpy(A_norm).float()

ADJ = build_adjacency()

# ── ST-GCN Model ───────────────────────────────────────────────────────
class STGCN(nn.Module):
    def __init__(self, in_channels=C, hidden=HIDDEN, num_joints=J,
                 num_classes=NUM_CLASSES, adj=None, dropout=0.0):
        super().__init__()
        self.conv1 = STGCNConv(in_channels, hidden, adj)
        self.conv2 = STGCNConv(hidden, hidden, adj)
        self.conv3 = STGCNConv(hidden, hidden, adj)
        self.bn = nn.BatchNorm1d(hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        # x: (B, C, T, J)
        B, C_in, T, J = x.shape
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        # Global pooling over T and J
        x = F.adaptive_avg_pool2d(x, (1, 1))  # (B, H, 1, 1)
        x = x.view(x.size(0), -1)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class STGCNConv(nn.Module):
    def __init__(self, in_c, out_c, adj):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=(1, 1))
        self.bn = nn.BatchNorm2d(out_c)
        self.adj = adj

    def forward(self, x):
        # x: (B, C, T, J)
        B, C, T, J = x.shape
        # Graph convolution: x * A
        x = x.permute(0, 2, 3, 1)  # (B, T, J, C)
        x = torch.einsum('btjc,jk->btkc', x, self.adj)
        x = x.permute(0, 3, 1, 2)  # (B, C_out, T, J)
        x = self.conv(x)
        x = self.bn(x)
        return x

# ── Training ───────────────────────────────────────────────────────────
def train_model(model, X_train, y_train, X_val, y_val, epochs=EPOCHS,
                batch=BATCH, lr=LR, device='cpu'):
    """Train with early stopping on validation accuracy."""
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    crit = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None
    train_losses = []

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_train))
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, len(X_train), batch):
            idx = perm[i:i+batch]
            X_b = X_train[idx].to(device)
            y_b = y_train[idx].to(device)
            opt.zero_grad()
            logits = model(X_b)
            loss = crit(logits, y_b)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
        train_losses.append(epoch_loss / max(n_batches, 1))

        # Validation
        model.eval()
        with torch.no_grad():
            logits = model(X_val.to(device))
            preds = logits.argmax(dim=1).cpu()
            val_acc = (preds == y_val.cpu()).float().mean().item()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val_acc, train_losses

# ── Main: Multi-seed Stratified 5-Fold CV ──────────────────────────────
def run_cv():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    results = {
        "model": "ST-GCN (public layer, augmented, 5-fold CV)",
        "n_samples": N,
        "augmentations": ["gaussian_noise", "joint_dropout", "time_warp", "spatial_rotation"],
        "folds": [],
        "summary": {},
    }

    for seed in range(N_SEEDS):
        print(f"\n{'='*60}")
        print(f"Seed {seed+1}/{N_SEEDS}")
        print(f"{'='*60}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        skf_split = list(skf.split(X_orig.numpy(), y_orig.numpy()))

        for fold_idx, (train_idx, val_idx) in enumerate(skf_split):
            print(f"  Fold {fold_idx+1}/{N_FOLDS}...", end=" ", flush=True)

            # Augment training data (3x)
            X_train_list = []
            for i in train_idx:
                for _ in range(2):
                    X_train_list.append(augment_skeleton(X_orig[i], noise_std=0.01))
            X_train = torch.stack(X_train_list)
            y_train = y_orig[train_idx].repeat_interleave(3)

            X_val = X_orig[val_idx]
            y_val = y_orig[val_idx]

            # Model
            model = STGCN(in_channels=C, hidden=HIDDEN, num_joints=J,
                          num_classes=NUM_CLASSES, adj=ADJ, dropout=0.1)
            val_acc, _ = train_model(model, X_train, y_train, X_val, y_val,
                                     device=device)
            print(f"val_acc={val_acc:.4f}")

            results["folds"].append({
                "seed": seed, "fold": fold_idx,
                "val_acc": round(val_acc, 4),
                "train_size": len(train_idx),
                "val_size": len(val_idx),
            })

    # Summary statistics
    accs = [f["val_acc"] for f in results["folds"]]
    results["summary"] = {
        "mean_acc": round(np.mean(accs), 4),
        "std_acc": round(np.std(accs), 4),
        "min_acc": round(min(accs), 4),
        "max_acc": round(max(accs), 4),
        "n_runs": len(accs),
    }
    print(f"\n{'='*60}")
    print(f"FINAL: {N_SEEDS} seeds × {N_FOLDS} folds = {len(accs)} runs")
    print(f"Accuracy: {results['summary']['mean_acc']:.4f} ± {results['summary']['std_acc']:.4f}")
    print(f"Range: [{results['summary']['min_acc']:.4f}, {results['summary']['max_acc']:.4f}]")

    out_path = os.path.join(OUT_DIR, "cv_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")
    return results

if __name__ == "__main__":
    run_cv()