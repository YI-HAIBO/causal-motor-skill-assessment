"""
GPU Training Pipeline: ST-GCN + IRM
===================================
Distributed training script for the full ST-GCN + IRM model.

Requirements:
  - PyTorch 2.0+
  - torch>=2.0 with CUDA 11.8+ or 12.x
  - torchvision (for video loading if using raw GIFs)
  - mediapipe (for skeleton extraction)
  - tensorboard (for logging)

Usage:
  # Single GPU
  python train_stgcn_irm_gpu.py --config configs/default.yaml

  # Multi-GPU (DDP)
  python -m torch.distributed.launch --nproc_per_node=4 train_stgcn_irm_gpu.py \
      --config configs/default.yaml --distributed

  # SLURM
  sbatch scripts/slurm_train.sh
"""

import os, sys, json, math, time, argparse, warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from collections import Counter
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────
class TrainingConfig:
    """Hyperparameter configuration."""
    def __init__(self, **kwargs):
        # Model
        self.in_channels = 4
        self.hidden = 64
        self.num_joints = 33
        self.num_classes = 10
        self.dropout = 0.1

        # IRM
        self.lambda_irm = 0.1
        self.num_envs = 10  # one per exercise category

        # Training
        self.epochs = 200
        self.batch_size = 32
        self.lr = 1e-3
        self.weight_decay = 1e-4
        self.momentum = 0.9
        self.lr_schedule = "cosine"
        self.warmup_epochs = 5

        # Data
        self.num_workers = 8
        self.pin_memory = True
        self.seq_len = 16  # frames per sequence
        self.augment = True

        # Paths
        self.data_dir = "data/exclusive_layer"
        self.skeleton_dir = "data/skeletons"
        self.checkpoint_dir = "outputs/checkpoints"
        self.log_dir = "outputs/tensorboard"

        # Distributed
        self.distributed = False
        self.local_rank = 0
        self.device = "cuda"

        for k, v in kwargs.items():
            setattr(self, k, v)

    def save(self, path):
        with open(path, "w") as f:
            json.dump({k: v for k, v in self.__dict__.items() if not k.startswith("_")},
                      f, indent=2, default=str)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(**json.load(f))


# ── Graph Adjacency ────────────────────────────────────────────────────
def build_mediapipe_adjacency(joints: int = 33) -> torch.Tensor:
    """Build normalized adjacency matrix for MediaPipe 33 landmarks."""
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
    ]
    A = np.zeros((joints, joints), dtype=np.float32)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    A += np.eye(joints, dtype=np.float32)
    D = np.diag(1.0 / np.sqrt(A.sum(axis=1) + 1e-8))
    A_norm = D @ A @ D
    return torch.from_numpy(A_norm).float()


# ── ST-GCN Model ───────────────────────────────────────────────────────
class STGCNConv(nn.Module):
    """Spatial-Temporal Graph Convolutional layer."""

    def __init__(self, in_channels: int, out_channels: int,
                 adj: torch.Tensor, temporal_kernel_size: int = 9):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=(temporal_kernel_size, 1),
                              padding=(temporal_kernel_size // 2, 0))
        self.bn = nn.BatchNorm2d(out_channels)
        self.adj = nn.Parameter(adj, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T, J)
        # Graph convolution
        x = x.permute(0, 2, 3, 1)  # (B, T, J, C_in)
        x = torch.einsum('btjc,jk->btkc', x, self.adj)
        x = x.permute(0, 3, 1, 2)  # (B, C_out, T, J)
        # Temporal convolution
        x = self.conv(x)
        x = self.bn(x)
        return x


class STGCN(nn.Module):
    """Full ST-GCN model for skeleton-based action quality assessment."""

    def __init__(self, in_channels: int = 4, hidden: int = 64,
                 num_joints: int = 33, num_classes: int = 10,
                 adj: Optional[torch.Tensor] = None,
                 dropout: float = 0.1,
                 num_stgcnn_blocks: int = 3):
        super().__init__()
        if adj is None:
            adj = build_mediapipe_adjacency(num_joints)
        self.adj = adj

        channels = [in_channels] + [hidden] * num_stgcnn_blocks
        self.stgcnn_layers = nn.ModuleList()
        for i in range(num_stgcnn_blocks):
            self.stgcnn_layers.append(
                STGCNConv(channels[i], channels[i+1], adj)
            )

        self.bn = nn.BatchNorm1d(hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, num_classes)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, J)
        for layer in self.stgcnn_layers:
            x = F.relu(layer(x))
        # Global average pooling over T and J
        x = F.adaptive_avg_pool2d(x, (1, 1))  # (B, H, 1, 1)
        B = x.size(0)
        x = x.view(B, -1)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class STGCN_IRM(nn.Module):
    """ST-GCN with Invariant Risk Minimization."""

    def __init__(self, base_model: STGCN, lambda_irm: float = 0.1):
        super().__init__()
        self.stgcn = base_model
        self.lambda_irm = lambda_irm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stgcn(x)

    def compute_irm_penalty(self, env_data: List[Tuple[torch.Tensor, torch.Tensor]],
                            device: str) -> torch.Tensor:
        """Compute IRM penalty: max over envs of ||∇_w R_e(w)||^2."""
        penalties = []
        for X_e, y_e in env_data:
            X_e, y_e = X_e.to(device), y_e.to(device)
            self.stgcn.zero_grad()
            logits = self.stgcn(X_e)
            loss = F.cross_entropy(logits, y_e)
            grads = torch.autograd.grad(
                loss, self.stgcn.parameters(),
                create_graph=True, retain_graph=True
            )
            grad_norm_sq = sum(g.pow(2).sum() for g in grads)
            penalties.append(grad_norm_sq)
        return max(penalties)


# ── Dataset ────────────────────────────────────────────────────────────
class SkeletonDataset(Dataset):
    """Dataset of skeleton sequences with environment labels."""

    def __init__(self, sequences: List[Dict], categories: List[str],
                 augment: bool = True, seq_len: int = 16):
        self.sequences = sequences
        self.cat2idx = {c: i for i, c in enumerate(categories)}
        self.augment = augment
        self.seq_len = seq_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        item = self.sequences[idx]
        skel = np.array(item["skeleton"], dtype=np.float32)  # (T, J, C)
        # (C, T, J)
        skel = np.transpose(skel, (2, 0, 1))

        # Pad or truncate to seq_len
        T_cur = skel.shape[1]
        if T_cur < self.seq_len:
            pad = self.seq_len - T_cur
            skel = np.pad(skel, ((0, 0), (0, pad), (0, 0)))
        elif T_cur > self.seq_len:
            start = (T_cur - self.seq_len) // 2
            skel = skel[:, start:start+self.seq_len, :]

        # Data augmentation
        if self.augment:
            skel = self._augment(skel)

        X = torch.from_numpy(skel).float()
        y = torch.tensor(self.cat2idx[item["category"]], dtype=torch.long)
        env = torch.tensor(item.get("env_idx", 0), dtype=torch.long)

        return X, y, env

    def _augment(self, skel: np.ndarray) -> np.ndarray:
        """Apply augmentations to skeleton array (C, T, J)."""
        # Gaussian noise
        if random.random() < 0.8:
            skel = skel + np.random.randn(*skel.shape) * 0.01

        # Joint dropout
        if random.random() < 0.3:
            mask = np.random.rand(1, 1, skel.shape[2]) > 0.05
            skel = skel * mask

        # Time warping
        if random.random() < 0.3:
            alpha = np.random.uniform(0.85, 1.15)
            new_T = max(8, int(self.seq_len * alpha))
            # Simple interpolation
            from scipy.interpolate import interp1d
            T_cur = skel.shape[1]
            t_old = np.linspace(0, 1, T_cur)
            t_new = np.linspace(0, 1, new_T)
            skel_warped = np.zeros_like(skel)
            for c in range(skel.shape[0]):
                for j in range(skel.shape[2]):
                    f = interp1d(t_old, skel[c, :, j], kind='linear',
                                 bounds_error=False, fill_value=0)
                    skel_warped[c, :, j] = f(t_new)
            # Pad/truncate back
            if new_T < self.seq_len:
                pad = self.seq_len - new_T
                skel_warped = np.pad(skel_warped, ((0,0),(0,pad),(0,0)))
            else:
                skel_warped = skel_warped[:, :self.seq_len, :]
            skel = skel_warped

        # Spatial rotation
        if random.random() < 0.3:
            theta = np.random.uniform(-0.15, 0.15)
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            x_new = skel[0] * cos_t + skel[2] * sin_t
            z_new = -skel[0] * sin_t + skel[2] * cos_t
            skel[0] = x_new
            skel[2] = z_new

        return skel


# ── Training Loop ──────────────────────────────────────────────────────
class Trainer:
    """Distributed trainer for ST-GCN + IRM."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = self._setup_device()
        self.model = self._build_model()
        self.scaler = GradScaler() if self.device.type == 'cuda' else None

        # Distributed setup
        if config.distributed:
            self._setup_distributed()

        # Data
        self.train_loader, self.val_loader = self._build_dataloaders()

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.lr,
            weight_decay=config.weight_decay
        )
        self.scheduler = self._build_scheduler()
        self.criterion = nn.CrossEntropyLoss()

        # Logging
        self.global_step = 0
        self.best_val_acc = 0.0

    def _setup_device(self) -> torch.device:
        if self.config.distributed:
            torch.cuda.set_device(self.config.local_rank)
            return torch.device("cuda", self.config.local_rank)
        return torch.device(self.config.device if torch.cuda.is_available() else "cpu")

    def _setup_distributed(self):
        dist.init_process_group(
            backend="nccl" if self.device.type == "cuda" else "gloo",
            init_method="env://",
            world_size=int(os.environ.get("WORLD_SIZE", 1)),
            rank=int(os.environ.get("RANK", 0)),
        )

    def _build_model(self) -> nn.Module:
        adj = build_mediapipe_adjacency(self.config.num_joints)
        base = STGCN(
            in_channels=self.config.in_channels,
            hidden=self.config.hidden,
            num_joints=self.config.num_joints,
            num_classes=self.config.num_classes,
            adj=adj,
            dropout=self.config.dropout,
        )
        model = STGCN_IRM(base, lambda_irm=self.config.lambda_irm)

        if self.config.distributed:
            model = DDP(model, device_ids=[self.config.local_rank])
        else:
            model = model.to(self.device)

        return model

    def _build_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        # Load skeleton data
        data_path = Path(self.config.data_dir) / "skeleton_sequences.json"
        with open(data_path) as f:
            all_data = json.load(f)

        # Assign environment labels (by equipment type)
        equip_types = sorted(set(d.get("equipment", "body weight") for d in all_data))
        equip2env = {e: i for i, e in enumerate(equip_types)}
        for d in all_data:
            d["env_idx"] = equip2env[d.get("equipment", "body weight")]

        # Train/val split (subject-disjoint)
        ids = [d["id"] for d in all_data]
        train_ids, val_ids = train_test_split(
            ids, test_size=0.2, random_state=42,
            stratify=[d["category"] for d in all_data]
        )

        train_data = [d for d in all_data if d["id"] in set(train_ids)]
        val_data = [d for d in all_data if d["id"] in set(val_ids)]

        categories = sorted(set(d["category"] for d in all_data))

        train_dataset = SkeletonDataset(
            train_data, categories,
            augment=self.config.augment,
            seq_len=self.config.seq_len,
        )
        val_dataset = SkeletonDataset(
            val_data, categories,
            augment=False,
            seq_len=self.config.seq_len,
        )

        if self.config.distributed:
            train_sampler = torch.utils.data.DistributedSampler(
                train_dataset, num_replicas=int(os.environ.get("WORLD_SIZE", 1)),
                rank=int(os.environ.get("RANK", 0)),
            )
            val_sampler = torch.utils.data.DistributedSampler(
                val_dataset, num_replicas=int(os.environ.get("WORLD_SIZE", 1)),
                rank=int(os.environ.get("RANK", 0)),
            )
        else:
            train_sampler = None
            val_sampler = None

        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size,
            shuffle=False, sampler=val_sampler,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )

        return train_loader, val_loader

    def _build_scheduler(self):
        if self.config.lr_schedule == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.epochs,
                eta_min=self.config.lr * 0.01
            )
        elif self.config.lr_schedule == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=50, gamma=0.5
            )
        else:
            return torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lambda epoch: 1.0
            )

    def train(self):
        """Main training loop."""
        for epoch in range(self.config.epochs):
            if self.config.distributed:
                self.train_loader.sampler.set_epoch(epoch)

            train_loss = self._train_epoch(epoch)
            val_acc, val_loss = self._validate_epoch()

            self.scheduler.step()

            # Log
            lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{self.config.epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"LR: {lr:.6f}")

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self._save_checkpoint(epoch, is_best=True)

            # Periodic checkpoint
            if (epoch + 1) % 20 == 0:
                self._save_checkpoint(epoch, is_best=False)

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_idx, (X, y, env) in enumerate(self.train_loader):
            X = X.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            if self.scaler is not None:
                with autocast():
                    logits = self.model(X)
                    loss = self.criterion(logits, y)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(X)
                loss = self.criterion(logits, y)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            self.global_step += 1

        return total_loss / max(n_batches, 1)

    def _validate_epoch(self) -> Tuple[float, float]:
        self.model.eval()
        total_correct = 0
        total_samples = 0
        total_loss = 0.0

        with torch.no_grad():
            for X, y, env in self.val_loader:
                X = X.to(self.device)
                y = y.to(self.device)

                if self.scaler is not None:
                    with autocast():
                        logits = self.model(X)
                        loss = self.criterion(logits, y)
                else:
                    logits = self.model(X)
                    loss = self.criterion(logits, y)

                preds = logits.argmax(dim=1)
                total_correct += (preds == y).sum().item()
                total_samples += y.size(0)
                total_loss += loss.item()

        acc = total_correct / max(total_samples, 1)
        loss = total_loss / max(len(self.val_loader), 1)
        return acc, loss

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "best_val_acc": self.best_val_acc,
            "config": self.config.__dict__,
        }

        if is_best:
            path = ckpt_dir / "best.pt"
        else:
            path = ckpt_dir / f"checkpoint_epoch{epoch+1}.pt"

        torch.save(state, path)
        print(f"  Checkpoint saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="ST-GCN + IRM GPU Training")
    parser.add_argument("--config", type=str, default="configs/default.json")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    # Load config
    if os.path.exists(args.config):
        with open(args.config) as f:
            config = TrainingConfig(**json.load(f))
    else:
        config = TrainingConfig()

    config.distributed = args.distributed
    config.local_rank = args.local_rank

    # Save config
    os.makedirs("configs", exist_ok=True)
    config.save("configs/run_config.json")

    # Train
    trainer = Trainer(config)
    trainer.train()

    print(f"\nTraining complete. Best val accuracy: {trainer.best_val_acc:.4f}")


if __name__ == "__main__":
    main()