# -*- coding: utf-8 -*-
"""
ST-GCN + IRM 完整模型训练（CPU 版）
- 从 47 个真实 GIF 提取骨架序列 (T=16, J=33, C=4)
- 构建轻量 ST-GCN（空间图卷积 + 时序卷积）
- 集成 IRM（Invariant Risk Minimization）
- 多环境划分（按 category 分环境，模拟 cross-domain 泛化）
- 输出：训练日志、模型权重、评估指标
"""
import json, os, sys, math, time
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------- 配置 ----------------
BASE = Path("C:/Users/Yi/WorkBuddy/2026-11-09-19-01-01")
GIF_DIR = BASE / "data/sample_gifs"
JSON_PATH = BASE / "data/exercises.json"
OUT_DIR = BASE / "outputs/stgcn_irm"
OUT_DIR.mkdir(parents=True, exist_ok=True)

T = 16          # 采样帧数
J = 33          # MediaPipe 关节数
C = 4           # (x, y, z, visibility)
IMG_SIZE = 224
HIDDEN = 64     # ST-GCN 隐藏通道
N_EPOCHS = 100
BATCH_SIZE = 8
LR = 1e-3
LAMBDA_IRM = 0.1   # IRM 惩罚权重
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------- 骨架邻接矩阵（MediaPipe 33 关节拓扑）----------------
# 基于 MediaPipe Pose Landmark 的骨骼连接定义
SKELETON_EDGES = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),
    (11,12),(12,14),(14,16),(16,18),(18,20),(20,22),(16,22),
    (11,13),(13,15),(15,17),(17,19),(19,21),(21,23),(15,23),
    (12,24),(24,26),(26,28),(28,30),(28,32),(24,27),(27,29),(29,31),
    (23,25),(25,27),
    (11,23),(12,24),
]

def build_adjacency(num_nodes=J):
    """构建归一化邻接矩阵（含自环）"""
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i, j in SKELETON_EDGES:
        A[i, j] = 1.0
        A[j, i] = 1.0
    # 自环
    A += np.eye(num_nodes, dtype=np.float32)
    # 行归一化（D^{-1/2} A D^{-1/2}）
    D = A.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D + 1e-8))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return torch.tensor(A_norm, dtype=torch.float32)

# ---------------- 数据集 ----------------
class PoseDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples  # list of {id, name, category, skeleton: (T,J,C)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        sk = np.array(s["skeleton"], dtype=np.float32)  # (T, J, C)
        # 转为 (C, T, J) → ST-GCN 输入格式
        x = torch.from_numpy(sk).permute(2, 0, 1)  # (C, T, J)
        # 类别标签
        label = self._category_label(s["category"])
        # 环境标签（按 category 分组作为环境）
        env = self._environment_label(s["category"])
        return x, label, env, s["id"]

    def _category_label(self, cat):
        return CATEGORY_MAP.get(cat, 0)

    def _environment_label(self, cat):
        """按 category 划分环境（每个 category 一个环境）"""
        return ENV_MAP.get(cat, 0)

# 类别 → 标签映射
ALL_CATEGORIES = [
    "upper arms", "upper legs", "back", "waist", "chest",
    "shoulders", "lower legs", "lower arms", "cardio", "neck"
]
CATEGORY_MAP = {c: i for i, c in enumerate(ALL_CATEGORIES)}
ENV_MAP = {c: i for i, c in enumerate(ALL_CATEGORIES)}  # 每 category 一个环境

# ---------------- ST-GCN 模型 ----------------
class STGCN(nn.Module):
    """轻量 ST-GCN：空间图卷积 + 时序卷积"""
    def __init__(self, in_channels=C, hidden=HIDDEN, num_joints=J,
                 num_classes=len(ALL_CATEGORIES), adj=None):
        super().__init__()
        self.adj = adj
        # 第一层：空间图卷积 + 时序卷积
        self.gc1 = GraphConv(in_channels, hidden, num_joints, adj)
        self.tcn1 = nn.Conv2d(hidden, hidden, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)

        self.gc2 = GraphConv(hidden, hidden, num_joints, adj)
        self.tcn2 = nn.Conv2d(hidden, hidden, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.bn2 = nn.BatchNorm2d(hidden)

        self.gc3 = GraphConv(hidden, hidden, num_joints, adj)
        self.tcn3 = nn.Conv2d(hidden, hidden, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.bn3 = nn.BatchNorm2d(hidden)

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        # x: (B, C, T, J)
        B, C_in, T, J = x.shape
        # 空间图卷积
        x = self.gc1(x)           # (B, hidden, T, J)
        x = self.tcn1(x)          # (B, hidden, T, J)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.gc2(x)
        x = self.tcn2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.gc3(x)
        x = self.tcn3(x)
        x = self.bn3(x)
        x = F.relu(x)

        # 全局平均池化
        x = x.mean(dim=(2, 3))    # (B, hidden)
        x = self.fc(x)            # (B, num_classes)
        return x


class GraphConv(nn.Module):
    """空间图卷积：X' = A_norm @ X @ W"""
    def __init__(self, in_c, out_c, num_joints, adj):
        super().__init__()
        self.num_joints = num_joints
        self.weight = nn.Parameter(torch.randn(num_joints, in_c, out_c) * 0.01)
        self.adj = adj  # (J, J) 固定邻接

    def forward(self, x):
        # x: (B, C, T, J)
        B, C_in, T, J = x.shape
        # (B, C, T, J) → (B, T, J, C) → (B*T, J, C)
        x = x.permute(0, 2, 3, 1).reshape(B * T, J, C_in)
        # 邻接传播：(B*T, J, C) @ (J, C, Out) = (B*T, J, Out)
        # 但 weight 是 (J, C, Out)，需要逐节点卷积
        # X' = sum_{k} A_{j,k} * X_k * W_j  →  (A_norm @ X) @ W
        # 先做邻接聚合: (B*T, J, C) → (B*T, J, C)
        adj = self.adj.to(x.device)
        x_agg = torch.bmm(adj.unsqueeze(0).expand(B * T, -1, -1), x)  # (B*T, J, C)
        # 逐节点线性变换
        # (B*T, J, C) @ (J, C, Out) → 需要 broadcast
        # weight: (J, C, Out)
        out = torch.einsum('b j c, j c o -> b j o', x_agg, self.weight)  # (B*T, J, Out)
        # 还原维度
        out = out.reshape(B, T, J, -1).permute(0, 3, 1, 2)  # (B, Out, T, J)
        return out


# ---------------- IRM 训练 ----------------
class IRMTrainer:
    def __init__(self, model, adj, device="cpu"):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        self.criterion = nn.CrossEntropyLoss()
        self.adj = adj

    def train(self, dataloader, n_epochs=N_EPOCHS):
        all_envs = sorted(set(sum(
            [batch[2].tolist() for batch in dataloader], [])))
        print(f"Environments: {len(all_envs)} ({all_envs})")
        print(f"Training on {self.device}")

        for epoch in range(n_epochs):
            self.model.train()
            total_loss = 0.0
            total_ce = 0.0
            total_irm = 0.0
            n_batches = 0

            for x, y, env, ids in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                env = env.to(self.device)
                B = x.size(0)

                self.optimizer.zero_grad()
                logits = self.model(x)  # (B, num_classes)
                ce_loss = self.criterion(logits, y)

                # IRM 梯度惩罚：对每个环境，计算梯度范数
                # 简化版：对每个样本计算梯度惩罚（用 logits 对输入的梯度）
                # 更标准的 IRM：对每个环境计算 R^epsilon 梯度
                # 这里用每个环境的 CE 梯度范数作为代理
                irm_loss = torch.tensor(0.0, device=self.device)
                for e in env.unique():
                    mask = env == e
                    if mask.sum() < 2:
                        continue
                    x_e = x[mask]
                    y_e = y[mask]
                    logits_e = self.model(x_e)
                    # 计算每个样本的梯度（需要 retain_graph）
                    # 简化：用 logits 对输入特征的 Jacobian 范数
                    x_e_req = x_e.detach().requires_grad_(True)
                    logits_e2 = self.model(x_e_req)
                    loss_e = self.criterion(logits_e2, y_e)
                    grad = torch.autograd.grad(loss_e, x_e_req, create_graph=True)[0]
                    irm_loss += grad.pow(2).mean()

                loss = ce_loss + LAMBDA_IRM * irm_loss / max(1, len(env.unique()))
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_ce += ce_loss.item()
                total_irm += irm_loss.item()
                n_batches += 1

            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}: loss={total_loss/n_batches:.4f} "
                      f"ce={total_ce/n_batches:.4f} "
                      f"irm={total_irm/n_batches:.4f}")

        return self.model

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_envs = []
        with torch.no_grad():
            for x, y, env, ids in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(y.cpu().tolist())
                all_envs.extend(env.cpu().tolist())
        acc = correct / total
        return acc, all_preds, all_labels, all_envs


# ---------------- 骨架提取 ----------------
def extract_skeleton_sequences():
    """从 GIF 提取骨架序列 (T, J, C)"""
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from PIL import Image, ImageSequence

    model_path = str(BASE / "data/models/pose_landmarker_lite.task")
    base_options = BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        output_segmentation_masks=False,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    with open(JSON_PATH, encoding="utf-8") as f:
        items = json.load(f)

    gifs_set = set(os.listdir(str(GIF_DIR)))
    samples = []
    n_ok = 0

    for it in items:
        gif_name = it.get("gif_url", "").split("/")[-1]
        if gif_name not in gifs_set:
            continue
        gif_path = str(GIF_DIR / gif_name)
        try:
            im = Image.open(gif_path)
            frames = [np.asarray(fr.convert("RGB")) for fr in ImageSequence.Iterator(im)]
            if not frames:
                continue
            # 采样 T 帧
            if len(frames) <= T:
                sampled = frames + [frames[-1]] * (T - len(frames))
            else:
                idx = np.linspace(0, len(frames) - 1, T).round().astype(int)
                sampled = [frames[i] for i in idx]
            # resize
            sampled = [np.asarray(Image.fromarray(f).resize((IMG_SIZE, IMG_SIZE))) for f in sampled]

            # 提取每帧关键点
            lms = []
            for f in sampled:
                with __import__("tempfile").NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    Image.fromarray(f).save(tmp.name)
                    tmp_path = tmp.name
                try:
                    mp_img = mp.Image.create_from_file(tmp_path)
                    result = landmarker.detect(mp_img)
                    if result.pose_landmarks:
                        pose = result.pose_landmarks[0]
                        lm = np.array([[l.x, l.y, l.z, l.visibility] for l in pose], dtype=np.float32)
                    else:
                        lm = np.zeros((J, C), dtype=np.float32)
                finally:
                    os.unlink(tmp_path)
                lms.append(lm)

            skeleton = np.stack(lms)  # (T, J, C)
            samples.append({
                "id": it.get("id"),
                "name": it.get("name", ""),
                "category": it.get("category", ""),
                "equipment": it.get("equipment", ""),
                "skeleton": skeleton.tolist(),
            })
            n_ok += 1
            if n_ok % 10 == 0:
                print(f"  Extracted {n_ok}/{len(gifs_set)}")
        except Exception as e:
            print(f"  FAIL {gif_name}: {e}")

    landmarker.close()
    print(f"Extraction done: {n_ok} samples")
    return samples


# ---------------- 主流程 ----------------
def main():
    print("=" * 60)
    print("ST-GCN + IRM Training (CPU)")
    print("=" * 60)

    # 1. 提取骨架序列
    skel_path = OUT_DIR / "skeleton_sequences.json"
    if skel_path.exists():
        print("Loading cached skeleton sequences...")
        with open(skel_path, encoding="utf-8") as f:
            samples = json.load(f)
    else:
        print("Extracting skeleton sequences from GIFs...")
        samples = extract_skeleton_sequences()
        with open(skel_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=1)
        print(f"Cached to {skel_path}")

    # 过滤有效样本
    valid_samples = [s for s in samples if s["category"] in CATEGORY_MAP]
    print(f"Valid samples: {len(valid_samples)} / {len(samples)}")

    # 类别分布
    from collections import Counter
    cat_counts = Counter(s["category"] for s in valid_samples)
    print("Category distribution:", dict(cat_counts))

    # 2. 构建数据集
    dataset = PoseDataset(valid_samples)
    # 分层划分：按 category 分层
    from sklearn.model_selection import train_test_split
    indices = np.arange(len(dataset))
    # 按类别分层
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    train_idx, test_idx = train_test_split(
        indices, test_size=0.3, random_state=SEED, stratify=labels
    )

    train_dataset = torch.utils.data.Subset(dataset, train_idx.tolist())
    test_dataset = torch.utils.data.Subset(dataset, test_idx.tolist())

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    # 3. 构建模型
    adj = build_adjacency(J)
    model = STGCN(in_channels=C, hidden=HIDDEN, num_joints=J,
                  num_classes=len(ALL_CATEGORIES), adj=adj)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # 4. 训练
    trainer = IRMTrainer(model, adj, device="cpu")
    trainer.train(train_loader, n_epochs=N_EPOCHS)

    # 5. 评估
    acc, preds, labels, envs = trainer.evaluate(test_loader)
    print(f"\nTest accuracy: {acc:.4f}")

    # 逐类准确率
    from sklearn.metrics import classification_report, confusion_matrix
    report = classification_report(labels, preds, target_names=ALL_CATEGORIES, output_dict=True)
    print("\nPer-class report:")
    for c in ALL_CATEGORIES:
        if c in report:
            print(f"  {c}: P={report[c]['precision']:.3f} R={report[c]['recall']:.3f} F1={report[c]['f1-score']:.3f}")

    cm = confusion_matrix(labels, preds, labels=list(range(len(ALL_CATEGORIES))))
    print("\nConfusion matrix:")
    print(cm)

    # 6. 保存模型
    torch.save(model.state_dict(), OUT_DIR / "stgcn_irm.pt")
    print(f"\nModel saved to {OUT_DIR / 'stgcn_irm.pt'}")

    # 7. 保存结果
    results = {
        "n_samples": len(valid_samples),
        "n_train": len(train_dataset),
        "n_test": len(test_dataset),
        "n_categories": len(ALL_CATEGORIES),
        "test_accuracy": float(acc),
        "per_class_report": {c: report[c] for c in ALL_CATEGORIES if c in report},
        "confusion_matrix": cm.tolist(),
        "model_params": sum(p.numel() for p in model.parameters()),
        "hidden_dim": HIDDEN,
        "n_epochs": N_EPOCHS,
        "lambda_irm": LAMBDA_IRM,
    }
    with open(OUT_DIR / "training_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"Results saved to {OUT_DIR / 'training_results.json'}")


if __name__ == "__main__":
    main()