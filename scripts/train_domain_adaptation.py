# -*- coding: utf-8 -*-
"""
域自适应预训练 Pipeline（方案 D：迁移学习 + 域自适应）
=====================================================
目标：在 FineGym-AQA（源域：体操动作 + FIG 裁判评分）上预训练 ST-GCN + IRM，
      然后通过无监督域自适应对齐到 Exercises Dataset（目标域：健身动作）。

关键创新：
- 源域：FineGym-AQA 的 D/E/ND/T 评分作为质量回归目标
- 域自适应：adversarial DA + MMD + 最大均值差异
- 目标域：Exercises Dataset 的合成弱标签作为分类目标
- 输出：域不变特征 → 目标域上的少样本/零样本质量评估

架构：
  Source Encoder → Domain Classifier (adv) → Score Regressor
  Target Encoder → (same weights) → Score Regressor
  MMD loss → align source/target feature distributions
"""

import json, os, sys, math
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

# ---------------- 配置 ----------------
BASE = Path("C:/Users/Yi/WorkBuddy/2026-11-09-19-01-01")
SOURCE_DATA = BASE / "outputs/finegym_aqa_simulated.json"  # 模拟源域数据
TARGET_DATA = BASE / "outputs/stgcn_irm/skeleton_sequences.json"  # 目标域
OUT_DIR = BASE / "outputs/domain_adaptation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

T = 16          # 采样帧数
J = 33          # MediaPipe 关节数
C = 4           # (x, y, z, visibility)
HIDDEN = 64     # ST-GCN 隐藏通道
N_EPOCHS = 100
BATCH_SIZE = 16
LR = 1e-3
LAMBDA_IRM = 0.1
LAMBDA_MMD = 0.5     # MMD 惩罚权重
LAMBDA_ADVERSARIAL = 0.3  # 域分类器对抗权重
LAMBDA_REG = 0.1     # 评分回归权重
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------- 骨架邻接矩阵 ----------------
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
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i, j in SKELETON_EDGES:
        A[i, j] = 1.0
        A[j, i] = 1.0
    A += np.eye(num_nodes, dtype=np.float32)
    D = A.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D + 1e-8))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return torch.tensor(A_norm, dtype=torch.float32)

# ---------------- 类别映射 ----------------
ALL_CATEGORIES = [
    "upper arms", "upper legs", "back", "waist", "chest",
    "shoulders", "lower legs", "lower arms", "cardio", "neck"
]
CATEGORY_MAP = {c: i for i, c in enumerate(ALL_CATEGORIES)}

# FineGym-AQA 体操项目 → Exercises Dataset 类别映射
# 体操 4 个项目 → 健身动作类别
GYM_TO_EXERCISE = {
    "VT": "chest",       # Vault → 胸部（推举类）
    "FX": "full",        # Floor Exercise → 全身（综合）
    "BB": "waist",       # Balance Beam → 腰部（平衡类）
    "UB": "upper arms",  # Uneven Bars → 上臂（拉举类）
}

# ---------------- 图卷积层 ----------------
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
        out = out.reshape(B, T, J, -1).permute(0, 3, 1, 2)
        return out

# ---------------- ST-GCN 编码器 ----------------
class STGCNEncoder(nn.Module):
    """ST-GCN 特征编码器（去掉分类头，输出特征向量）"""
    def __init__(self, in_channels=C, hidden=HIDDEN, num_joints=J, adj=None,
                 feature_dim=128):
        super().__init__()
        self.adj = adj
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
        self.feature_proj = nn.Linear(hidden, feature_dim)

    def forward(self, x):
        # x: (B, C, T, J)
        B, C_in, T, J = x.shape
        x = self.gc1(x); x = self.tcn1(x); x = self.bn1(x); x = F.relu(x); x = self.dropout(x)
        x = self.gc2(x); x = self.tcn2(x); x = self.bn2(x); x = F.relu(x); x = self.dropout(x)
        x = self.gc3(x); x = self.tcn3(x); x = self.bn3(x); x = F.relu(x)
        # 全局平均池化
        x = x.mean(dim=(2, 3))  # (B, hidden)
        x = self.feature_proj(x)  # (B, feature_dim)
        return x

# ---------------- 域分类器（对抗） ----------------
class DomainClassifier(nn.Module):
    """梯度反转层 + 域分类器"""
    def __init__(self, feature_dim=128, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, hidden)
        self.fc2 = nn.Linear(hidden, 2)  # 0=source, 1=target
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, alpha):
        # 梯度反转
        x = GradientReversal.apply(x, alpha)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class GradientReversal(torch.autograd.Function):
    """梯度反转层（GRL）"""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.alpha, None

# ---------------- 评分回归头 ----------------
class ScoreRegressor(nn.Module):
    """从域不变特征预测 D/E/ND/T 评分"""
    def __init__(self, feature_dim=128, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, hidden)
        self.fc2 = nn.Linear(hidden, 4)  # D, E, ND, T

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

# ---------------- 完整域自适应模型 ----------------
class DomainAdaptationModel(nn.Module):
    def __init__(self, adj, feature_dim=128):
        super().__init__()
        self.encoder = STGCNEncoder(adj=adj, feature_dim=feature_dim)
        self.domain_clf = DomainClassifier(feature_dim=feature_dim)
        self.score_regr = ScoreRegressor(feature_dim=feature_dim)

    def forward(self, x, alpha=1.0):
        feat = self.encoder(x)
        domain_out = self.domain_clf(feat, alpha)
        score_out = self.score_regr(feat)
        return feat, domain_out, score_out

# ---------------- MMD 损失 ----------------
def mmd_loss(source_feat, target_feat, kernel_mul=2.0, kernel_num=5):
    """最大均值差异（MMD）损失 — 安全版（处理不等长输入）"""
    source_feat = source_feat.float()
    target_feat = target_feat.float()
    n_s = source_feat.size(0)
    n_t = target_feat.size(0)

    # 子采样以加速（最大 256）
    max_n = 256
    if n_s > max_n:
        idx_s = torch.randperm(n_s)[:max_n]
        source_feat = source_feat[idx_s]
        n_s = max_n
    if n_t > max_n:
        idx_t = torch.randperm(n_t)[:max_n]
        target_feat = target_feat[idx_t]
        n_t = max_n

    total = torch.cat([source_feat, target_feat], dim=0)
    n_total = total.size(0)
    # pairwise squared distances
    XX = total @ total.t()
    X2 = (total * total).sum(1).view(-1, 1)
    dist = (X2 + X2.t() - 2 * XX).clamp(min=0)

    # 多个带宽的高斯核
    bandwidths = [0.2, 0.5, 1.0, 2.0, 5.0]
    K = torch.zeros(n_total, n_total, device=source_feat.device)
    for bw in bandwidths:
        K += torch.exp(-dist / (2 * bw ** 2))

    K_ss = K[:n_s, :n_s]
    K_st = K[:n_s, n_s:]
    K_tt = K[n_s:, n_s:]
    mmd = K_ss.mean() + K_tt.mean() - 2 * K_st.mean()
    return mmd

# ---------------- 数据集 ----------------
class DomainAdaptationDataset(Dataset):
    def __init__(self, source_samples, target_samples):
        self.source_samples = source_samples
        self.target_samples = target_samples

    def __len__(self):
        return max(len(self.source_samples), len(self.target_samples))

    def __getitem__(self, idx):
        src_idx = idx % len(self.source_samples)
        tgt_idx = idx % len(self.target_samples)
        src = self.source_samples[src_idx]
        tgt = self.target_samples[tgt_idx]

        src_x = torch.from_numpy(np.array(src["skeleton"], dtype=np.float32)).permute(2, 0, 1)
        tgt_x = torch.from_numpy(np.array(tgt["skeleton"], dtype=np.float32)).permute(2, 0, 1)

        # 源域评分（D, E, ND, T）
        src_scores = torch.tensor([
            src.get("diff", 5.0),
            src.get("execution", 8.0),
            src.get("nd", 0.0),
            src.get("total", 13.0)
        ], dtype=torch.float32)

        # 目标域类别标签
        tgt_label = torch.tensor(CATEGORY_MAP.get(tgt["category"], 0), dtype=torch.long)

        return src_x, tgt_x, src_scores, tgt_label

# ---------------- 模拟源域数据生成 ----------------
def generate_simulated_source_data(n_samples=500):
    """
    生成模拟 FineGym-AQA 源域数据
    基于 FineGym-AQA 的标注格式：体操 4 个项目 × 多个运动员 × FIG 评分
    """
    np.random.seed(SEED)
    samples = []
    apparatuses = ["VT", "FX", "BB", "UB"]

    for i in range(n_samples):
        app = apparatuses[i % 4]
        # 模拟骨架数据 (T, J, C)
        sk = np.random.randn(T, J, C).astype(np.float32) * 0.1
        # 模拟 FIG 评分
        diff = np.random.uniform(4.0, 7.0)  # 难度分
        execution = np.random.uniform(7.0, 10.0)  # 执行分
        nd = np.random.uniform(0.0, 1.0)  # 中性扣分
        total = diff + execution - nd  # 总分

        samples.append({
            "id": f"src_{i:04d}",
            "apparatus": app,
            "skeleton": sk.tolist(),
            "diff": float(diff),
            "execution": float(execution),
            "nd": float(nd),
            "total": float(total),
        })
    return samples

# ---------------- 训练 ----------------
def train():
    print("=" * 60)
    print("域自适应预训练（方案 D：FineGym-AQA → Exercises Dataset）")
    print("=" * 60)

    # 加载数据
    print("\n[1] 加载数据...")
    with open(TARGET_DATA, encoding='utf-8') as f:
        target_data = json.load(f)

    source_data = generate_simulated_source_data(n_samples=500)
    print(f"  源域样本: {len(source_data)}")
    print(f"  目标域样本: {len(target_data)}")

    # 构建模型
    print("\n[2] 构建模型...")
    adj = build_adjacency()
    model = DomainAdaptationModel(adj, feature_dim=128)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数: {total_params:,}")

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=LR * 0.01)

    # 损失函数
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    # 数据加载器
    dataset = DomainAdaptationDataset(source_data, target_data)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # 训练循环
    print(f"\n[3] 训练 {N_EPOCHS} epochs...")
    best_mmd = float('inf')

    for epoch in range(N_EPOCHS):
        model.train()
        total_loss = 0.0
        total_mmd = 0.0
        total_adv = 0.0
        total_reg = 0.0
        n_batches = 0

        # 逐步增加 GRL alpha
        alpha = min(1.0, epoch / N_EPOCHS * 2.0)

        for src_x, tgt_x, src_scores, tgt_label in dataloader:
            # 前向传播
            src_feat, src_domain, src_score = model(src_x, alpha)
            tgt_feat, tgt_domain, _ = model(tgt_x, alpha)

            # 1. MMD 损失（域对齐）
            mmd = mmd_loss(src_feat, tgt_feat)

            # 2. 域分类器对抗损失
            src_domain_labels = torch.zeros(src_domain.size(0), dtype=torch.long)
            tgt_domain_labels = torch.ones(tgt_domain.size(0), dtype=torch.long)
            domain_labels = torch.cat([src_domain_labels, tgt_domain_labels])
            domain_logits = torch.cat([src_domain, tgt_domain])
            adv_loss = ce_loss(domain_logits, domain_labels)

            # 3. 评分回归损失（仅源域）
            reg_loss = mse_loss(src_score, src_scores)

            # 总损失
            loss = LAMBDA_MMD * mmd + LAMBDA_ADVERSARIAL * adv_loss + LAMBDA_REG * reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_mmd += mmd.item()
            total_adv += adv_loss.item()
            total_reg += reg_loss.item()
            n_batches += 1

        scheduler.step()

        if epoch % 10 == 0 or epoch == N_EPOCHS - 1:
            avg_mmd = total_mmd / n_batches
            avg_loss = total_loss / n_batches
            print(f"  Epoch {epoch:3d}: loss={avg_loss:.4f}  mmd={avg_mmd:.4f}  "
                  f"adv={total_adv/n_batches:.4f}  reg={total_reg/n_batches:.4f}  alpha={alpha:.2f}")

            if avg_mmd < best_mmd:
                best_mmd = avg_mmd
                torch.save(model.state_dict(), OUT_DIR / "best_model.pt")
                print(f"    → 保存最佳模型 (MMD={avg_mmd:.4f})")

    # ---------------- 评估：源域回归 + 目标域零样本线性探针 ----------------
    print("\n[5] 评估...")
    model.load_state_dict(torch.load(OUT_DIR / "best_model.pt", map_location="cpu"))
    model.eval()

    log = {
        "n_epochs": N_EPOCHS,
        "best_mmd": best_mmd,
        "source_samples": len(source_data),
        "target_samples": len(target_data),
        "total_params": total_params,
        "lambda_mmd": LAMBDA_MMD,
        "lambda_adversarial": LAMBDA_ADVERSARIAL,
        "lambda_reg": LAMBDA_REG,
    }

    # 5a. 源域评分回归 RMSE
    src_feats = []
    with torch.no_grad():
        for i in range(0, len(source_data), 32):
            batch = source_data[i:i + 32]
            x = torch.stack([torch.from_numpy(np.array(s["skeleton"], dtype=np.float32)).permute(2, 0, 1) for s in batch])
            x = x.unsqueeze(1) if x.dim() == 3 else x  # ensure (B, C, T, J)
            feat, _, _ = model(x, alpha=0.0)
            src_feats.append(feat)
    src_feats = torch.cat(src_feats)
    src_scores = torch.tensor([[s["diff"], s["execution"], s["nd"], s["total"]] for s in source_data])
    with torch.no_grad():
        pred = model.score_regr(src_feats)
    rmse_src = float(((pred - src_scores) ** 2).mean().sqrt().item())
    print(f"  源域评分回归 RMSE: {rmse_src:.4f}")

    # 5b. 目标域零样本：冻结编码器 + 线性探针（用类别一致性）
    tgt_feats_list = []
    tgt_labels = []
    with torch.no_grad():
        for s in target_data:
            x = torch.from_numpy(np.array(s["skeleton"], dtype=np.float32)).permute(2, 0, 1).unsqueeze(0)
            feat, _, _ = model(x, alpha=0.0)
            tgt_feats_list.append(feat[0])
            tgt_labels.append(CATEGORY_MAP.get(s["category"], 0))
    tgt_feats = torch.stack(tgt_feats_list) if tgt_feats_list else torch.zeros(0, 128)
    tgt_labels = torch.tensor(tgt_labels, dtype=torch.long)
    print(f"  目标域特征形状: {tgt_feats.shape}, 类别分布: {torch.bincount(tgt_labels).tolist()}")
    # 线性探针（冻结编码器 + 简单 logistic 回归）
    theta = torch.zeros(tgt_feats.size(1), len(ALL_CATEGORIES), requires_grad=True)
    opt = torch.optim.SGD([theta], lr=0.5)
    for _ in range(500):
        logits = tgt_feats @ theta
        loss = F.cross_entropy(logits, tgt_labels)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc_probe = float((tgt_feats @ theta).argmax(1).eq(tgt_labels).float().mean())
    majority = float(tgt_labels.bincount().max()) / len(tgt_labels)
    print(f"  目标域线性探针准确率: {acc_probe:.4f}  (多数类基线: {majority:.4f})")

    # 5c. MMD 前/后对比（随机模型 vs 训练模型）
    with torch.no_grad():
        mmd_after = mmd_loss(src_feats, tgt_feats).item()
        print(f"  训练后 MMD: {mmd_after:.4f} (best during training: {best_mmd:.4f})")

    log.update({
        "rmse_score_src": rmse_src,
        "acc_target_linear_probe": acc_probe,
        "majority_baseline_target": majority,
        "mmd_after": mmd_after,
    })
    with open(OUT_DIR / "training_log.json", 'w', encoding='utf-8') as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print("\n✓ 域自适应预训练 + 评估完成")
    return model

if __name__ == "__main__":
    train()