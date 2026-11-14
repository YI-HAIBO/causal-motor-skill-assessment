# -*- coding: utf-8 -*-
"""
用真实骨架特征训练简单分类器（MLP），产出真实分类准确率
- 输入: data/pose_features.json (47 样本, 103 维)
- 输出: 分类准确率 / 混淆矩阵 / 特征可分性分析
- 结果写入 data/classification_results.json，供论文实验节使用
"""
import json, os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold

BASE = "C:/Users/Yi/WorkBuddy/2026-11-09-19-01-01"
FEAT_PATH = os.path.join(BASE, "data/pose_features.json")
OUT_PATH = os.path.join(BASE, "data/classification_results.json")

# 固定随机种子
np.random.seed(42)


def main():
    with open(FEAT_PATH, encoding="utf-8") as f:
        data = json.load(f)

    # 构建特征矩阵
    X = []
    y = []
    names = []
    for r in data:
        if "features" not in r:
            continue
        X.append(r["features"])
        y.append(r["category"])
        names.append(r["name"])

    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Classes: {np.unique(y)}")
    print(f"Class counts: {dict(zip(*np.unique(y, return_counts=True)))}")

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ---- 交叉验证 ----
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # MLP
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                        solver="adam", max_iter=2000, random_state=42)
    mlp_scores = cross_val_score(mlp, X_scaled, y, cv=cv, scoring="accuracy")
    print(f"\nMLP 5-fold CV accuracy: {mlp_scores.mean():.4f} ± {mlp_scores.std():.4f}")

    # SVM (RBF)
    svm = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)
    svm_scores = cross_val_score(svm, X_scaled, y, cv=cv, scoring="accuracy")
    print(f"SVM 5-fold CV accuracy: {svm_scores.mean():.4f} ± {svm_scores.std():.4f}")

    # 基线：随机猜测
    from scipy.stats import mode
    baseline = max(np.bincount(np.searchsorted(np.unique(y), y))).sum() / len(y)
    # 更准确的基线
    class_counts = np.bincount(np.searchsorted(np.unique(y), y))
    majority_baseline = class_counts.max() / len(y)
    print(f"Majority-class baseline: {majority_baseline:.4f}")

    # ---- 训练全量模型并报告 ----
    mlp_full = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                             solver="adam", max_iter=2000, random_state=42)
    mlp_full.fit(X_scaled, y)
    y_pred = mlp_full.predict(X_scaled)
    train_acc = accuracy_score(y, y_pred)
    print(f"\nMLP train accuracy (full): {train_acc:.4f}")

    # 混淆矩阵
    classes = sorted(np.unique(y))
    cm = confusion_matrix(y, y_pred, labels=classes)
    print("\nConfusion matrix:")
    print(cm)

    # 逐类准确率 (recall)
    report = classification_report(y, y_pred, labels=classes, output_dict=True)
    per_class_acc = {c: report[c]["recall"] for c in classes}
    print("\nPer-class recall:")
    for c, a in per_class_acc.items():
        print(f"  {c}: {a:.4f}")

    # ---- 特征可分性：类间/类内距离比 ----
    # 用降维后的前2主成分计算
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    # 类间距离（类中心两两平均距离）
    centers = {}
    for c in classes:
        centers[c] = X_pca[y == c].mean(axis=0)
    inter_dist = 0.0
    pairs = 0
    cs = list(centers.keys())
    for i in range(len(cs)):
        for j in range(i + 1, len(cs)):
            inter_dist += np.linalg.norm(centers[cs[i]] - centers[cs[j]])
            pairs += 1
    inter_dist /= pairs
    # 类内距离（类内样本到类中心的平均距离）
    intra_dist = 0.0
    for c in classes:
        intra_dist += np.mean(np.linalg.norm(X_pca[y == c] - centers[c], axis=1))
    intra_dist /= len(classes)
    separation_ratio = inter_dist / intra_dist if intra_dist > 0 else 0
    print(f"\nPCA(2) inter-class distance: {inter_dist:.4f}")
    print(f"PCA(2) intra-class distance: {intra_dist:.4f}")
    print(f"Separation ratio: {separation_ratio:.4f}")

    # ---- 保存结果 ----
    results = {
        "n_samples": len(X),
        "n_features": X.shape[1],
        "n_classes": len(classes),
        "classes": classes,
        "class_counts": dict(zip(classes, class_counts.tolist())),
        "mlp_cv_accuracy_mean": float(mlp_scores.mean()),
        "mlp_cv_accuracy_std": float(mlp_scores.std()),
        "mlp_cv_fold_scores": mlp_scores.tolist(),
        "svm_cv_accuracy_mean": float(svm_scores.mean()),
        "svm_cv_accuracy_std": float(svm_scores.std()),
        "svm_cv_fold_scores": svm_scores.tolist(),
        "majority_baseline": float(majority_baseline),
        "mlp_train_accuracy": float(train_acc),
        "per_class_recall": per_class_acc,
        "confusion_matrix": cm.tolist(),
        "pca_inter_class_dist": float(inter_dist),
        "pca_intra_class_dist": float(intra_dist),
        "pca_separation_ratio": float(separation_ratio),
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()