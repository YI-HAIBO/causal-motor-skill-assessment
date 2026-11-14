# Causal and Explainable Fine-Grained Motor Skill Assessment with Cross-Domain Transfer

This repository contains the **code and data** accompanying the manuscript
*"Causal and Explainable Fine-Grained Motor Skill Assessment with Cross-Domain Transfer"*
(Yi, H., IEEE Transactions on Cybernetics).

The paper proposes a structural causal model (SCM) for fine-grained motor skill
assessment that (i) treats viewpoint and body size as pure shortcut variables,
(ii) treats equipment as a partially causal variable injected as a structured
prior via FiLM, (iii) learns a causally invariant representation through
invariant risk minimization (IRM) plus mutual-information suppression, and
(iv) outputs auditable counterfactual defect explanations with a probabilistic
Kendall-tau faithfulness bound. All experiments here use only public or
IRB-exempt data.

## Contents

- `scripts/` — training, evaluation, and reproducibility code.
  - `train_stgcn_irm.py` — ST-GCN + IRM pilot training (CPU).
  - `train_stgcn_irm_gpu.py` — GPU training pipeline.
  - `train_classifier.py` — skeleton-feature classifier baseline.
  - `train_domain_adaptation.py` — domain-adaptive pre-training (Scheme D).
  - `train_public_layer_scaleup.py` — multi-seed stratified 5-fold CV.
  - `experiment_causal_validation.py` — counterfactual localization (A),
    cross-equipment shift (B), and MLP baseline (C).
  - `run_pilot_500.py` — reproducible ST-GCN + IRM pilot on the 500
    real skeleton sequences (70/30 split).
  - `generate_supplementary.py` — figure generation.
- `data/` — desensitized skeleton coordinates and derived features.
  - `exercises.json` — Exercises Dataset metadata (MIT-licensed JSON fields).
  - `pose_features.json` — 500-sample skeleton feature vectors.
  - `pose_results.json` — per-sample pose extraction metrics.
  - `classification_results.json` — MLP/SVM cross-validation results.
- `outputs/` — recorded experiment results (JSON).
  - `stgcn_irm/` — pilot training results and 500 desensitized skeleton sequences.
  - `public_layer_scaleup/` — 5-seed x 5-fold cross-validation results.
  - `causal_validation/` — counterfactual localization, cross-equipment shift,
    and baseline results.
  - `domain_adaptation/` — domain-adaptive pre-training log.
- `configs/` — training and environment configurations.

## Reproducing the results

```bash
# 1. Train the ST-GCN + IRM pilot on the 500 real skeleton sequences
python scripts/run_pilot_500.py

# 2. Multi-seed stratified 5-fold cross-validation
python scripts/train_public_layer_scaleup.py

# 3. Causal validation experiments (counterfactual localization,
#    cross-equipment shift, MLP baseline)
python scripts/experiment_causal_validation.py
```

Requires Python 3.11+, PyTorch, NumPy, scikit-learn. The pilot training runs in
approximately 30 minutes on a consumer CPU.

## Data provenance

- The Exercises Dataset metadata (`data/exercises.json`) is released under the
  MIT license by its authors. The associated media (GIFs and thumbnails) are
  copyrighted by Gym Visual (`https://gymvisual.com/`) and are **not** included
  here; only the MIT-licensed metadata and desensitized skeleton coordinates are
  distributed.
- The FineGym-AQA benchmark is a public dataset; the scoring-regression probe in
  the manuscript uses only its publicly released annotations.
- All recorded results in `outputs/` are real and reproducible from the included
  code and data.

## License

Code and data in this repository are released under the MIT License. See
`LICENSE`.
