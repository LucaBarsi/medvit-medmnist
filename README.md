# CNN vs ViT vs Hybrid (MedViT) for Medical Image Classification

Portfolio project: comparing architectures trained **from scratch** on MedMNIST.

**Research question:** in medical image classification with limited data and no
pre-training, does a hybrid CNN-Transformer architecture (MedViT-style) outperform
a pure CNN and a pure ViT in performance (accuracy/AUC) and in robustness to
image corruptions?

## Phase 1 summary (PneumoniaMNIST baselines) Four architectures were trained from scratch on PneumoniaMNIST: SimpleCNN (23.6k params), ResNet-18 (11.17M), ViT (138k) and a simplified MedViT (447k). Increased model complexity did not translate into clearly better performance than the SimpleCNN baseline. ViT showed a specificity drop (0.50) consistent with its lack of convolutional inductive bias; MedViT's hybrid design improved AUC over ViT (0.963 vs 0.957) but only marginally recovered specificity (0.526 vs 0.500), suggesting that combining local and global signal helps overall ranking but not necessarily class-boundary bias in this simplified implementation. Full results in `results/` and `notebooks/02-05`.

 ## Phase 2 summary (DermaMNIST) The same four architectures were trained on DermaMNIST (RGB, 7 classes, ~59x class imbalance vs ~3x in PneumoniaMNIST). The ranking completely reversed: ViT led in AUC (0.920) and SimpleCNN was worst (0.906), the opposite of PneumoniaMNIST. ResNet-18 had the best balanced accuracy (0.501) and the most consistent per-class recall, while dermatofibroma (the rarest class) had near-zero recall across most models, confirming severe imbalance dominates performance on rare classes more than architecture choice. MedViT underperformed plain ViT here (unlike phase 1), with a distinct error pattern funneling several classes into "benign keratosis" and "melanocytic nevi". Key conclusion: no architecture is universally best — relative performance depends on data characteristics (channels, class count, imbalance severity). Full results in `results/` and `notebooks/06-07`
 
> Full documentation (methodology, results, analysis) will be written at the end.

## Structure
- `src/`: reusable code (data, models, training, metrics)
- `notebooks/`: exploration and analyses
- `configs/`: experiment configurations
- `results/`: metrics, plots and checkpoints

## Setup
```bash
python -m venv .venv
# Windows (PowerShell): .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/01_eda_pneumonia.ipynb
```

