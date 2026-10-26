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

## Phase 3 summary (OrganAMNIST)

OrganAMNIST (grayscale abdominal CT, 11 organ classes, 34,561 training images)
has a mild imbalance (4.5x) and, unlike DermaMNIST, plenty of examples even in
its rarest class (1,357 images). All four architectures reached balanced
accuracy between 0.84 and 0.91, versus 0.36-0.50 on DermaMNIST. ResNet-18 was
again the strongest (balanced accuracy 0.914, AUC 0.996) and MedViT again beat
plain ViT (0.882 vs 0.844). The weakest class for every model was the left
kidney (recall 0.63-0.77), mostly confused with the right kidney and spleen.

## Results at a glance

Test-set results, one training run per model and subset (seed 42).

| Balanced accuracy | PneumoniaMNIST | DermaMNIST | OrganAMNIST |
|---|---|---|---|
| SimpleCNN (24k params) | 0.825 | 0.361 | 0.892 |
| ResNet-18 (11.2M) | **0.839** | **0.501** | **0.914** |
| ViT (139k) | 0.750 | 0.456 | 0.844 |
| MedViT, simplified (448k) | 0.762 | 0.381 | 0.882 |

| AUC | PneumoniaMNIST | DermaMNIST | OrganAMNIST |
|---|---|---|---|
| SimpleCNN | **0.968** | 0.906 | 0.993 |
| ResNet-18 | 0.953 | **0.921** | **0.996** |
| ViT | 0.957 | 0.920 | 0.986 |
| MedViT, simplified | 0.963 | 0.910 | 0.991 |

## Conclusion

Research question: does a hybrid CNN-Transformer (MedViT-style) outperform a
pure CNN and a pure ViT on small medical images, with limited data and no
pre-training?

In this study, not against the best CNN: the simplified MedViT never beat
ResNet-18. It did beat the plain ViT on two of three subsets.

1. **ResNet-18 was the most consistent model** by balanced accuracy in all
   three subsets. Its margin over the next best model was small on
   PneumoniaMNIST (+0.015, close to run-to-run variation), and larger on
   DermaMNIST (+0.045) and OrganAMNIST (+0.022).
2. **MedViT vs ViT depends on the data.** MedViT was ahead on PneumoniaMNIST
   (+0.012 balanced accuracy) and OrganAMNIST (+0.038), and behind on
   DermaMNIST (-0.076). Only the OrganAMNIST gain and the DermaMNIST loss are
   clearly beyond run-to-run noise.
3. **Examples per class matter a lot.** On DermaMNIST (rarest class: 80
   images) the rarest class had recall of 0.00-0.13 in every model, while
   OrganAMNIST (rarest class: 1,357 images) reached 0.84-0.91 balanced
   accuracy. This is confounded with task difficulty and is not isolated here.
4. **Metric choice matters.** AUC saturates on easier tasks and gaps below
   ~0.01 are within noise; balanced accuracy separated the models better.

## Limitations

- One training run per model and subset; reruns showed variation of roughly
  0.005-0.01 in AUC and balanced accuracy.
- MedViT is a simplified re-implementation: its attention mechanisms are
  replaced by simpler stand-ins, so conclusions apply to this implementation,
  not to the original MedViT.
- No hyperparameter tuning: all models share the same learning rate, epochs
  and early stopping. All images are 28x28, and no model is pre-trained.

## Future work

- Repeat with several seeds and report mean and standard deviation.
- Controlled experiment: subsample OrganAMNIST classes to ~80 images to
  separate the effect of example count from task difficulty.
- Class-weighted loss or oversampling for DermaMNIST's rare classes.
- Robustness (noise, blur, contrast) and interpretability (attention maps,
  Grad-CAM).
- Higher resolutions (MedMNIST+ at 64/128/224) and a fuller MedViT
  implementation.