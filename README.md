# CNN vs ViT vs Hybrid (MedViT) for Medical Image Classification

Portfolio project: comparing architectures trained **from scratch** on MedMNIST.

**Research question:** in medical image classification with limited data and no
pre-training, does a hybrid CNN-Transformer architecture (MedViT-style) outperform
a pure CNN and a pure ViT in performance (accuracy/AUC) and in robustness to
image corruptions?

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

## Roadmap
1. [x] Project structure + PneumoniaMNIST EDA
2. [ ] Training/evaluation pipeline (simple CNN)
3. [ ] ResNet from scratch
4. [ ] ViT from scratch
5. [ ] Simplified MedViT
6. [ ] Additional subsets (DermaMNIST, OrganAMNIST)
7. [ ] Robustness and interpretability
8. [ ] Final documentation
