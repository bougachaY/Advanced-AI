
# Programming Practical 6: Self-Supervised Learning

This Programming Practical explores **Self-Supervised Learning (SSL)** — training deep networks to learn useful representations *without labeled data*. You will progress through three Colab notebooks covering pretext tasks, contrastive learning (SimCLR), and state-of-the-art foundation models (MAE & DINOv2). All experiments run on CIFAR-10 with a shared backbone so results are directly comparable.

---

## Notebook 1 — Pretext Tasks

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/paulnovello/Advanced-AI/blob/main/PP6%3ASSL/ssl_part1_pretext_tasks.ipynb)

Solution: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/paulnovello/Advanced-AI/blob/main/PP6%3A%SSL/ssl_part1_pretext_tasks_solution.ipynb)

All experiments share the **same small CNN backbone** (3 blocks, 256-d output). You first establish a **random baseline** (untrained backbone) and a **supervised upper bound** (~75–85%), then implement three pretext tasks:

1. **Rotation Prediction** — Rotate images by {0°, 90°, 180°, 270°} and predict the angle. Implement `RotationDataset` and a linear head (256 → 4).
2. **Jigsaw Puzzle** — Shuffle a 2×2 grid of patches and predict the permutation. Linear head: 256 → 24.
3. **Colorization** — Convert to grayscale and predict original colors using MSE loss and a small decoder.

Each task is evaluated via **linear probing** and **t-SNE** on CIFAR-10 classes — even though no class labels are used during pretraining.

---

## Notebook 2 — SimCLR: Contrastive Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/paulnovello/Advanced-AI/blob/main/PP6%3ASSL/ssl_part2_simclr.ipynb)

Solution: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/paulnovello/Advanced-AI/blob/main/PP6%3A%SSL/ssl_part2_simclr_solution.ipynb)

SimCLR (Chen et al., 2020) takes one image, creates **two augmented views**, and trains the network to pull them together while pushing apart views from different images.

You will implement:

1. **Augmentation pipeline** — random crop + color jitter + horizontal flip + grayscale, applied **twice independently** per image.
2. **SimCLR model** — the same backbone + a 2-layer MLP projection head (discarded after training).
3. **NT-Xent loss** — the contrastive objective over $2N$ views per batch.
4. **Training** — 10 epochs, batch size 256.
5. **Evaluation** — linear probing, t-SNE, cosine similarity matrix, positive/negative similarity histograms.
6. **Bonus: temperature ablation** — explore the effect of $\tau$ on training and downstream accuracy.

Expected linear probe: **~65–70%**, surpassing all pretext tasks from Notebook 1.

---

## Notebook 3 — Exploring Pre-trained MAE & DINOv2

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/paulnovello/Advanced-AI/blob/main/PP6%3ASSL/ssl_part3_mae_dino.ipynb)

In this notebook you explore **state-of-the-art foundation models** trained at scale — no training required, just inference and analysis.

### Part A — MAE (He et al., CVPR 2022)

Load `facebook/vit-mae-base` and experiment with masked image reconstruction:

- Visualize reconstructions at different masking ratios (25%, 50%, 75%, 90%)
- Run linear probing on CIFAR-10 (expected: ~60–70%)

### Part B — DINOv2 (Oquab et al., 2023)

Load `dinov2_vits14` and explore its emergent properties:

- **Attention maps** — the [CLS] token naturally segments objects without segmentation training
- **Foreground segmentation** — threshold attention to create binary masks
- **PCA on patch features** — semantically similar regions get similar colors across images
- **k-NN classification** — expected ~95–97% on CIFAR-10 with **no training at all**
- **DINOv2 vs supervised ViT** — compare attention patterns

---

## Summary

| Method | Linear Probe (CIFAR-10) | Source |
|---|---|---|
| Random baseline | ~28% | NB 1 |
| Rotation | ~55% | NB 1 |
| Jigsaw | ~52% | NB 1 |
| Colorization | ~45% | NB 1 |
| **SimCLR** | **~65–70%** | NB 2 |
| MAE | ~60–70% | NB 3 |
| **DINOv2** | **~97–98%** | NB 3 |

## Please take the time to give feedback!

Please fill out the [feedback form](https://docs.google.com/forms/d/e/1FAIpQLSd4qRiPho43N8hZEpKEBhLpUe0W-wOoYNQRZj24-elrwj3esA/viewform?usp=publish-editor) to help us improve future practical sessions!
