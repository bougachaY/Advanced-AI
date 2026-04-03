"""
Evaluate CLIP on CIFAR-100.

Zero-shot classification only:
Build text features for all 100 class names using a prompt template,
then rank them by cosine similarity with each image's features.

Usage:
  uv run python eval_cifar100.py
  uv run python eval_cifar100.py --model_name="ViT-L/14" --device="cuda"
  uv run python eval_cifar100.py --batch_size=64
  uv run python eval_cifar100.py --num_samples=1000
"""

import ssl
# School/corporate networks often have a self-signed cert in the chain;
# patch the global SSL context before any download is triggered.
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import numpy as np
from tqdm import tqdm

from clip_api import load, tokenize

import torchvision.datasets as datasets

# -----------------------------------------------------------------------------
model_name        = "ViT-B/32"           # CLIP model variant
device            = "cpu"                 # 'cpu' or 'cuda'
batch_size        = 32                    # batch size for feature extraction
prompt_template   = "a photo of a {}."   # {} is replaced by the class name
data_root         = "./data"              # where to download CIFAR-100
seed              = 1337
num_samples       = None                  # if set, evaluate on only this many test images
# -----------------------------------------------------------------------------
exec(open("configurator.py").read())
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
np.random.seed(seed)

# =============================================================================
# Load model
# =============================================================================

print(f"Loading CLIP model '{model_name}' on {device}...")
model, preprocess = load(model_name, device=device)
model.eval()

# =============================================================================
# Load CIFAR-100 datasets
# Use torchvision's class list — guaranteed to match label indices.
# =============================================================================

print("Loading CIFAR-100...")
cifar100_test = datasets.CIFAR100(
    root=data_root, train=False, download=True, transform=preprocess
)

# `dataset.classes` is the authoritative index→name mapping (alphabetical order)
class_names = cifar100_test.classes   # list of 100 strings

if num_samples is not None:
    if num_samples <= 0:
        raise ValueError(f"num_samples must be > 0, got {num_samples}")
    num_samples = min(num_samples, len(cifar100_test))
    indices = torch.randperm(len(cifar100_test))[:num_samples].tolist()
    cifar100_test = torch.utils.data.Subset(cifar100_test, indices)
    print(f"Using {num_samples} / 10000 CIFAR-100 test images.")

# =============================================================================
# 1. Zero-Shot Classification
# =============================================================================

print("\n" + "=" * 60)
print("1. ZERO-SHOT CLASSIFICATION ON CIFAR-100")
print("=" * 60)

# Build one text feature per class using the prompt template
print(f"Prompt template: \"{prompt_template}\"")
prompts = [prompt_template.format(name) for name in class_names]
text_tokens = tokenize(prompts).to(device)

with torch.no_grad():
    text_features = model.encode_text(text_tokens)           # [100, D]
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# Evaluate on the test set
test_loader = torch.utils.data.DataLoader(
    cifar100_test, batch_size=batch_size, shuffle=False, num_workers=0
)

n_correct_top1 = 0
n_total = 0

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Zero-shot"):
        images = images.to(device)
        labels = labels.to(device)

        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = model.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()  # [B, 100]

        preds_top1 = logits.argmax(dim=1)
        n_correct_top1 += (preds_top1 == labels).sum().item()
        n_total += images.size(0)

zs_top1 = n_correct_top1 / n_total * 100
print(f"\nZero-shot on CIFAR-100 test set ({n_total} images):")
print(f"  Top-1 accuracy: {zs_top1:.2f}%")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Zero-shot  top-1 : {zs_top1:.2f}%")
print()
print("Expected values for ViT-B/32 (CLIP paper, ImageNet prompts):")
print("  Zero-shot top-1  ≈ 65.1%")
