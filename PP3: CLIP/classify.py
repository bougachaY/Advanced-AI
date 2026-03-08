"""
Zero-shot image classification with CLIP.

Given an image and a list of candidate labels, rank the labels by how well
they match the image according to the CLIP model.

Usage examples:
  uv run python classify.py --image_path="cat.jpg"
  uv run python classify.py --image_path="photo.jpg" --model_name="ViT-L/14" --top_k=3
  uv run python classify.py --image_path="photo.jpg" --labels="['a cat','a dog','a car']"
  uv run python classify.py --image_path="photo.jpg" --prompt_template="this is a photo of a {}"
"""

import os
import torch
from PIL import Image
from clip_api import load, tokenize, available_models

# -----------------------------------------------------------------------------
# Configuration — all variables below can be overridden from the command line:
#   uv run python classify.py --image_path="dog.jpg" --top_k=3
model_name = "ViT-B/32"   # model variant; see available_models() for full list
image_path = None          # path to the input image (required)
labels = [                 # candidate text labels (without the prompt template)
    "a cat",
    "a dog",
    "a car",
    "a bird",
    "a person",
    "a house",
    "a tree",
    "an airplane",
    "a boat",
    "a horse",
]
prompt_template = "a photo of {}"  # each label is inserted as {}
top_k = 5                  # number of top predictions to display
device = "cpu"             # 'cpu', 'cuda', 'cuda:0', etc.
seed = 1337
# -----------------------------------------------------------------------------
exec(open("configurator.py").read())  # overrides from command line or config file
# -----------------------------------------------------------------------------

assert image_path is not None, (
    "Please provide an image path, e.g.:\n"
    "  uv run python classify.py --image_path=\"photo.jpg\""
)
assert os.path.isfile(image_path), f"Image not found: {image_path}"
assert model_name in available_models(), (
    f"Unknown model '{model_name}'. Available: {available_models()}"
)

torch.manual_seed(seed)

# ── load model ────────────────────────────────────────────────────────────────
print(f"Loading CLIP model '{model_name}' on {device}...")
model, preprocess = load(model_name, device=device)
model.eval()

# ── preprocess image ──────────────────────────────────────────────────────────
image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

# ── tokenise labels ───────────────────────────────────────────────────────────
prompts = [prompt_template.format(label) for label in labels]
text_tokens = tokenize(prompts).to(device)

# ── run inference ─────────────────────────────────────────────────────────────
with torch.no_grad():
    logits_per_image, _ = model(image, text_tokens)
    probs = logits_per_image.softmax(dim=-1).squeeze(0)   # shape [num_labels]

# ── display results ───────────────────────────────────────────────────────────
k = min(top_k, len(labels))
top_probs, top_indices = probs.topk(k)

print(f"\nImage : {image_path}")
print(f"Model : {model_name}")
print(f"Prompt: \"{prompt_template}\"")
print()
print(f"Top-{k} predictions:")
print("-" * 40)
for rank, (prob, idx) in enumerate(zip(top_probs, top_indices), start=1):
    label = labels[idx.item()]
    print(f"  {rank}. {label:<25s}  {prob.item()*100:.1f}%")
print("-" * 40)