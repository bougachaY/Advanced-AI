# Programming Practical 2: ViT, CLIP, VLM

This practical session explores the key building blocks behind modern vision-language models through three progressive notebooks. Each one builds on the concepts introduced in the previous one.

## 1. Vision Transformer (ViT)

We start with the **Vision Transformer**, the architecture that brought the transformer paradigm from NLP to computer vision. In this notebook you will load a pretrained ViT, visualize how an image is split into patch tokens, inspect attention maps, and run image classification inference.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/paulnovello/Advanced-AI/blob/main/PP2%3A%20Vision/VIT_solution.ipynb)

## 2. CLIP

A vanilla ViT can only classify images into a fixed set of categories. **CLIP** (Contrastive Language-Image Pretraining) overcomes this limitation by learning a shared embedding space for images and text. In this notebook you will see how CLIP aligns visual and textual representations, enabling zero-shot classification, image-text retrieval, and open-vocabulary recognition.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/paulnovello/Advanced-AI/blob/main/PP2%3A%20Vision/CLIP_solution.ipynb)

## 3. Vision-Language Model (VLM)

With a vision encoder (ViT) and a way to bridge images and text (CLIP), the final step is to build a **Vision-Language Model** that can actually *reason* about images in natural language. In this notebook you will dissect a LLaVA-style VLM (SigLIP + MLP projector + Qwen2) and understand how each component contributes to the full pipeline — from pixels to conversational answers.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/paulnovello/Advanced-AI/blob/main/PP2%3A%20Vision/VLM_solution.ipynb)
