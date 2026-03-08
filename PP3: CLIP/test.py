"""
Minimal shape tests for PP3 CLIP implementation.

Usage:
  uv run python test.py --resattention
  uv run python test.py --vit
  uv run python test.py --clip
  uv run python test.py                  # runs all tests
"""

import argparse
import sys

import torch


def test_resattention():
    from model import ResidualAttentionBlock

    d_model = 32
    n_head = 4
    seq_len = 17
    batch_size = 3

    block = ResidualAttentionBlock(d_model=d_model, n_head=n_head)
    x = torch.randn(seq_len, batch_size, d_model)
    y = block(x)

    assert y.shape == (seq_len, batch_size, d_model), (
        f"ResidualAttentionBlock output shape mismatch: got {tuple(y.shape)}, "
        f"expected {(seq_len, batch_size, d_model)}"
    )
    print(f"[OK] --resattention shape: {tuple(y.shape)}")


def test_vit():
    from model import VisionTransformer

    batch_size = 2
    input_resolution = 224
    patch_size = 32
    width = 64
    layers = 2
    heads = 4
    output_dim = 512

    vit = VisionTransformer(
        input_resolution=input_resolution,
        patch_size=patch_size,
        width=width,
        layers=layers,
        heads=heads,
        output_dim=output_dim,
    )
    x = torch.randn(batch_size, 3, input_resolution, input_resolution)
    y = vit(x)

    assert y.shape == (batch_size, output_dim), (
        f"VisionTransformer output shape mismatch: got {tuple(y.shape)}, "
        f"expected {(batch_size, output_dim)}"
    )
    print(f"[OK] --vit shape: {tuple(y.shape)}")


def test_clip():
    from model import CLIP

    batch_size = 2
    embed_dim = 512
    image_resolution = 224
    vision_layers = 2
    vision_width = 64
    vision_patch_size = 32
    context_length = 77
    vocab_size = 50000
    transformer_width = 64
    transformer_heads = 1
    transformer_layers = 2

    model = CLIP(
        embed_dim=embed_dim,
        image_resolution=image_resolution,
        vision_layers=vision_layers,
        vision_width=vision_width,
        vision_patch_size=vision_patch_size,
        context_length=context_length,
        vocab_size=vocab_size,
        transformer_width=transformer_width,
        transformer_heads=transformer_heads,
        transformer_layers=transformer_layers,
    )
    images = torch.randn(batch_size, 3, image_resolution, image_resolution)
    text = torch.randint(0, vocab_size, (batch_size, context_length), dtype=torch.long)
    logits_per_image, logits_per_text = model(images, text)

    expected = (batch_size, batch_size)
    assert logits_per_image.shape == expected, (
        f"CLIP logits_per_image shape mismatch: got {tuple(logits_per_image.shape)}, expected {expected}"
    )
    assert logits_per_text.shape == expected, (
        f"CLIP logits_per_text shape mismatch: got {tuple(logits_per_text.shape)}, expected {expected}"
    )
    print(
        f"[OK] --clip shapes: logits_per_image={tuple(logits_per_image.shape)}, "
        f"logits_per_text={tuple(logits_per_text.shape)}"
    )


def main():
    parser = argparse.ArgumentParser(description="Run PP3 CLIP shape tests.")
    parser.add_argument("--resattention", action="store_true", help="Test ResidualAttentionBlock.")
    parser.add_argument("--vit", action="store_true", help="Test VisionTransformer.")
    parser.add_argument("--clip", action="store_true", help="Test CLIP forward.")
    args = parser.parse_args()

    selected = []
    if args.resattention:
        selected.append(("resattention", test_resattention))
    if args.vit:
        selected.append(("vit", test_vit))
    if args.clip:
        selected.append(("clip", test_clip))

    if not selected:
        selected = [
            ("resattention", test_resattention),
            ("vit", test_vit),
            ("clip", test_clip),
        ]

    torch.manual_seed(0)
    failed = False

    for name, fn in selected:
        try:
            fn()
        except Exception as exc:
            failed = True
            print(f"[FAIL] --{name}: {exc}")

    if failed:
        sys.exit(1)
    print("All selected tests passed.")


if __name__ == "__main__":
    main()

