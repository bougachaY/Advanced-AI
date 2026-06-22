"""Shape tests for the ModalityProjector.

Run with:  pytest tests/test_modality_projector.py
"""

import pytest
import torch
import torch.nn as nn

from models.config import VLMConfig, ViTConfig, LMConfig, ProjectorConfig
from models.modality_projector import ModalityProjector


@pytest.fixture
def cfg():
    return VLMConfig(
        vit=ViTConfig(hidden_dim=768),
        lm=LMConfig(hidden_dim=960),
        projector=ProjectorConfig(pixel_shuffle_factor=4, image_token_length=64),
    )


B = 2
VIT_PATCHES = 1024   # (512/16)^2


class TestModalityProjector:
    def test_init(self, cfg):
        """__init__ must compute input_dim and create a bias-free linear proj.

        pixel_shuffle merges pixel_shuffle_factor² neighbouring patches, so
        each merged token has dimension:
            input_dim = vit.hidden_dim × pixel_shuffle_factor²
                      = 768 × 16 = 12288
        self.proj then maps input_dim → lm.hidden_dim without bias.
        """
        mp = ModalityProjector(cfg)
        assert mp.input_dim == 12288
        assert isinstance(mp.proj, nn.Linear)
        assert mp.proj.weight.shape == (960, 12288)
        assert mp.proj.bias is None

    def test_pixel_shuffle_shape(self, cfg):
        """pixel_shuffle merges neighbouring patches into fewer, wider tokens.

        For factor=4 and 1024 input patches:
            output tokens = 1024 / 4² = 64
            output dim    = 768 × 4² = 12288
        Each output token aggregates a 4×4 spatial neighbourhood of patches.
        """
        mp = ModalityProjector(cfg)
        x = torch.randn(B, VIT_PATCHES, cfg.vit.hidden_dim)
        shuffled = mp.pixel_shuffle(x)
        factor = cfg.projector.pixel_shuffle_factor
        expected_dim = cfg.vit.hidden_dim * (factor ** 2)
        expected_tokens = VIT_PATCHES // (factor ** 2)
        assert shuffled.shape == (B, expected_tokens, expected_dim), (
            f"pixel_shuffle output shape mismatch: {shuffled.shape}"
        )

    def test_forward_shape(self, cfg):
        """Full projector maps ViT tokens [B, 1024, 768] → LM tokens [B, 64, 960].

        Two stages:
        1. pixel_shuffle:  1024 × 768  →  64 × 12288  (spatial compression)
        2. proj (Linear):  64 × 12288  →  64 × 960    (dimension projection)
        """
        mp = ModalityProjector(cfg)
        x = torch.randn(B, VIT_PATCHES, cfg.vit.hidden_dim)
        out = mp(x)
        assert out.shape == (
            B, cfg.projector.image_token_length, cfg.lm.hidden_dim
        ), f"Projector output shape mismatch: {out.shape}"

    def test_input_dim_attribute(self, cfg):
        """input_dim must equal vit.hidden_dim × pixel_shuffle_factor².

        This value is the width of each token after pixel_shuffle and before
        proj.  With vit.hidden_dim=768 and factor=4: 768 × 16 = 12288.
        Storing it as an attribute makes the Linear size self-documenting.
        """
        mp = ModalityProjector(cfg)
        factor = cfg.projector.pixel_shuffle_factor
        expected = cfg.vit.hidden_dim * (factor ** 2)
        assert hasattr(mp, 'input_dim'), (
            "ModalityProjector must have attribute 'input_dim'"
        )
        assert mp.input_dim == expected, (
            f"input_dim should be {expected} (768×16), got {mp.input_dim}"
        )

    def test_proj_is_linear(self, cfg):
        """self.proj must be an nn.Linear mapping input_dim → lm.hidden_dim.

        Weight shape: (lm.hidden_dim, input_dim) = (960, 12288).
        No bias — the projector convention follows SmolLM2 linear layers.
        """
        mp = ModalityProjector(cfg)
        assert hasattr(mp, 'proj'), (
            "ModalityProjector must have attribute 'proj'"
        )
        assert isinstance(mp.proj, nn.Linear)
        assert mp.proj.weight.shape == (cfg.lm.hidden_dim, mp.input_dim)

    def test_dtype_preserved(self, cfg):
        """Projector must not silently cast tokens to a different dtype."""
        mp = ModalityProjector(cfg)
        x = torch.randn(
            B, VIT_PATCHES, cfg.vit.hidden_dim, dtype=torch.float32
        )
        out = mp(x)
        assert out.dtype == torch.float32
