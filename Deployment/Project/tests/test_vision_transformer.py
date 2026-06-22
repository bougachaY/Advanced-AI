"""Shape tests for the ViT components.

All tests run on CPU with a tiny configuration (small image, few blocks)
so they complete in seconds.
Run with:  pytest tests/test_vision_transformer.py
"""

import pytest
import torch
import torch.nn as nn

from models.config import ViTConfig
from models.vision_transformer import (
    ViTPatchEmbeddings, ViTAttention, ViTMLP, ViTBlock, ViT,
)


# Tiny config for fast CPU tests — different from the real 512/12/768 values
# so shape bugs are caught immediately.
@pytest.fixture
def cfg():
    return ViTConfig(
        img_size=64,     # 64×64 images  →  (64/8)^2 = 64 patches
        patch_size=8,
        hidden_dim=32,
        inter_dim=64,    # 2 × hidden_dim
        n_heads=4,       # head_dim = 32/4 = 8
        n_blocks=2,
        dropout=0.0,
        ln_eps=1e-6,
        cls_flag=False,
    )


B = 2   # batch size used throughout


class TestViTPatchEmbeddings:
    def test_init(self, cfg):
        """__init__ must create a Conv2d that extracts patches in one pass.

        The conv kernel and stride must both equal patch_size so each kernel
        application covers exactly one non-overlapping patch.
        out_channels == hidden_dim so each patch is immediately projected.
        position_embedding is a learnable [1, num_patches, hidden_dim] tensor
        that is broadcast-added to every sample in the batch.
        """
        model = ViTPatchEmbeddings(cfg)
        assert model.conv.weight.shape == (32, 3, 8, 8)
        assert model.conv.stride == (8, 8)
        assert model.position_embedding.shape == (1, 64, 32)

    def test_output_shape(self, cfg):
        """forward must return [B, num_patches, hidden_dim].

        For img_size=64 and patch_size=8:
            num_patches = (64 // 8) ** 2 = 64
        """
        model = ViTPatchEmbeddings(cfg)
        x = torch.randn(B, 3, cfg.img_size, cfg.img_size)
        out = model(x)
        n_patches = (cfg.img_size // cfg.patch_size) ** 2
        expected = (B, n_patches, cfg.hidden_dim)
        assert out.shape == expected, f"Expected {expected}, got {out.shape}"

    def test_position_embedding_added(self, cfg):
        """Position embedding must actually be added to the patch features.

        Strategy: run forward with position_embedding zeroed, then with it
        filled with ones.  The two outputs must differ — if they don't,
        position_embedding is never added.
        """
        model = ViTPatchEmbeddings(cfg)
        model.position_embedding.data.zero_()
        x = torch.randn(B, 3, cfg.img_size, cfg.img_size)
        out_no_pos = model(x).clone()
        model.position_embedding.data.fill_(1.0)
        out_with_pos = model(x)
        assert not torch.allclose(out_no_pos, out_with_pos), (
            "Position embedding had no effect"
        )


class TestViTAttention:
    def test_init(self, cfg):
        """__init__ must fuse Q, K, V into a single qkv_proj linear layer.

        This is required for weight loading: ViT.from_pretrained concatenates
        the separate HF q/k/v matrices into one tensor and copies it into
        qkv_proj.weight.  Both qkv_proj and out_proj must have bias=True
        (SigLIP2 uses biases in attention).
        head_dim = hidden_dim // n_heads = 32 // 4 = 8.
        """
        model = ViTAttention(cfg)
        assert model.head_dim == 8
        assert model.qkv_proj.weight.shape == (96, 32)
        assert model.out_proj.weight.shape == (32, 32)
        assert model.qkv_proj.bias is not None
        assert model.out_proj.bias is not None

    def test_output_shape(self, cfg):
        """Attention is a token-mixing operation — it must not change shape."""
        model = ViTAttention(cfg)
        T = (cfg.img_size // cfg.patch_size) ** 2  # 64
        x = torch.randn(B, T, cfg.hidden_dim)
        out = model(x)
        assert out.shape == (B, T, cfg.hidden_dim)

    def test_bidirectional(self, cfg):
        """ViT attention is NOT causal — no causal mask, all tokens attend."""
        model = ViTAttention(cfg)
        model.eval()
        T = 4
        x = torch.randn(B, T, cfg.hidden_dim)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (B, T, cfg.hidden_dim)
        assert out.abs().sum() > 0


class TestViTMLP:
    def test_init(self, cfg):
        """fc1 expands hidden_dim → inter_dim; fc2 projects back.

        The activation between them must be GELU (SigLIP2 uses GELU, not ReLU
        or SiLU).  Shapes:
            fc1.weight: (inter_dim, hidden_dim) = (64, 32)
            fc2.weight: (hidden_dim, inter_dim) = (32, 64)
        """
        model = ViTMLP(cfg)
        assert model.fc1.weight.shape == (64, 32)
        assert model.fc2.weight.shape == (32, 64)
        assert isinstance(model.activation_fn, nn.GELU)

    def test_output_shape(self, cfg):
        """MLP is applied position-wise — shape [B, T, C] in, [B, T, C] out."""
        model = ViTMLP(cfg)
        T = 64
        x = torch.randn(B, T, cfg.hidden_dim)
        out = model(x)
        assert out.shape == (B, T, cfg.hidden_dim)


class TestViTBlock:
    def test_init(self, cfg):
        """A ViTBlock contains exactly two LayerNorms, one attention, one MLP.

        Attribute names matter: ViT.from_pretrained maps HF keys to ln1, ln2,
        attn, mlp — any other names will break weight loading.
        Each LayerNorm normalises over hidden_dim = 32 features.
        """
        model = ViTBlock(cfg)
        assert model.ln1.weight.shape == (32,)
        assert model.ln2.weight.shape == (32,)
        assert isinstance(model.attn, ViTAttention)
        assert isinstance(model.mlp, ViTMLP)

    def test_output_shape(self, cfg):
        """Block is a residual unit — shape is unchanged."""
        model = ViTBlock(cfg)
        T = 64
        x = torch.randn(B, T, cfg.hidden_dim)
        out = model(x)
        assert out.shape == (B, T, cfg.hidden_dim)

    def test_residual_connection(self, cfg):
        """Residuals must preserve input when all projection weights are zero.

        When all Linear weights are zero, attention and MLP output zero.
        The x = x + 0 pattern means block output must equal input.
        If this fails, your residual connection is missing or wrong.
        """
        model = ViTBlock(cfg)
        for p in model.parameters():
            p.data.zero_()
        model.ln1.weight.data.fill_(1.0)
        model.ln2.weight.data.fill_(1.0)
        T = 4
        x = torch.randn(B, T, cfg.hidden_dim)
        out = model(x)
        assert out.shape == (B, T, cfg.hidden_dim)


class TestViT:
    def test_init(self, cfg):
        """ViT must build n_blocks transformer blocks and a final LayerNorm.

        patch_embedding converts images to tokens; blocks refine them;
        layer_norm normalises the output before it is passed to the projector.
        """
        model = ViT(cfg)
        assert len(model.blocks) == 2
        assert isinstance(model.patch_embedding, ViTPatchEmbeddings)
        assert model.layer_norm.weight.shape == (32,)

    def test_output_shape(self, cfg):
        """Full ViT maps [B, 3, H, W] → [B, num_patches, hidden_dim]."""
        model = ViT(cfg)
        x = torch.randn(B, 3, cfg.img_size, cfg.img_size)
        out = model(x)
        num_patches = (cfg.img_size // cfg.patch_size) ** 2
        assert out.shape == (B, num_patches, cfg.hidden_dim)

    def test_output_dtype(self, cfg):
        """Output dtype must be float32 — no accidental half-precision cast."""
        model = ViT(cfg)
        x = torch.randn(B, 3, cfg.img_size, cfg.img_size)
        out = model(x)
        assert out.dtype == torch.float32

    def test_different_batch_sizes(self, cfg):
        """Model must handle any batch size (no hard-coded B dimension)."""
        model = ViT(cfg)
        num_patches = (cfg.img_size // cfg.patch_size) ** 2
        for b in [1, 3, 4]:
            x = torch.randn(b, 3, cfg.img_size, cfg.img_size)
            out = model(x)
            assert out.shape == (b, num_patches, cfg.hidden_dim)


@pytest.mark.slow
class TestViTPretrainedLoading:
    """Load real SigLIP2 weights and verify architecture.

    Skipped by default — requires ~350 MB download.
    Run with:  pytest tests/test_vision_transformer.py -m slow
    """

    @pytest.fixture(scope="class")
    def pretrained(self):
        cfg = ViTConfig()
        model = ViT.from_pretrained(cfg)
        return model, cfg

    def test_parameter_count(self, pretrained):
        """SigLIP2-base-patch16-512 has ~86 M parameters."""
        model, _ = pretrained
        n = sum(p.numel() for p in model.parameters())
        assert 80_000_000 < n < 100_000_000, (
            f"Unexpected param count: {n:,}"
        )

    def test_config_updated(self, pretrained):
        """from_pretrained must mutate cfg to match the HF config."""
        _, cfg = pretrained
        assert cfg.hidden_dim == 768
        assert cfg.n_heads == 12
        assert cfg.n_blocks == 12
        assert cfg.patch_size == 16
        assert cfg.img_size == 512

    def test_patch_embedding_shape(self, pretrained):
        """Conv weight shape must match (hidden_dim, 3, patch, patch)."""
        model, cfg = pretrained
        w = model.patch_embedding.conv.weight
        assert w.shape == (
            cfg.hidden_dim, 3, cfg.patch_size, cfg.patch_size
        )

    def test_position_embedding_shape(self, pretrained):
        """Position embedding shape must be (1, n_patches, hidden_dim)."""
        model, cfg = pretrained
        n_patches = (cfg.img_size // cfg.patch_size) ** 2
        assert model.patch_embedding.position_embedding.shape == (
            1, n_patches, cfg.hidden_dim
        )
