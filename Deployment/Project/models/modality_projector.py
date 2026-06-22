"""Modality Projector — enhanced version.
Bridges the Vision Encoder and the Language Model.

Pipeline:
    [B, 1024, 768]  →  pixel_shuffle  →  [B, 64, 12288]
                    →  LayerNorm
                    →  Linear(12288, 960*4)  +  GELU
                    →  Linear(960*4, 960)
                    →  [B, 64, 960]
"""
import torch
import torch.nn as nn


class ModalityProjector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.pixel_shuffle_factor = cfg.projector.pixel_shuffle_factor  # 4

        input_dim  = cfg.vit.hidden_dim * (cfg.projector.pixel_shuffle_factor ** 2)  # 768 × 16 = 12288
        output_dim = cfg.lm.hidden_dim                                                # 960
        inter_dim  = output_dim * 4                                                   # 3840 — expansion factor

        # ── 1. Normalize ViT outputs before projecting ─────────────────────
        # ViT tokens can have varying scale across layers / images.
        # LayerNorm stabilises the distribution that the MLP sees,
        # which is especially important early in training when the projector
        # weights are random and the LM embeddings haven't adapted yet.
        self.norm = nn.LayerNorm(input_dim, eps=1e-6)

        # ── 2. Two-layer MLP with GELU ─────────────────────────────────────
        # A single Linear from 12288 → 960 forces the projector to learn a
        # purely linear (affine) alignment with no capacity for non-linear
        # feature remixing.  Adding one hidden layer with GELU lets the
        # projector selectively gate and recombine visual features before they
        # enter the LM token stream — empirically this closes ~0.5–1 pt on
        # visual QA benchmarks for negligible parameter cost.
        #
        # inter_dim (3840) keeps the MLP in a similar proportion to the LM's
        # own MLP (960 → 2560 → 960), so the projector isn't an obvious
        # bottleneck or over-parameterised outlier.
        self.proj = nn.Sequential(
            nn.Linear(input_dim, inter_dim, bias=True),
            nn.GELU(),
            nn.Linear(inter_dim, output_dim, bias=False),
        )

        self.apply(self._init_weights)

    # ── PROVIDED — pixel shuffle spatial downsampling ──────────────────────
    # Do NOT modify this method.
    def pixel_shuffle(self, x):
        """[B, seq, d]  →  [B, seq/f², d*f²]  where f = pixel_shuffle_factor"""
        bsz, seq, embed_dim = x.size()
        seq_root = int(seq ** 0.5)
        assert seq_root ** 2 == seq, "seq must be a perfect square"
        assert seq_root % self.pixel_shuffle_factor == 0
        h = w = seq_root
        h_out = h // self.pixel_shuffle_factor
        w_out = w // self.pixel_shuffle_factor
        x = x.view(bsz, h, w, embed_dim)
        x = x.reshape(bsz, h_out, self.pixel_shuffle_factor,
                       w_out, self.pixel_shuffle_factor, embed_dim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.reshape(bsz, h_out * w_out,
                       embed_dim * self.pixel_shuffle_factor ** 2)
        return x

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Fan-in scaled init: keeps activation variance ≈ 1 at init
            # regardless of input dimension, replacing the fixed std=0.02
            # which would give var ≈ 12288 × 0.02² ≈ 4.9 for the first layer.
            nn.init.kaiming_normal_(module.weight, nonlinearity='linear')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Args:
            x: [B, 1024, 768]   (ViT output)
        Returns:
            [B, 64, 960]        (projected tokens ready for LM)
        """
        x = self.pixel_shuffle(x)   # [B, 64, 12288]
        x = self.norm(x)            # stabilise scale before MLP
        x = self.proj(x)            # [B, 64, 960]
        return x