"""Shape + correctness tests for the Language Model components.

Key test: test_kv_cache_consistency — verifies that running the full sequence
at once (prefill) gives the same logits as running the prefix first and then
appending one token at a time (decode with KV cache).  If this test fails,
there is a bug in your KV cache or RoPE implementation.

Run with:  pytest tests/test_language_model.py
"""

import pytest
import torch

from models.config import LMConfig
from models.language_model import (
    RMSNorm, LMAttention, LMMLP, LMBlock, LanguageModel, RotaryEmbedding,
)


@pytest.fixture
def cfg():
    return LMConfig(
        hidden_dim=32,
        inter_dim=64,
        rms_eps=1e-5,
        re_base=10000,
        max_position_embeddings=128,
        vocab_size=256,
        base_vocab_size=255,
        n_heads=4,        # head_dim = 32/4 = 8
        n_kv_heads=2,     # n_kv_groups = 2
        n_blocks=2,
        dropout=0.0,
        attn_scaling=1.0,
        tie_weights=True,
    )


B, T = 2, 10


class TestRMSNorm:
    def test_init(self, cfg):
        """__init__ must create one learnable weight per hidden dimension.

        Unlike LayerNorm, RMSNorm has no bias — only a gain vector.
        rms_eps must be stored as an attribute (used in forward to avoid
        division by zero when the RMS is very small).
        """
        norm = RMSNorm(cfg)
        assert norm.weight.shape == (32,)
        assert norm.rms_eps == 1e-5

    def test_output_shape(self, cfg):
        """RMSNorm is a pointwise operation — it must not change shape."""
        norm = RMSNorm(cfg)
        x = torch.randn(B, T, cfg.hidden_dim)
        assert norm(x).shape == (B, T, cfg.hidden_dim)

    def test_scale_invariance(self, cfg):
        """Multiplying the input by a scalar must not change the output.

        RMSNorm divides by the root-mean-square of x, so scaling x by k
        scales both x and its RMS by k — the ratio stays constant.
        If this fails, you forgot to normalise (divide by the RMS).
        """
        norm = RMSNorm(cfg)
        norm.weight.data.fill_(1.0)
        norm.eval()
        x = torch.randn(B, T, cfg.hidden_dim)
        with torch.no_grad():
            out1 = norm(x)
            out2 = norm(x * 10)
        assert torch.allclose(out1, out2, atol=1e-5), (
            "RMSNorm output changed when input was scaled by 10"
        )

    def test_weight_effect(self, cfg):
        """Doubling the weight parameter must exactly double the output.

        After normalisation the output is just x_norm * weight, so
        out(2w) = 2 * out(w) for any input x.
        """
        norm = RMSNorm(cfg)
        x = torch.randn(B, T, cfg.hidden_dim)
        with torch.no_grad():
            norm.weight.data.fill_(1.0)
            out1 = norm(x)
            norm.weight.data.fill_(2.0)
            out2 = norm(x)
        assert torch.allclose(out2, 2 * out1, atol=1e-6)


class TestRotaryEmbedding:
    def test_init(self, cfg):
        """__init__ (PROVIDED) must precompute inv_freq of shape [dim/2].

        dim = head_dim = hidden_dim // n_heads = 32 // 4 = 8
        inv_freq[i] = 1 / (re_base ^ (2i / dim))  for i in 0..dim/2-1
        → shape [4].  Also checks that dim and re_base are stored correctly.
        """
        rope = RotaryEmbedding(cfg)
        assert rope.inv_freq.shape == (4,)
        assert rope.dim == 8
        assert rope.re_base == 10000


class TestLMAttention:
    def test_init(self, cfg):
        """GQA projections: q is full width, k/v use n_kv_heads × head_dim.

        With n_heads=4, n_kv_heads=2, head_dim=8:
            q_proj : Linear(32 → 32)   weight (32, 32)
            k_proj : Linear(32 → 16)   weight (16, 32)   16 = 2×8
            v_proj : Linear(32 → 16)   weight (16, 32)
            out_proj: Linear(32 → 32)  weight (32, 32)
        All projections are bias-free (SmolLM2 convention).
        n_kv_groups = n_heads // n_kv_heads = 2.
        head_dim = hidden_dim // n_heads = 8.
        """
        attn = LMAttention(cfg)
        assert attn.head_dim == 8
        assert attn.n_kv_groups == 2
        assert attn.q_proj.weight.shape == (32, 32)
        assert attn.k_proj.weight.shape == (16, 32)
        assert attn.v_proj.weight.shape == (16, 32)
        assert attn.out_proj.weight.shape == (32, 32)
        assert attn.q_proj.bias is None

    def test_output_shape_prefill(self, cfg):
        """Prefill (no cache): output shape matches input, cache has key/value."""
        attn = LMAttention(cfg)
        x = torch.randn(B, T, cfg.hidden_dim)
        rope = RotaryEmbedding(cfg)
        pos_ids = torch.arange(T).unsqueeze(0).expand(B, -1)
        cos, sin = rope(pos_ids)
        out, cache = attn(x, cos, sin, block_kv_cache=None)
        assert out.shape == (B, T, cfg.hidden_dim)
        assert 'key' in cache and 'value' in cache

    def test_kv_cache_shape(self, cfg):
        """Cached K and V must be [B, n_kv_heads, T, head_dim].

        Note: only n_kv_heads (not n_heads) KV pairs are stored — that is
        the whole point of Grouped-Query Attention.
        """
        attn = LMAttention(cfg)
        x = torch.randn(B, T, cfg.hidden_dim)
        rope = RotaryEmbedding(cfg)
        pos_ids = torch.arange(T).unsqueeze(0).expand(B, -1)
        cos, sin = rope(pos_ids)
        _, cache = attn(x, cos, sin, block_kv_cache=None)
        head_dim = cfg.hidden_dim // cfg.n_heads
        assert cache['key'].shape == (B, cfg.n_kv_heads, T, head_dim)
        assert cache['value'].shape == (B, cfg.n_kv_heads, T, head_dim)


class TestLMMLp:
    def test_init(self, cfg):
        """SwiGLU MLP has three projections: gate, up (both expand), down (contracts).

        gate_proj and up_proj both map hidden_dim → inter_dim.
        down_proj maps inter_dim → hidden_dim.
        All are bias-free (SmolLM2 convention).
        Shapes with hidden_dim=32, inter_dim=64:
            gate_proj.weight: (64, 32)
            up_proj.weight:   (64, 32)
            down_proj.weight: (32, 64)
        """
        mlp = LMMLP(cfg)
        assert mlp.gate_proj.weight.shape == (64, 32)
        assert mlp.up_proj.weight.shape == (64, 32)
        assert mlp.down_proj.weight.shape == (32, 64)

    def test_output_shape(self, cfg):
        """MLP is applied position-wise — shape [B, T, C] in, [B, T, C] out."""
        mlp = LMMLP(cfg)
        x = torch.randn(B, T, cfg.hidden_dim)
        assert mlp(x).shape == (B, T, cfg.hidden_dim)


class TestLMBlock:
    def test_init(self, cfg):
        """LMBlock must wire two RMSNorms, one LMAttention, and one LMMLP.

        Attribute names matter: LanguageModel.from_pretrained maps HF keys to
        norm1, norm2, attn, mlp — any other names will break weight loading.
        """
        block = LMBlock(cfg)
        assert isinstance(block.norm1, RMSNorm)
        assert isinstance(block.norm2, RMSNorm)
        assert isinstance(block.attn, LMAttention)
        assert isinstance(block.mlp, LMMLP)

    def test_output_shape(self, cfg):
        """Block is a residual unit — shape is unchanged and cache is returned."""
        block = LMBlock(cfg)
        x = torch.randn(B, T, cfg.hidden_dim)
        rope = RotaryEmbedding(cfg)
        pos_ids = torch.arange(T).unsqueeze(0).expand(B, -1)
        cos, sin = rope(pos_ids)
        out, cache = block(x, cos, sin)
        assert out.shape == (B, T, cfg.hidden_dim)
        assert cache is not None


class TestLanguageModel:
    def test_init(self, cfg):
        """Embedding table and output head must share the same tensor (weight tying).

        SmolLM2 uses tied weights: head.weight IS token_embedding.weight.
        This halves the parameter count for the vocabulary matrices.
        data_ptr() returns the memory address — equality means the same tensor.
        """
        model = LanguageModel(cfg)
        assert model.token_embedding.weight.shape == (256, 32)
        assert model.head.weight.shape == (256, 32)
        assert len(model.blocks) == 2
        assert (
            model.head.weight.data_ptr()
            == model.token_embedding.weight.data_ptr()
        )

    def test_forward_shape(self, cfg):
        """forward() returns (hidden_states, kv_cache).

        hidden_states shape: [B, T, hidden_dim] — same as input.
        kv_cache: list of n_blocks dicts (one per transformer block).
        Note: forward takes embeddings, not token ids.
        """
        model = LanguageModel(cfg)
        x = torch.randn(B, T, cfg.hidden_dim)  # embeddings, not token ids
        hidden, kv = model(x)
        assert hidden.shape == (B, T, cfg.hidden_dim)
        assert len(kv) == cfg.n_blocks

    def test_head_shape(self, cfg):
        """head projects hidden states to one logit per vocabulary entry."""
        model = LanguageModel(cfg)
        x = torch.randn(B, T, cfg.hidden_dim)
        hidden, _ = model(x)
        logits = model.head(hidden)
        assert logits.shape == (B, T, cfg.vocab_size)

    def test_kv_cache_consistency(self, cfg):
        """Prefill + single-token decode must match full-sequence forward.

        This is the key correctness test for KV caching + RoPE.

        Strategy:
          1. Full sequence [tok_0, ..., tok_{T-1}] in one shot → logits_full
          2. Prefix [tok_0, ..., tok_{T-2}] to build the KV cache
          3. Just [tok_{T-1}] with start_pos=T-1 → logits_last
          4. logits_last should equal logits_full[:, -1, :]
        """
        torch.manual_seed(42)
        model = LanguageModel(cfg)
        model.eval()

        x = torch.randn(1, T, cfg.hidden_dim)

        with torch.no_grad():
            hidden_full, _ = model(x, kv_cache=None, start_pos=0)
            logits_full = model.head(hidden_full)           # [1, T, vocab]

            hidden_prefix, kv = model(
                x[:, :-1, :], kv_cache=None, start_pos=0
            )
            hidden_last, _ = model(
                x[:, -1:, :], kv_cache=kv, start_pos=T - 1
            )
            logits_last = model.head(hidden_last)           # [1, 1, vocab]

        torch.testing.assert_close(
            logits_full[:, -1:, :], logits_last,
            atol=1e-4, rtol=1e-4,
            msg=(
                "KV cache decode does not match full-sequence forward — "
                "check your KV concatenation and start_pos in RoPE."
            ),
        )


@pytest.mark.slow
class TestLanguageModelPretrainedLoading:
    """Load real SmolLM2-360M weights and verify architecture.

    Skipped by default — requires ~720 MB download.
    Run with:  pytest tests/test_language_model.py -m slow
    """

    @pytest.fixture(scope="class")
    def pretrained(self):
        from models.config import LMConfig
        cfg = LMConfig()
        model = LanguageModel.from_pretrained(cfg)
        return model, cfg

    def test_parameter_count(self, pretrained):
        """SmolLM2-360M-Instruct has ~362 M parameters."""
        model, _ = pretrained
        n = sum(p.numel() for p in model.parameters())
        assert 300_000_000 < n < 420_000_000, (
            f"Unexpected param count: {n:,}"
        )

    def test_config_updated(self, pretrained):
        """from_pretrained must mutate cfg to match the HF config."""
        _, cfg = pretrained
        assert cfg.hidden_dim == 960
        assert cfg.n_heads == 15
        assert cfg.n_kv_heads == 5
        assert cfg.n_blocks == 32
        assert cfg.inter_dim == 2560

    def test_inv_freq_shape(self, pretrained):
        """inv_freq must have shape [head_dim / 2]."""
        model, cfg = pretrained
        head_dim = cfg.hidden_dim // cfg.n_heads  # 64
        assert model.rotary_embd.inv_freq.shape == (head_dim // 2,)

    def test_weight_tying(self, pretrained):
        """Embedding and output head must share the same tensor."""
        model, _ = pretrained
        assert (
            model.head.weight.data_ptr()
            == model.token_embedding.weight.data_ptr()
        )

    def test_vocab_extended(self, pretrained):
        """Embedding rows must equal vocab_size (49152 + 1 image token)."""
        model, cfg = pretrained
        assert model.token_embedding.weight.shape[0] == cfg.vocab_size
        assert cfg.vocab_size == 49153
