"""Integration tests: load pretrained weights and verify.

These tests download ~800 MB from HuggingFace and run on GPU (or CPU).
They are marked @pytest.mark.slow and skipped by default.

Run with:  pytest tests/test_pretrained_loading.py -m slow

If your architecture is correct, from_pretrained() will succeed and the
model will run a forward pass.  A wrong parameter name, wrong Linear size,
or wrong weight tying will raise an error here.
"""

import pytest
import torch

from models.config import VLMConfig


pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def cfg():
    return VLMConfig()


@pytest.fixture(scope="module")
def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class TestViTPretrainedLoading:
    def test_loads_without_error(self, cfg):
        """ViT.from_pretrained must complete without raising an exception.

        Any attribute name mismatch (e.g. wrong conv or qkv_proj name) causes
        a key error when loading the state_dict — fix your __init__ first.
        """
        from models.vision_transformer import ViT
        model = ViT.from_pretrained(cfg.vit)
        assert model is not None

    def test_parameter_count(self, cfg):
        """SigLIP2-base-patch16-512 has ~86 M parameters."""
        from models.vision_transformer import ViT
        model = ViT.from_pretrained(cfg.vit)
        n = sum(p.numel() for p in model.parameters())
        assert 80_000_000 < n < 100_000_000, f"Unexpected param count: {n:,}"

    def test_forward_shape(self, cfg, device):
        """A 512×512 image must produce [1, 1024, 768] features."""
        from models.vision_transformer import ViT
        model = ViT.from_pretrained(cfg.vit).to(device).eval()
        x = torch.randn(1, 3, 512, 512, device=device)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1024, cfg.vit.hidden_dim)


class TestLMPretrainedLoading:
    def test_loads_without_error(self, cfg):
        """LanguageModel.from_pretrained must complete without exceptions.

        Any attribute name mismatch (e.g. wrong q_proj or out_proj name) causes
        a key error when loading the state_dict — fix your __init__ first.
        """
        from models.language_model import LanguageModel
        model = LanguageModel.from_pretrained(cfg.lm)
        assert model is not None

    def test_parameter_count(self, cfg):
        """SmolLM2-360M-Instruct has ~362 M parameters."""
        from models.language_model import LanguageModel
        model = LanguageModel.from_pretrained(cfg.lm)
        n = sum(p.numel() for p in model.parameters())
        assert 300_000_000 < n < 420_000_000, f"Unexpected param count: {n:,}"

    def test_forward_shape(self, cfg, device):
        """With real weights, forward must return hidden states [1, T, 960]."""
        from models.language_model import LanguageModel
        model = LanguageModel.from_pretrained(cfg.lm).to(device).eval()
        x = torch.randn(1, 32, cfg.lm.hidden_dim, device=device)
        with torch.no_grad():
            hidden, _ = model(x)
        assert hidden.shape == (1, 32, cfg.lm.hidden_dim)

    def test_kv_cache_consistency_with_pretrained(self, cfg, device):
        """With real weights loaded, prefill+decode must match full forward."""
        from models.language_model import LanguageModel
        torch.manual_seed(0)
        model = LanguageModel.from_pretrained(cfg.lm).to(device).eval()
        T = 16
        x = torch.randn(1, T, cfg.lm.hidden_dim, device=device)
        with torch.no_grad():
            hidden_full, _ = model(x, kv_cache=None, start_pos=0)
            logits_full = model.head(hidden_full)
            hidden_prefix, kv = model(x[:, :-1], kv_cache=None, start_pos=0)
            hidden_last, _ = model(x[:, -1:], kv_cache=kv, start_pos=T - 1)
            logits_last = model.head(hidden_last)
        torch.testing.assert_close(
            logits_full[:, -1:], logits_last, atol=1e-3, rtol=1e-3
        )


class TestVLMPretrainedForward:
    def test_vlm_forward_with_pretrained_weights(self, cfg, device):
        """Full VLM forward pass with pretrained ViT + LM weights.

        Verifies that all components wire together correctly end-to-end:
        image → ViT → projector → LM → logits.  If loss is NaN, check your
        cross-entropy masking (image token positions use ignore_index=-100).
        """
        from models.vision_language_model import VisionLanguageModel
        model = VisionLanguageModel(cfg, load_backbone=True).to(device).eval()
        tokenizer = model.tokenizer

        T = 128
        n_img = cfg.projector.image_token_length
        image_string = tokenizer.image_token * n_img
        prompt = image_string + "What is in this image?"
        input_ids = torch.tensor(
            tokenizer.encode(prompt, add_special_tokens=False)[:T]
        ).unsqueeze(0).to(device)
        if input_ids.size(1) < T:
            pad = torch.full(
                (1, T - input_ids.size(1)),
                tokenizer.pad_token_id,
                device=device,
            )
            input_ids = torch.cat([input_ids, pad], dim=1)

        pixel_values = torch.randn(1, 3, 512, 512, device=device)
        targets = input_ids.clone()
        targets[:, :n_img] = -100

        with torch.no_grad():
            logits, loss = model(input_ids, pixel_values, targets=targets)

        assert torch.isfinite(loss), f"Loss is not finite: {loss}"
        assert logits.shape[-1] == cfg.lm.vocab_size
