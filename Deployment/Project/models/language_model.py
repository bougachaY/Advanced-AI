"""Causal Language Model (SmolLM2-360M) — student skeleton.

Architecture overview
─────────────────────
  Input embeddings  [B, T, 960]   (provided by the VLM wrapper)
      │
  32 × LMBlock      RMSNorm → LMAttention (GQA + RoPE + KV cache) → residual
                    RMSNorm → LMMLP (SiLU gate)                    → residual
      │  [B, T, 960]
  RMSNorm
      │
  Output  [B, T, 960]   (caller applies self.head for logits)

Key numbers (from LMConfig):
  hidden_dim  = 960
  inter_dim   = 2560
  n_heads     = 15   (query heads)   head_dim = 960/15 = 64
  n_kv_heads  = 5    (KV heads)      n_kv_groups = 15/5 = 3
  n_blocks    = 32
  vocab_size  = 49153

Pretrained weights: LanguageModel.from_pretrained(cfg: LMConfig)  ← PROVIDED.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Normalises WITHOUT subtracting the mean:
        out = x * rsqrt( mean(x²) + eps ) * weight
    """
    def __init__(self, cfg):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(cfg.hidden_dim))    # learnable scale of shape [hidden_dim], initialized to
        #                      # all ones (use nn.Parameter)
        self.rms_eps = cfg.rms_eps

        # raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args/Returns: [B, T, hidden_dim]

        Hint: torch.rsqrt(t) computes 1/sqrt(t) element-wise.
              Take the mean over the last dimension (keepdim=True).
        """
        # TODO: Compute the inverse RMS of x along the last dimension
        #       (keepdim=True), then scale x by it and the learned weight.
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.rms_eps) * self.weight


# ─────────────────────────────────────────────────────────────────────────────
class RotaryEmbedding(nn.Module):
    """Rotary Positional Embeddings (RoPE).

    Instead of adding a positional vector, RoPE *rotates* the query and key
    vectors by an angle proportional to the token's position.  Relative
    distance is preserved in the dot-product, enabling length generalisation.

    How it works
    ────────────
    1.  Assign a frequency to each dimension pair (2i, 2i+1):
            inv_freq[i] = 1 / (base ^ (2i / dim))   i = 0…dim/2-1
        These decay geometrically so early dims rotate fast and later dims
        rotate slowly.

    2.  For position t, compute:
            freqs[t, i] = t × inv_freq[i]            shape [T, dim/2]

    3.  Duplicate along the feature axis:
            emb = cat([freqs, freqs], dim=-1)         shape [T, dim]
        This interleaving is required by rotate_half() below.

    4.  Return cos(emb) and sin(emb) — passed to apply_rotary_pos_embd()
        which rotates Q and K inside each attention layer.

    Key numbers (from LMConfig):
        dim  = hidden_dim / n_heads = 960 / 15 = 64
        base = re_base = 100_000
        inv_freq shape: [dim/2] = [32]
        cos/sin output shape: [B, T, dim] = [B, T, 64]
    """
    def __init__(self, cfg):
        """PROVIDED — precomputes inv_freq and stores config scalars."""
        super().__init__()
        assert cfg.hidden_dim % cfg.n_heads == 0
        self.dim = cfg.hidden_dim // cfg.n_heads  # head_dim = 64
        self.re_base = cfg.re_base                # 100_000
        self.max_position_embeddings = cfg.max_position_embeddings
        self.attn_scaling = cfg.attn_scaling      # 1.0

        # inv_freq[i] = 1 / (re_base ^ (2i / dim)),  shape [dim/2]
        inv_freq = 1.0 / (
            self.re_base ** (
                torch.arange(0, self.dim, 2).float() / self.dim
            )
        )
        self.register_buffer("inv_freq", inv_freq)

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor):
        """
        Args:
            position_ids: [B, T]  integer positions, e.g. [[0,1,…,T-1]]
        Returns:
            cos: [B, T, dim]
            sin: [B, T, dim]

        TODO 1 — Scale down the stored frequencies proportionally if the
                 sequence is longer than the precomputed maximum length.

        TODO 2 — Flatten position_ids to a 1-D vector, then use unsqueeze
                 to broadcast it against inv_freq so you get one frequency
                 per (position, dimension) pair.
                 Reshape back to [B, T, dim/2].

        TODO 3 — Concatenate freqs with itself along the last dimension to
                 cover both halves of the head dimension.
                 Result: [B, T, dim]

        TODO 4 — Compute cosine and sine of the embeddings, scale each by
                 attn_scaling, and return both.
        """
        # Step 1: Scale down frequencies if needed
        if position_ids.max() > self.max_position_embeddings:
            scaling_factor = position_ids.max() / self.max_position_embeddings
            inv_freq = self.inv_freq / scaling_factor
        else:
            inv_freq = self.inv_freq
        
        # Step 2: Compute frequencies for each position
        positions = position_ids.flatten()
        freqs = positions.unsqueeze(-1) * inv_freq.unsqueeze(0)  # [B*T, dim/2]
        freqs = freqs.view(position_ids.size(0), position_ids.size(1), -1)  # [B, T, dim/2]
        
        # Step 3: Duplicate frequencies along the last dimension
        freqs = torch.cat([freqs, freqs], dim=-1)  # [B, T, dim]
        
        # Step 4: Compute and scale cosine and sine
        cos = torch.cos(freqs) * self.attn_scaling
        sin = torch.sin(freqs) * self.attn_scaling
        return cos, sin
        
        


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_embd(q, k, cos, sin, unsqueeze_dim=1):
    """Apply RoPE to query and key tensors.  PROVIDED — do not modify.

    cos/sin: [B, T, dim]  →  unsqueezed to [B, 1, T, dim] for broadcasting.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (q * cos) + (rotate_half(q) * sin), \
           (k * cos) + (rotate_half(k) * sin)


# ─────────────────────────────────────────────────────────────────────────────
class LMAttention(nn.Module):
    """Grouped-Query Attention (GQA) with KV caching and RoPE.

    GQA uses fewer KV heads than query heads to save memory.
    n_heads=15, n_kv_heads=5  →  n_kv_groups=3 (each KV head shared by 3 Q).

    KV cache: during generation, past K/V are stored and concatenated with
    the new token's K/V so only one token is processed per decode step.

    Key attribute names (DO NOT rename — from_pretrained depends on them):
        q_proj   nn.Linear(960, 960,          bias=False)
        k_proj   nn.Linear(960, 5×64=320,     bias=False)
        v_proj   nn.Linear(960, 5×64=320,     bias=False)
        out_proj nn.Linear(960, 960,          bias=False)
    """
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads       # 15
        self.n_kv_heads = cfg.n_kv_heads  # 5
        self.hidden_dim = cfg.hidden_dim  # 960
        self.dropout = cfg.dropout

        assert self.n_heads % self.n_kv_heads == 0

        self.n_kv_groups = self.n_heads // self.n_kv_heads   # query heads per KV head (n_heads // n_kv_heads)
        self.head_dim = self.hidden_dim // self.n_heads      # embedding dimension per attention head
        self.q_proj = nn.Linear(self.hidden_dim, self.n_heads * self.head_dim, bias=False)        # Linear: hidden_dim → n_heads × head_dim (no bias)
        self.k_proj = nn.Linear(self.hidden_dim, self.n_kv_heads * self.head_dim, bias=False)        # Linear: hidden_dim → n_kv_heads × head_dim (no bias)
        self.v_proj = nn.Linear(self.hidden_dim, self.n_kv_heads * self.head_dim, bias=False)        # Linear: same output shape as k_proj (no bias)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)      # Linear: hidden_dim → hidden_dim (no bias)
        self.attn_dropout = nn.Dropout(self.dropout)  # Dropout on attention weights
        self.resid_dropout = nn.Dropout(self.dropout) # Dropout on the output
        self.sdpa = hasattr(F, 'scaled_dot_product_attention')  # check if PyTorch has the built-in function

        #raise NotImplementedError

    def forward(self, x, cos, sin, attention_mask=None, block_kv_cache=None):
        """
        Args:
            x:              [B, T_curr, 960]
            cos, sin:       [B, T_curr, 64]
            attention_mask: [B, T_total] 1=attend, 0=pad  (optional)
            block_kv_cache: dict {'key': …, 'value': …} or None (prefill)
        Returns:
            output:         [B, T_curr, 960]
            block_kv_cache: updated cache dict

        TODO 1 — Project x with q_proj, k_proj, v_proj separately. Use view
                 to split the last dimension into (n_heads, head_dim) or
                 (n_kv_heads, head_dim), then transpose to put heads first.
                 q:   [B, n_heads,    T_curr, head_dim]
                 k,v: [B, n_kv_heads, T_curr, head_dim]

        TODO 2 — Apply the rotary positional embeddings to queries and keys.

        TODO 3 — On the first (prefill) call initialize the cache with the
                 current K and V. On subsequent (decode) calls, concatenate
                 the cached K and V before the current ones along the
                 sequence dimension (dim=2).

        TODO 4 — Use repeat_interleave along the head dimension (dim=1) to
                 replicate each KV head n_kv_groups times, so every query
                 head has a matching key and value.
                 k_exp, v_exp: [B, n_heads, T_kv, head_dim]

        TODO 5 — Build an additive mask: padding positions should become the
                 most negative float in q's dtype (use torch.finfo to get
                 it). Use two unsqueezes to broadcast over the batch and
                 head dimensions.

        TODO 6 — Call scaled_dot_product_attention with the additive mask.
                 Set is_causal=True only when T_curr equals T_kv and
                 T_curr > 1 (prefill, not single-token decoding).

        TODO 7 — Transpose the head and sequence dimensions back, call
                 contiguous to fix the memory layout, then collapse heads
                 into the channel dimension with view. Apply out_proj and
                 resid_dropout, and return together with the updated cache.
        """
        # -------- Step 1
        # Project to Q, K, V
        q = self.q_proj(x)   # [B, T_curr, 960]
        k = self.k_proj(x)   # [B, T_curr, 320]
        v = self.v_proj(x)   # [B, T_curr, 320]

        B, T_curr, _ = x.shape

        # Split into heads
        q = q.view(B, T_curr, self.n_heads, self.head_dim)
        k = k.view(B, T_curr, self.n_kv_heads, self.head_dim)
        v = v.view(B, T_curr, self.n_kv_heads, self.head_dim)

        # Move heads before sequence dimension
        q = q.transpose(1, 2)  # [B, n_heads,    T_curr, head_dim]
        k = k.transpose(1, 2)  # [B, n_kv_heads, T_curr, head_dim]
        v = v.transpose(1, 2)  # [B, n_kv_heads, T_curr, head_dim]

        # --------- Step 2 
        q, k = apply_rotary_pos_embd(q, k, cos, sin)

        # --------- Step 3
        # Initialize or update KV cache
        if block_kv_cache is None:
            block_kv_cache = {
                "key": k,
                "value": v,
            }
        else:
            k = torch.cat([block_kv_cache["key"], k], dim=2)
            v = torch.cat([block_kv_cache["value"], v], dim=2)

            block_kv_cache = {
                "key": k,
                "value": v,
            }

        # ------- Step 4
        k_exp = torch.repeat_interleave(k, self.n_kv_groups, dim=1)
        v_exp = torch.repeat_interleave(v, self.n_kv_groups, dim=1)

        # ------- Step 5
        attn_mask = None
        if attention_mask is not None:
            # attention_mask: [B, T_kv] (1 = keep, 0 = pad)
            attn_mask = (1.0 - attention_mask).to(dtype=q.dtype) * torch.finfo(q.dtype).min
            attn_mask = attn_mask.view(q.size(0), 1, 1, -1)  # [B, 1, 1, T_kv]

        # -------- Step 6
        is_causal = (q.size(2) == k_exp.size(2)) and (q.size(2) > 1)

        attn_output = F.scaled_dot_product_attention(
            q,
            k_exp,
            v_exp,
            attn_mask=attn_mask if not is_causal else None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        #--------- Step 7
        B, n_heads, T_curr, head_dim = attn_output.shape

        # bring sequence dimension second
        attn_output = attn_output.transpose(1, 2).contiguous()
        # [B, T_curr, n_heads, head_dim]

        # merge heads
        attn_output = attn_output.view(B, T_curr, self.hidden_dim)
        # [B, T_curr, 960]

        # final projection
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, block_kv_cache

        #raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
class LMMLP(nn.Module):
    """Gated MLP (SwiGLU-style).

    gate_proj: hidden_dim → inter_dim   (960 → 2560)
    up_proj:   hidden_dim → inter_dim   (960 → 2560)
    down_proj: inter_dim  → hidden_dim  (2560 → 960)

    Output = down_proj( silu(gate_proj(x)) × up_proj(x) )
    """
    def __init__(self, cfg):
        super().__init__()

        self.gate_proj = nn.Linear(cfg.hidden_dim, cfg.inter_dim, bias=False)   # Linear: hidden_dim → inter_dim, no bias (gate branch)
        self.up_proj = nn.Linear(cfg.hidden_dim, cfg.inter_dim, bias=False)     # Linear: hidden_dim → inter_dim, no bias (value branch)
        self.down_proj = nn.Linear(cfg.inter_dim, cfg.hidden_dim, bias=False)   # Linear: inter_dim → hidden_dim, no bias

        #raise NotImplementedError

    def forward(self, x):
        """
        Args/Returns: [B, T, hidden_dim]

        TODO: Apply silu to the gate projection, multiply element-wise
              with the up projection, then project back down.
        """
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        x = self.down_proj(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
class LMBlock(nn.Module):
    """Pre-norm residual block.  Attention returns (output, cache) — unpack!"""
    def __init__(self, cfg):
        super().__init__()

        self.norm1 = RMSNorm(cfg)  # RMSNorm applied before attention
        self.attn = LMAttention(cfg)    # the LMAttention sub-layer
        self.norm2 = RMSNorm(cfg)  # RMSNorm applied before the MLP
        self.mlp = LMMLP(cfg)     # the LMMLP sub-layer

        #raise NotImplementedError

    def forward(self, x, cos, sin, attention_mask=None, block_kv_cache=None):
        """
        Args:
            x:              [B, T, hidden_dim]
            cos, sin:       [B, T, head_dim]
            attention_mask: [B, T_total] or None
            block_kv_cache: dict or None
        Returns:
            x:              [B, T, hidden_dim]
            block_kv_cache: updated dict

        Follow the same pre-norm residual pattern as ViTBlock, but
        attention returns a (output, cache) tuple — unpack it.
        """
        # TODO: Two pre-norm residual sub-layers (attention, then MLP).
        attn_out, updated_cache = self.attn(self.norm1(x), cos, sin, attention_mask, block_kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, updated_cache


# ─────────────────────────────────────────────────────────────────────────────
class LanguageModel(nn.Module):
    """Full causal language model.

    forward() receives embeddings (not token ids) when called from the VLM,
    because image placeholder tokens have already been replaced with visual
    embeddings by VisionLanguageModel._replace_img_tokens_with_embd().

    self.head is applied externally by the VLM only when computing loss.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tie_weights = cfg.tie_weights

        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)  # Embedding table: vocab_size → hidden_dim
        self.rotary_embd = RotaryEmbedding(cfg)       # the RotaryEmbedding module
        self.blocks = nn.ModuleList([LMBlock(cfg) for _ in range(cfg.n_blocks)])  # ModuleList of n_blocks LMBlock layers
        self.norm = RMSNorm(cfg)             # final RMSNorm
        self.head = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)             # Linear: hidden_dim → vocab_size (no bias)

        if self.tie_weights:
            self.head.weight = self.token_embedding.weight

        #raise NotImplementedError

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask=None,
        kv_cache=None,
        start_pos: int = 0,
    ):
        """Run the language model on a sequence of embeddings.

        Args:
            x:              [B, T, hidden_dim]  — embeddings (NOT token ids)
            attention_mask: [B, T_total] 1=attend, 0=pad  (or None)
            kv_cache:       list of per-block dicts, or None (prefill)
            start_pos:      position of first token in x
                            (0 during prefill; total_seq_len-1 during decode)
        Returns:
            hidden:   [B, T, hidden_dim]
            kv_cache: updated list of per-block dicts

        TODO 1: Unpack batch size and current sequence length from x.

        TODO 2: Use arange to build integer position indices from start_pos
                to start_pos+T_curr, expand to [B, T_curr], then compute
                the rotary cos/sin embeddings.

        TODO 3: Initialize the KV cache as a list of None values (one entry
                per block) if no cache was passed in.

        TODO 4: Loop over the blocks, passing updated hidden states and
                writing each block's returned cache back into the list.

        TODO 5: Apply the final RMS normalization.

        TODO 6: Return the hidden states and the updated KV cache.
        """
        # raise NotImplementedError
        # TODO 1
        B, T_curr, _ = x.size()
        
        # TODO 2
        vectorized_positions = torch.arange(start_pos, start_pos + T_curr, device=x.device).unsqueeze(0).expand(B, T_curr)
        cos, sin = self.rotary_embd(vectorized_positions)
        
        # TODO 3 : On ne crée la liste vide que si c'est le tout premier passage
        if kv_cache is None:
            kv_cache = [None] * len(self.blocks)
            
        # TODO 4 : On boucle et on met à jour la liste existante sur place
        for i, block in enumerate(self.blocks):
            x, kv_cache[i] = block(x, cos, sin, attention_mask, kv_cache[i])
            
        # TODO 5 & 6
        x = self.norm(x)
        return x, kv_cache
        
    # ── Provided: greedy generation for the standalone LM ────────────────────
    @torch.inference_mode()
    def generate(self, inputs: torch.Tensor, max_new_tokens: int = 20):
        """Greedy autoregressive generation from token ids.  PROVIDED."""
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
        generated = inputs.clone()
        out, kv = self.forward(
            self.token_embedding(generated), kv_cache=None, start_pos=0
        )
        last = self.head(out[:, -1, :])
        for i in range(max_new_tokens):
            next_tok = torch.argmax(last, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_tok], dim=1)
            pos = generated.size(1) - 1
            if i == max_new_tokens - 1:
                break
            out, kv = self.forward(
                self.token_embedding(next_tok), kv_cache=kv, start_pos=pos
            )
            last = self.head(out[:, -1, :])
        return generated

    # ── Provided: loads pretrained SmolLM2-360M weights ──────────────────────
    @classmethod
    def from_pretrained(cls, cfg):
        from transformers import AutoConfig
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError
        import safetensors
        import json
        import torch.nn.init as init

        hf = AutoConfig.from_pretrained(cfg.model_type)
        original_vocab = hf.vocab_size

        cfg.hidden_dim = hf.hidden_size
        cfg.inter_dim = hf.intermediate_size
        cfg.rms_eps = hf.rms_norm_eps
        #cfg.re_base = hf.rope_parameters.get('rope_theta', 100000)
        cfg.re_base = getattr(hf, 'rope_theta',100000)
        cfg.max_position_embeddings = hf.max_position_embeddings
        cfg.n_heads = hf.num_attention_heads
        cfg.n_kv_heads = hf.num_key_value_heads
        cfg.dropout = hf.attention_dropout
        cfg.n_blocks = hf.num_hidden_layers

        if cfg.vocab_size < original_vocab:
            raise ValueError(
                f"cfg.vocab_size ({cfg.vocab_size}) < pretrained "
                f"({original_vocab})"
            )

        model = cls(cfg)

        try:
            idx = hf_hub_download(
                repo_id=cfg.model_type,
                filename="model.safetensors.index.json",
            )
            with open(idx) as f:
                index = json.load(f)
            fnames = sorted(set(index['weight_map'].values()))
            sf_files = [
                hf_hub_download(repo_id=cfg.model_type, filename=fn)
                for fn in fnames
            ]
        except EntryNotFoundError:
            sf_files = [
                hf_hub_download(
                    repo_id=cfg.model_type, filename="model.safetensors"
                )
            ]

        sd = model.state_dict()
        mapping = {
            'model.embed_tokens.weight': 'token_embedding.weight',
            'model.norm.weight':         'norm.weight',
        }
        for i in range(cfg.n_blocks):
            lp = f'model.layers.{i}.'
            bp = f'blocks.{i}.'
            mapping.update({
                f'{lp}self_attn.q_proj.weight': f'{bp}attn.q_proj.weight',
                f'{lp}self_attn.k_proj.weight': f'{bp}attn.k_proj.weight',
                f'{lp}self_attn.v_proj.weight': f'{bp}attn.v_proj.weight',
                f'{lp}self_attn.o_proj.weight': f'{bp}attn.out_proj.weight',
                f'{lp}mlp.gate_proj.weight': f'{bp}mlp.gate_proj.weight',
                f'{lp}mlp.up_proj.weight': f'{bp}mlp.up_proj.weight',
                f'{lp}mlp.down_proj.weight': f'{bp}mlp.down_proj.weight',
                f'{lp}input_layernorm.weight': f'{bp}norm1.weight',
                f'{lp}post_attention_layernorm.weight': f'{bp}norm2.weight',
            })

        loaded = set()
        for sf in sf_files:
            with safetensors.safe_open(sf, framework="pt", device="cpu") as f:
                for hf_key, our_key in mapping.items():
                    if our_key in loaded or hf_key not in f.keys():
                        continue
                    if our_key not in sd:
                        continue
                    t = f.get_tensor(hf_key)
                    if (hf_key == 'model.embed_tokens.weight'
                            and t.shape[0] != sd[our_key].shape[0]):
                        sd[our_key][:t.shape[0]].copy_(t)
                        init.normal_(
                            sd[our_key][t.shape[0]:], mean=0.0, std=0.02
                        )
                        print(
                            f"Extended embeddings: "
                            f"{t.shape[0]} → {sd[our_key].shape[0]}"
                        )
                    elif t.shape == sd[our_key].shape:
                        sd[our_key].copy_(t)
                    loaded.add(our_key)

        model.load_state_dict(sd)
        if cfg.tie_weights:
            model.head.weight = model.token_embedding.weight

        n = sum(p.numel() for p in model.parameters())
        print(f"Loaded {cfg.model_type} — {n:,} parameters")
        return model
