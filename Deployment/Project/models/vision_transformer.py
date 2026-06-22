"""Vision Transformer (ViT) encoder — student skeleton.

Architecture overview
─────────────────────
  Input image  [B, 3, 512, 512]
      │
  ViTPatchEmbeddings   Conv2d(16×16) → flatten → + positional embedding
      │  [B, 1024, 768]
  12 × ViTBlock        LayerNorm → ViTAttention → residual
                       LayerNorm → ViTMLP       → residual
      │  [B, 1024, 768]
  LayerNorm
      │
  Output  [B, 1024, 768]   (one 768-dim token per 16×16 patch)

Key numbers (from ViTConfig):
  img_size   = 512   → (512/16)² = 1024 patches
  hidden_dim = 768
  inter_dim  = 3072  (= 4 × 768)
  n_heads    = 12    → head_dim = 768/12 = 64
  n_blocks   = 12

Pretrained weights: ViT.from_pretrained(cfg: ViTConfig)  ← PROVIDED.
That function expects:
  • a single qkv_proj  nn.Linear(768, 3×768)  per block
  • the attribute names below to match exactly
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
class ViTPatchEmbeddings(nn.Module):
    """Convert a batch of images into patch embeddings + positional encoding.

    Step 1 — patch extraction:
        Conv2d(kernel=patch_size, stride=patch_size) extracts patches.
        Output: [B, hidden_dim, img_size/p, img_size/p]

    Step 2 — flatten + transpose:
        Flatten spatial dims → [B, hidden_dim, num_patches]
        Transpose            → [B, num_patches, hidden_dim]

    Step 3 — add position embeddings:
        Broadcast-add a learnable [1, num_patches, hidden_dim] tensor.
    """
    def __init__(self, cfg):
        super().__init__()
        self.img_size = cfg.img_size    # 512
        self.patch_size = cfg.patch_size  # 16
        self.hidden_dim = cfg.hidden_dim  # 768

        self.num_patches = (self.img_size // self.patch_size)**2          # total patches = (img_size // patch_size)²
        self.conv = nn.Conv2d(in_channels=3, out_channels=self.hidden_dim, kernel_size=self.patch_size, stride=self.patch_size) # Conv2d patch extractor: kernel_size and stride
        #                                 # both equal to patch_size, in_channels=3,
        #                                 # out_channels=hidden_dim, padding="valid"
        self.position_embedding = nn.Parameter(torch.zeros(1, self.num_patches, self.hidden_dim)) # learnable nn.Parameter of shape
        #                                 # [1, num_patches, hidden_dim]

        #raise NotImplementedError

    def forward(self, x):
        """
        Args:
            x: [B, 3, img_size, img_size]
        Returns:
            [B, num_patches, hidden_dim]  =  [B, 1024, 768]
        """
        x = self.conv(x) # TODO 1: Apply the convolutional patch extractor.
        #         Output: [B, hidden_dim, 32, 32]
    

        x = x.flatten(2)# TODO 2: Flatten the two spatial dimensions into one.
        #         Output: [B, hidden_dim, 1024]

        x = x.transpose(1, 2) # TODO 3: Swap the patch and channel dimensions.
        #         Output: [B, 1024, hidden_dim]

        x = x + self.position_embedding # TODO 4: Add the learned position embeddings to each patch token.
        #         Output: [B, 1024, hidden_dim]

        return x #raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
class ViTAttention(nn.Module):
    """Multi-head self-attention for the ViT encoder (bidirectional).

    Uses a single combined qkv_proj so that SigLIP2 weights can be loaded
    directly (from_pretrained concatenates the separate Q, K, V matrices).

    Key attributes (DO NOT rename — weight loading depends on them):
        qkv_proj : nn.Linear(hidden_dim, 3 * hidden_dim, bias=True)
        out_proj  : nn.Linear(hidden_dim, hidden_dim, bias=True)
    """
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads       # 12
        self.hidden_dim = cfg.hidden_dim  # 768
        assert self.hidden_dim % self.n_heads == 0
        self.dropout = cfg.dropout

        self.head_dim = self.hidden_dim // self.n_heads         # embedding dimension per attention head
        self.qkv_proj = nn.Linear(self.hidden_dim, 3*self.hidden_dim, bias=True)          # single Linear: hidden_dim → 3 × hidden_dim
        #                              # (Q, K, V packed together; bias=True)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)         # Linear: hidden_dim → hidden_dim (bias=True)
        self.attn_dropout = nn.Dropout(self.dropout)      # Dropout on attention weights
        self.resid_dropout = nn.Dropout(self.dropout)     # Dropout on the output projection
        self.sdpa = hasattr(F, "scaled_dot_product_attention")           # True if F.scaled_dot_product_attention is available

        #raise NotImplementedError

    def forward(self, x):
        """
        Args:
            x: [B, T, C]  where T=1024, C=768
        Returns:
            [B, T, C]
        """
        B, T, C = x.size()

        qkv = self.qkv_proj(x) # TODO 1: Project x to queries, keys, and values in one shot with
        q, k, v = qkv.chunk(3, dim=-1) #         qkv_proj, then split into three equal chunks along the
        #         last dimension.
        #         q, k, v each → [B, T, C]

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)# TODO 2: Use view to introduce the head dimension, then transpose
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)#         so heads come before the sequence.
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)#         Each of q, k, v → [B, n_heads, T, head_dim]

        # TODO 3: Attend.
        if self.sdpa:
            y = F.scaled_dot_product_attention(q,k,v, dropout_p=self.dropout if self.training else 0.0, is_causal=False)  # Use scaled_dot_product_attention; set is_causal=False
        #       because the vision encoder attends to all patches in
        #       both directions.
        else: #(fallback)
            scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attn = F.softmax(scores, dim=-1)
            attn = self.attn_dropout(attn)
            y = attn @ v

        # TODO 4: Transpose the head and sequence dimensions back, call
        y = y.transpose(1, 2)
        y = y.contiguous().view(B, T, C)    #contiguous, then collapse the head dimension into the channel dimension with view.
        y = self.out_proj(y)       #         Apply out_proj then resid_dropout.
        y = self.resid_dropout(y)#         Output: [B, T, C]
        return y

        #raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
class ViTMLP(nn.Module):
    """Two-layer MLP with GELU used inside each ViT block.

    fc1: hidden_dim → inter_dim   (768 → 3072)
    fc2: inter_dim  → hidden_dim  (3072 → 768)
    """
    def __init__(self, cfg):
        super().__init__()

        self.hidden_dim = cfg.hidden_dim
        self.inter_dim = cfg.inter_dim

        self.activation_fn = nn.GELU()    # GELU activation (approximate='tanh')
        self.fc1 = nn.Linear(self.hidden_dim, self.inter_dim)              # Linear: hidden_dim → inter_dim
        self.fc2 = nn.Linear(self.inter_dim, self.hidden_dim)              # Linear: inter_dim → hidden_dim
        self.dropout = nn.Dropout(cfg.dropout)          # Dropout

        #raise NotImplementedError

    def forward(self, x):
        """
        Args/Returns: [B, T, hidden_dim]

        TODO: Pass through fc1, apply the GELU activation, then fc2,
              then dropout.
        """
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x
        #raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
class ViTBlock(nn.Module):
    """Pre-norm residual block: attention sub-layer then MLP sub-layer."""
    def __init__(self, cfg):
        super().__init__()

        self.ln1 = nn.LayerNorm(cfg.hidden_dim, eps=cfg.ln_eps)    # LayerNorm applied before attention
        self.attn = ViTAttention(cfg)  # the ViTAttention sub-layer
        self.ln2 = nn.LayerNorm(cfg.hidden_dim, eps=cfg.ln_eps) # LayerNorm applied before the MLP
        self.mlp = ViTMLP(cfg)  # the ViTMLP sub-layer

    def forward(self, x):
        """
        Args/Returns: [B, T, hidden_dim]

        Pre-norm residual pattern:
            x = x + attn(ln1(x))
            x = x + mlp(ln2(x))
        """
        x = x + self.attn(self.ln1(x)) # TODO: Apply attention with pre-norm and residual, then the MLP
        x = x + self.mlp(self.ln2(x))  # with pre-norm and residual (pattern shown in the docstring).
        
        return x


# ─────────────────────────────────────────────────────────────────────────────
class ViT(nn.Module):
    """Full Vision Transformer encoder."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.cls_flag = cfg.cls_flag

        self.patch_embedding = ViTPatchEmbeddings(cfg)  # the ViTPatchEmbeddings sub-module
        self.dropout = nn.Dropout(cfg.dropout)          # Dropout
        self.blocks = nn.ModuleList([ViTBlock(cfg) for _ in range(cfg.n_blocks)])          # ModuleList of n_blocks ViTBlock layers
        self.layer_norm = nn.LayerNorm(cfg.hidden_dim, eps=cfg.ln_eps)      # final LayerNorm (hidden_dim, eps=cfg.ln_eps)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Args:
            x: [B, 3, 512, 512]
        Returns:
            [B, 1024, 768]

        TODO 1: Compute the patch embeddings.  → [B, 1024, 768]
        TODO 2: Apply dropout.
        TODO 3: Pass the sequence through each transformer block in order.
        TODO 4: Apply the final layer normalization.
        TODO 5: Return all patch tokens (cls_flag is False for this encoder).
        """
        
        x = self.patch_embedding(x)
        
        x = self.dropout(x)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.layer_norm(x)
        
        return x

    # ── Provided: loads pretrained SigLIP2 weights ────────────────────────────
    @classmethod
    def from_pretrained(cls, cfg):
        from transformers import SiglipVisionConfig
        from huggingface_hub import hf_hub_download
        import safetensors

        hf = SiglipVisionConfig.from_pretrained(cfg.model_type)
        cfg.dropout = hf.attention_dropout
        cfg.hidden_dim = hf.hidden_size
        cfg.img_size = hf.image_size
        cfg.inter_dim = hf.intermediate_size
        cfg.ln_eps = hf.layer_norm_eps
        cfg.n_heads = hf.num_attention_heads
        cfg.n_blocks = hf.num_hidden_layers
        cfg.patch_size = hf.patch_size
        model = cls(cfg)

        sf = hf_hub_download(repo_id=cfg.model_type, filename="model.safetensors")
        sd = model.state_dict()

        mapping = {
            'vision_model.embeddings.patch_embedding.weight':
                'patch_embedding.conv.weight',
            'vision_model.embeddings.patch_embedding.bias':
                'patch_embedding.conv.bias',
            'vision_model.embeddings.position_embedding.weight':
                'patch_embedding.position_embedding',
            'vision_model.post_layernorm.weight': 'layer_norm.weight',
            'vision_model.post_layernorm.bias':   'layer_norm.bias',
        }
        for i in range(cfg.n_blocks):
            p = f'vision_model.encoder.layers.{i}'
            b = f'blocks.{i}'
            mapping.update({
                f'{p}.layer_norm1.weight': f'{b}.ln1.weight',
                f'{p}.layer_norm1.bias':   f'{b}.ln1.bias',
                f'{p}.layer_norm2.weight': f'{b}.ln2.weight',
                f'{p}.layer_norm2.bias':   f'{b}.ln2.bias',
                f'{p}.mlp.fc1.weight':     f'{b}.mlp.fc1.weight',
                f'{p}.mlp.fc1.bias':       f'{b}.mlp.fc1.bias',
                f'{p}.mlp.fc2.weight':     f'{b}.mlp.fc2.weight',
                f'{p}.mlp.fc2.bias':       f'{b}.mlp.fc2.bias',
                f'{p}.self_attn.out_proj.weight': f'{b}.attn.out_proj.weight',
                f'{p}.self_attn.out_proj.bias':   f'{b}.attn.out_proj.bias',
            })

        with safetensors.safe_open(sf, framework="pt", device="cpu") as f:
            for hf_key, our_key in mapping.items():
                if hf_key in f.keys() and our_key in sd:
                    t = f.get_tensor(hf_key)
                    if t.shape == sd[our_key].shape:
                        sd[our_key].copy_(t)
                    elif 'position_embedding' in hf_key:
                        sd[our_key].copy_(t.unsqueeze(0))

            for i in range(cfg.n_blocks):
                p = f'vision_model.encoder.layers.{i}.self_attn'
                q = f.get_tensor(f'{p}.q_proj.weight')
                k = f.get_tensor(f'{p}.k_proj.weight')
                v = f.get_tensor(f'{p}.v_proj.weight')
                sd[f'blocks.{i}.attn.qkv_proj.weight'].copy_(
                    torch.cat([q, k, v], dim=0)
                )
                qb = f.get_tensor(f'{p}.q_proj.bias')
                kb = f.get_tensor(f'{p}.k_proj.bias')
                vb = f.get_tensor(f'{p}.v_proj.bias')
                sd[f'blocks.{i}.attn.qkv_proj.bias'].copy_(
                    torch.cat([qb, kb, vb], dim=0)
                )

        model.load_state_dict(sd)
        n = sum(p.numel() for p in model.parameters())
        print(f"Loaded {cfg.model_type} — {n:,} parameters")
        return model
