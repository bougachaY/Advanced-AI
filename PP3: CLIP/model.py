"""
CLIP (Contrastive Language-Image Pre-Training) model architecture.

References:
1) Original OpenAI CLIP implementation:
   https://github.com/openai/CLIP/blob/main/clip/model.py
2) Paper: "Learning Transferable Visual Models From Natural Language Supervision"
   https://arxiv.org/abs/2103.00020
"""

from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16.

    During mixed-precision training, activations may be fp16, but LayerNorm
    requires fp32 for numerical stability. This wrapper casts to fp32,
    applies the norm, then casts back.
    """

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
    

class ResidualAttentionBlock(nn.Module):
    """One transformer block: pre-norm multi-head self-attention + pre-norm MLP, both with residual connections.

    Architecture (pre-norm variant):
        x = x + Attention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))

    The MLP expands the hidden dimension by a factor of 4, applies QuickGELU,
    then projects back to d_model.

    Parameters
    ----------
    d_model : int
        Token embedding dimension.
    n_head : int
        Number of attention heads.
    attn_mask : torch.Tensor, optional
        Causal mask for autoregressive text encoding (None for vision).
    """

    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)

        c_fc   = # TODO: nn.Linear from d_model to d_model * 4 (the MLP expansion layer)
        c_proj = # TODO: nn.Linear from d_model * 4 back to d_model (the MLP projection layer)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc",   c_fc),
            ("gelu",   QuickGELU()),
            ("c_proj", c_proj)
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = # TODO: attention sub-layer with residual and layer norm
        x = # TODO: MLP sub-layer with residual and layer norm
        return x


class Transformer(nn.Module):
    """Stack of ResidualAttentionBlock layers.

    Parameters
    ----------
    width : int
        Token embedding dimension (d_model for each block).
    layers : int
        Number of transformer blocks.
    heads : int
        Number of attention heads per block.
    attn_mask : torch.Tensor, optional
        Causal mask forwarded to every block.
    """

    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    """Vision Transformer (ViT) image encoder.

    Splits an image into fixed-size patches, embeds each patch with a Conv2d
    (equivalent to a linear projection), prepends a learnable [CLS] token,
    adds positional embeddings, and feeds the sequence through a Transformer.
    The [CLS] token output is then projected to the shared embedding space.

    Parameters
    ----------
    input_resolution : int
        Expected input image size (square), e.g. 224.
    patch_size : int
        Size of each patch (square), e.g. 32 for ViT-B/32.
    width : int
        Patch embedding dimension.
    layers : int
        Number of transformer blocks.
    heads : int
        Number of attention heads.
    output_dim : int
        Dimension of the final projected embedding (shared with text encoder).
    """

    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        # Patch embedding: conv with kernel=stride=patch_size extracts non-overlapping patches
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [batch_size, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [batch_size, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [batch_size, grid ** 2, width]
        class_embedding = self.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = # TODO: prepend class_embedding as the first token using torch.cat
        #   concatenate along dim=1. Output shape: [batch_size, grid**2 + 1, width]
        x = # TODO: add self.positional_embedding to x (cast positional_embedding to x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # [batch_size, grid ** 2 + 1, width] -> [grid ** 2 + 1, batch_size, width] (required by nn.MultiheadAttention)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # [grid ** 2 + 1, batch_size, width] -> [batch_size, grid ** 2 + 1, width]

        x = # TODO: apply self.ln_post to the class token only (position 0: x[:, 0, :])
        if self.proj is not None:
            x = # TODO: project x to output_dim using self.proj (matrix multiply: x @ self.proj)
        return x


class CLIP(nn.Module):
    """CLIP: Contrastive Language-Image Pre-Training.

    Dual-encoder model that maps images and text into a shared embedding space.
    Training maximises cosine similarity between matched (image, text) pairs
    and minimises it for unmatched pairs (contrastive loss).

    Vision encoder: either a ModifiedResNet (when vision_layers is a tuple)
                    or a VisionTransformer (when vision_layers is an int).
    Text encoder:   a Transformer with a causal attention mask, whose output
                    at the [EOT] token position is projected to embed_dim.

    Parameters
    ----------
    embed_dim : int
        Shared embedding dimension for image and text features.
    image_resolution : int
        Input image resolution.
    vision_layers : Union[Tuple[int, int, int, int], int]
        If tuple → ModifiedResNet (ResNet layer counts). If int → VisionTransformer.
    vision_width : int
        Channel width of the vision encoder.
    vision_patch_size : int
        Patch size for ViT (ignored for ResNet).
    context_length : int
        Maximum text sequence length (77 for all CLIP models).
    vocab_size : int
        Tokenizer vocabulary size.
    transformer_width : int
        Embedding dimension of the text transformer.
    transformer_heads : int
        Number of attention heads in the text transformer.
    transformer_layers : int
        Number of transformer blocks in the text encoder.
    """

    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length

        vision_heads = vision_width // 64
        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim
        )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        """Encode a batch of images into the shared embedding space.

        Parameters
        ----------
        image : torch.Tensor
            Batch of images, shape [batch_size, 3, H, W].

        Returns
        -------
        torch.Tensor
            Image features, shape [batch_size, embed_dim].
        """
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        """Encode a batch of tokenised text sequences into the shared embedding space.

        The output is taken from the position of the [EOT] token (the highest
        token id in each sequence), then projected with self.text_projection.

        Parameters
        ----------
        text : torch.Tensor
            Token ids, shape [batch_size, context_length].

        Returns
        -------
        torch.Tensor
            Text features, shape [batch_size, embed_dim].
        """
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, context_length, transformer_width]

        x = # TODO: add self.positional_embedding to x (cast to self.dtype)
        x = x.permute(1, 0, 2)  # [batch_size, context_length, transformer_width] -> [context_length, batch_size, transformer_width]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # [context_length, batch_size, transformer_width] -> [batch_size, context_length, transformer_width]
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, context_length, transformer_width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = # TODO: index x at the EOT position for each sequence, then project with self.text_projection
        #   Hint: text.argmax(dim=-1) gives the position of the highest token id (= EOT) in each row.
        #   Use x[torch.arange(x.shape[0]), ...] to gather those positions, then matrix-multiply by
        #   self.text_projection.

        return x

    def forward(self, image, text):
        """Compute cosine similarity logits between all image-text pairs in the batch.

        Returns two logit matrices:
          - logits_per_image[i, j] = similarity of image i with text j
          - logits_per_text[i, j]  = similarity of text i with image j  (= logits_per_image.T)

        Parameters
        ----------
        image : torch.Tensor  [batch_size, 3, H, W]
        text  : torch.Tensor  [batch_size, context_length]

        Returns
        -------
        logits_per_image : torch.Tensor  [batch_size, batch_size]
        logits_per_text  : torch.Tensor  [batch_size, batch_size]
        """
        image_features = self.encode_image(image)
        text_features  = self.encode_text(text)

        # normalized features
        image_features = # TODO: L2-normalize image_features along dim=1 (divide by its norm, keepdim=True)
        text_features  = # TODO: L2-normalize text_features  along dim=1

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = # TODO: logit_scale * image_features @ text_features.t()
        logits_per_text  = # TODO: transpose of logits_per_image

        # shape = [batch_size, batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()
