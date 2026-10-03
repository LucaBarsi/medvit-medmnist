"""Model definitions.

SimpleCNN is intentionally shallow: its only job is to validate the full
pipeline (data -> training loop -> metrics -> checkpoint) end to end before
we invest time in ResNet, ViT and MedViT.
"""
from typing import List

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """A small 3-block CNN for 28x28 (or similarly small) grayscale/RGB images."""

    def __init__(self, n_channels: int, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(n_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28 -> 14

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14 -> 7

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # -> (B, 64, 1, 1), works for any input size
        )
        self.classifier = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class BasicBlock(nn.Module):
    """Two 3x3 convolutions with a residual (skip) connection, as in
    He et al. 2015. `downsample` projects the skip connection when the
    spatial size or channel count changes between input and output.
    """
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity  # the residual connection
        return self.relu(out)


class ResNet18Small(nn.Module):
    """ResNet-18, adapted for small (e.g. 28x28) images.

    The original ResNet-18 (He et al. 2015) was designed for 224x224
    ImageNet images: its "stem" is a 7x7 stride-2 convolution followed by a
    3x3 stride-2 max pool, which alone divides the input by 4 before any
    residual block runs. On 28x28 inputs that stem would shrink the image
    to 7x7 immediately, leaving almost nothing for the residual stages to
    work with. This is a well-known issue, and the standard fix (used e.g.
    for CIFAR-sized ResNets) is to replace the stem with a single 3x3
    stride-1 convolution and drop the max pool, keeping the four residual
    stages (with channel counts 64/128/256/512) unchanged. Downsampling
    then happens only via the stride-2 convolutions inside stages 2-4.
    """

    def __init__(self, n_channels: int, n_classes: int,
                blocks_per_stage: List[int] = (2, 2, 2, 2)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.stage1 = self._make_stage(64, 64, blocks_per_stage[0], stride=1)
        self.stage2 = self._make_stage(64, 128, blocks_per_stage[1], stride=2)
        self.stage3 = self._make_stage(128, 256, blocks_per_stage[2], stride=2)
        self.stage4 = self._make_stage(256, 512, blocks_per_stage[3], stride=2)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, n_classes)

    @staticmethod
    def _make_stage(in_channels: int, out_channels: int, n_blocks: int,
                    stride: int) -> nn.Sequential:
        layers = [BasicBlock(in_channels, out_channels, stride=stride)]
        for _ in range(n_blocks - 1):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class PatchEmbedding(nn.Module):
    """Splits the image into non-overlapping patches and linearly projects
    each one into an embedding vector. A single strided convolution does
    both the splitting and the projection in one step: a conv with
    kernel_size == stride == patch_size applies the *same* linear layer to
    every non-overlapping patch, which is mathematically identical to
    flattening each patch and multiplying by a weight matrix.
    """

    def __init__(self, n_channels: int, patch_size: int, embed_dim: int, img_size: int):
        super().__init__()
        assert img_size % patch_size == 0, "image size must be divisible by patch size"
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(n_channels, embed_dim, kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)                  # (B, embed_dim, n_patches)
        return x.transpose(1, 2)          # (B, n_patches, embed_dim)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention, implemented explicitly (no nn.MultiheadAttention)
    so every step of the computation is visible:
      1. project the input into Query, Key and Value
      2. split each into `n_heads` independent subspaces
      3. compute scaled dot-product attention per head
      4. concatenate the heads back together and project once more
    """

    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape  # batch, n_tokens (patches + cls), embed_dim

        qkv = self.qkv(x)                                   # (B, N, 3*D)
        qkv = qkv.reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                     # (3, B, n_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # (B, n_heads, N, N)
        attn_weights = attn_scores.softmax(dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = attn_weights @ v                                # (B, n_heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, D)             # concat heads back
        return self.proj(out)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: LayerNorm -> attention -> residual,
    LayerNorm -> MLP -> residual. Pre-norm (norm before the sublayer,
    rather than after) is the convention used by ViT and most modern
    transformers, and trains more stably than the original post-norm design.
    """

    def __init__(self, embed_dim: int, n_heads: int, mlp_ratio: float = 4.0,
                dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    """Vision Transformer, built from scratch (Dosovitskiy et al., 2020).

    Sized for small images and small datasets rather than ImageNet: the
    original ViT uses 16x16 patches on 224x224 images (196 patches) with
    embed_dim=768 and depth=12. Those defaults would leave almost no
    patches on a 28x28 image and would massively overfit on a few thousand
    training images, so this version uses a much smaller patch size and a
    narrower, shallower transformer — a config in line with ViT variants
    used in small-image benchmarks.
    """

    def __init__(self, n_channels: int, n_classes: int, img_size: int = 28,
                patch_size: int = 4, embed_dim: int = 64, depth: int = 4,
                n_heads: int = 4, mlp_ratio: float = 2.0, dropout: float = 0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(n_channels, patch_size, embed_dim, img_size)
        n_patches = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)                          # (B, n_patches, embed_dim)

        cls_tokens = self.cls_token.expand(B, -1, -1)     # (B, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)              # (B, n_patches+1, embed_dim)
        x = x + self.pos_embed
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        cls_output = x[:, 0]                               # classification uses only the CLS token
        return self.head(cls_output)


class ECBBlock(nn.Module):
    """Efficient Convolution Block (simplified).

    MedViT's real ECB combines a convolutional multi-head attention block
    with a "Locally feed-forward Network" (LFFN). This simplification keeps
    the spirit — local, efficient feature mixing with a residual connection
    — using a depthwise 3x3 conv (spatial mixing, one filter per channel,
    cheap) followed by a pointwise 1x1 "expand -> GELU -> project" pair
    (channel mixing), which is the same depthwise + pointwise pattern used
    by efficient CNNs (e.g. MobileNet, ConvNeXt). This serves as both the
    "convolutional attention" and the "LFFN" roles of the real ECB, merged
    into a single block for simplicity.
    """

    def __init__(self, channels: int, expand_ratio: float = 2.0):
        super().__init__()
        hidden = int(channels * expand_ratio)
        self.depthwise = nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                                   groups=channels, bias=False)
        self.norm1 = nn.BatchNorm2d(channels)
        self.pointwise1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.act = nn.GELU()
        self.pointwise2 = nn.Conv2d(hidden, channels, kernel_size=1)
        self.norm2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.norm1(self.depthwise(x))              # local spatial mixing
        x = x + self.norm2(self.pointwise2(self.act(self.pointwise1(x))))  # channel mixing
        return x


class ConvAttentionBranch(nn.Module):
    """Stand-in for MedViT's "multi-head convolutional attention" branch,
    meant to capture high-frequency (local, fine-grained) signal. A real
    convolutional-attention mechanism is involved; here a depthwise conv
    over the 2D token grid plays the same qualitative role — local mixing
    restricted to nearby tokens, as opposed to the global self-attention
    branch below.
    """

    def __init__(self, dim: int, grid_size: int):
        super().__init__()
        self.grid_size = grid_size
        self.depthwise = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.pointwise = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, N, D = tokens.shape
        H = W = self.grid_size
        x = tokens.transpose(1, 2).reshape(B, D, H, W)    # tokens -> 2D grid
        x = self.pointwise(self.depthwise(x))
        return x.flatten(2).transpose(1, 2)                # 2D grid -> tokens


class HybridAttention(nn.Module):
    """MedViT's central idea: combine a global self-attention branch
    (low-frequency signal, long-range dependencies — reuses the same
    MultiHeadSelfAttention built for the ViT) with a local convolutional
    branch (high-frequency signal, fine detail), then merge the two.
    """

    def __init__(self, dim: int, n_heads: int, grid_size: int, dropout: float = 0.0):
        super().__init__()
        self.global_branch = MultiHeadSelfAttention(dim, n_heads, dropout)
        self.local_branch = ConvAttentionBranch(dim, grid_size)
        self.merge = nn.Linear(dim * 2, dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        global_out = self.global_branch(tokens)
        local_out = self.local_branch(tokens)
        return self.merge(torch.cat([global_out, local_out], dim=-1))


class LocalFeedForward(nn.Module):
    """LFFN: rearranges the token sequence back into a 2D grid, applies a
    depthwise conv (giving the MLP a notion of spatial locality that a
    plain, position-independent MLP does not have), then rearranges back
    into a sequence before the usual pointwise expand/project.
    """

    def __init__(self, dim: int, grid_size: int, mlp_ratio: float = 2.0):
        super().__init__()
        self.grid_size = grid_size
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.depthwise = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, N, D = tokens.shape
        H = W = self.grid_size
        x = self.fc1(tokens)                                # (B, N, hidden)
        x = x.transpose(1, 2).reshape(B, -1, H, W)          # tokens -> 2D grid
        x = self.act(self.depthwise(x))
        x = x.flatten(2).transpose(1, 2)                    # 2D grid -> tokens
        return self.fc2(x)


class LTBBlock(nn.Module):
    """Local Transformer Block: pre-norm hybrid attention + pre-norm LFFN,
    each with a residual connection — the same skeleton as the plain
    TransformerBlock used in the ViT, but with the hybrid attention and
    the locality-aware feed-forward network described above.
    """

    def __init__(self, dim: int, n_heads: int, grid_size: int,
                mlp_ratio: float = 2.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = HybridAttention(dim, n_heads, grid_size, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = LocalFeedForward(dim, grid_size, mlp_ratio)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens + self.attn(self.norm1(tokens))
        tokens = tokens + self.ffn(self.norm2(tokens))
        return tokens


class MedViTSmall(nn.Module):
    """Simplified MedViT: a hierarchical hybrid of convolutional (ECB) and
    transformer (LTB) stages, in the spirit of MedViT (Manzari et al., 2023).

    Design, and how it differs from the real MedViT:
    - Early stages are pure ECB (convolutional): cheap, local, good at the
      fine-grained texture that matters most while the feature map is
      still large (28x28, 14x14).
    - The final stage tokenizes the feature map and runs LTB blocks
      (hybrid global+local attention), where global context is most useful
      and the token count is small enough to be cheap.
    - The real MedViT's attention mechanisms (a tailored convolutional
      multi-head attention, and an "efficient" self-attention variant) are
      replaced here by the simpler stand-ins above (ECBBlock,
      ConvAttentionBranch) — see their docstrings. The overall idea —
      "convolutional locality early, hybrid global/local attention late,
      in a pyramid of decreasing resolution and increasing channels" — is
      preserved; the exact attention math is not.
    """

    def __init__(self, n_channels: int, n_classes: int, img_size: int = 28,
                dropout: float = 0.1):
        super().__init__()
        # Stage 1: ECB @ 28x28, 32 channels
        self.stem = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.GELU(),
        )
        self.stage1 = nn.Sequential(ECBBlock(32), ECBBlock(32))
        self.down1 = nn.Sequential(                          # 28x28 -> 14x14, 32 -> 64
            nn.Conv2d(32, 64, kernel_size=2, stride=2), nn.BatchNorm2d(64), nn.GELU(),
        )

        # Stage 2: ECB @ 14x14, 64 channels
        self.stage2 = nn.Sequential(ECBBlock(64))
        self.down2 = nn.Sequential(                          # 14x14 -> 7x7, 64 -> 128
            nn.Conv2d(64, 128, kernel_size=2, stride=2), nn.BatchNorm2d(128), nn.GELU(),
        )

        # Stage 3: LTB @ 7x7 (49 tokens), 128 channels
        grid_size = img_size // 4  # two stride-2 downsamplings
        self.grid_size = grid_size
        self.pos_embed = nn.Parameter(torch.zeros(1, grid_size * grid_size, 128))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.stage3 = nn.ModuleList([
            LTBBlock(dim=128, n_heads=4, grid_size=grid_size, dropout=dropout)
            for _ in range(2)
        ])
        self.norm = nn.LayerNorm(128)
        self.head = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)                                    # (B, 128, 7, 7)

        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)                 # (B, 49, 128)
        tokens = tokens + self.pos_embed

        for block in self.stage3:
            tokens = block(tokens)
        tokens = self.norm(tokens)

        pooled = tokens.mean(dim=1)                           # global average over tokens
        return self.head(pooled)


def build_model(name: str, n_channels: int, n_classes: int) -> nn.Module:
    """Factory so notebooks can select a model by name/config instead of
    importing classes directly.
    """
    registry = {
        "simple_cnn": SimpleCNN,
        "resnet18": ResNet18Small,
        "vit": ViT,
        "medvit": MedViTSmall,
    }
    if name not in registry:
        raise ValueError(f"Unknown model '{name}'. Available: {list(registry)}")
    return registry[name](n_channels=n_channels, n_classes=n_classes)
