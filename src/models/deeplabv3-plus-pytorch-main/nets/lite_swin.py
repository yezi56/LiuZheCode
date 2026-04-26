import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def window_partition(x, window_size):
    batch_size, height, width, channels = x.shape
    x = x.view(
        batch_size,
        height // window_size,
        window_size,
        width // window_size,
        window_size,
        channels,
    )
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, channels)
    return windows


def window_reverse(windows, window_size, height, width, batch_size):
    x = windows.view(
        batch_size,
        height // window_size,
        width // window_size,
        window_size,
        window_size,
        -1,
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch_size, height, width, -1)
    return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, dropout=0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size=7, num_heads=4, attn_dropout=0.0, proj_dropout=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x):
        batch_windows, num_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch_windows, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(batch_windows, num_tokens, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads=4, window_size=7, shift_size=0, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=window_size, num_heads=num_heads, proj_dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, hidden_features=int(dim * mlp_ratio), dropout=dropout)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        shortcut = x

        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm1(x)

        pad_h = (self.window_size - height % self.window_size) % self.window_size
        pad_w = (self.window_size - width % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        _, padded_h, padded_w, _ = x.shape

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        windows = window_partition(x, self.window_size)
        windows = self.attn(windows)
        x = window_reverse(windows, self.window_size, padded_h, padded_w, batch_size)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = x[:, :height, :width, :]
        x = x.permute(0, 3, 1, 2).contiguous()
        x = shortcut + x

        x_ffn = x.permute(0, 2, 3, 1).contiguous()
        x_ffn = self.norm2(x_ffn)
        x_ffn = self.mlp(x_ffn)
        x_ffn = x_ffn.permute(0, 3, 1, 2).contiguous()
        x = x + x_ffn
        return x


class LightweightSwinBranch(nn.Module):
    """
    Small Swin Transformer branch for segmentation fusion.

    Design goal:
    - reuse MobileNetV2 low-level features instead of duplicating the image stem
    - keep window attention local to control parameter and memory growth
    - use alternating regular / shifted windows for stronger local-global mixing
    """

    def __init__(
        self,
        in_channels=24,
        embed_dim=192,
        depth=4,
        num_heads=4,
        window_size=7,
        mlp_ratio=2.0,
        patch_stride=4,
        out_channels=128,
        dropout=0.0,
    ):
        super().__init__()
        self.patch_stride = patch_stride
        self.window_size = window_size

        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_stride, stride=patch_stride, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        self.pos_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, groups=embed_dim, bias=False)

        blocks = []
        for idx in range(depth):
            shift = 0 if idx % 2 == 0 else window_size // 2
            blocks.append(
                SwinBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=shift,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.out_proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, target_size=None):
        x = self.patch_embed(x)
        x = x + self.pos_conv(x)

        for block in self.blocks:
            x = block(x)

        x = self.out_proj(x)
        if target_size is not None and x.shape[2:] != target_size:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=True)
        return x
