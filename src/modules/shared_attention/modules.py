from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1, stride: int = 1,
                 padding: int | None = None, groups: int = 1, activation: nn.Module | None = None) -> None:
        if padding is None:
            padding = kernel_size // 2
        if activation is None:
            activation = nn.SiLU()
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            activation,
        )


class SEAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class EfficientChannelAttention(nn.Module):
    def __init__(self, channels: int, gamma: int = 2, b: int = 1) -> None:
        super().__init__()
        kernel_size = int(abs((math.log2(channels) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        kernel_size = max(kernel_size, 3)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = self.act(y.transpose(-1, -2).unsqueeze(-1))
        return x * y


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return x * self.act(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.act(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn


class CBAMBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction=reduction)
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class CAA(nn.Module):
    def __init__(self, channels: int, h_kernel_size: int = 11, v_kernel_size: int = 11) -> None:
        super().__init__()
        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = ConvBNAct(channels, channels)
        self.h_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, h_kernel_size),
            padding=(0, h_kernel_size // 2),
            groups=channels,
            bias=False,
        )
        self.v_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=(v_kernel_size, 1),
            padding=(v_kernel_size // 2, 0),
            groups=channels,
            bias=False,
        )
        self.conv2 = ConvBNAct(channels, channels)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.avg_pool(x)
        attn = self.conv1(attn)
        attn = self.h_conv(attn)
        attn = self.v_conv(attn)
        attn = self.conv2(attn)
        return x * self.act(attn)


class CPCA(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channel_attention = SEAttention(channels, reduction=8)
        self.local = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.branch5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels, bias=False)
        self.branch7 = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels, bias=False)
        self.project = ConvBNAct(channels, channels, kernel_size=1)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        spatial = self.local(x) + self.branch5(x) + self.branch7(x)
        spatial = self.project(spatial)
        return x * self.act(spatial)


class ZPool(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([torch.max(x, 1, keepdim=True)[0], torch.mean(x, 1, keepdim=True)], dim=1)


class AttentionGate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.compress = ZPool()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.conv(self.compress(x))


class TripletAttention(nn.Module):
    def __init__(self, channels: int, no_spatial: bool = False) -> None:
        super().__init__()
        self.no_spatial = no_spatial
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.hw = AttentionGate()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_perm1 = self.cw(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        x_perm2 = self.hc(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        if self.no_spatial:
            return 0.5 * (x_perm1 + x_perm2)
        x_out = self.hw(x)
        return (x_out + x_perm1 + x_perm2) / 3.0


class ShuffleAttention(nn.Module):
    def __init__(self, channels: int, groups: int = 8) -> None:
        super().__init__()
        groups = max(1, min(groups, channels // 2 if channels >= 2 else 1))
        self.groups = groups
        split_channels = channels // (2 * groups)
        split_channels = max(split_channels, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = nn.Parameter(torch.zeros(1, split_channels, 1, 1))
        self.cbias = nn.Parameter(torch.ones(1, split_channels, 1, 1))
        self.sweight = nn.Parameter(torch.zeros(1, split_channels, 1, 1))
        self.sbias = nn.Parameter(torch.ones(1, split_channels, 1, 1))
        self.sigmoid = nn.Sigmoid()
        self.channels = channels

    def channel_shuffle(self, x: torch.Tensor, groups: int) -> torch.Tensor:
        b, c, h, w = x.shape
        x = x.reshape(b, groups, c // groups, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x.reshape(b, c, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if c % (2 * self.groups) != 0:
            return x
        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)
        xn = self.avg_pool(x_0)
        xn = self.sigmoid(self.cweight * xn + self.cbias) * x_0
        xs = self.sigmoid(self.sweight * x_1 + self.sbias) * x_1
        out = torch.cat([xn, xs], dim=1).reshape(b, c, h, w)
        return self.channel_shuffle(out, 2)


class EMCAM(nn.Module):
    def __init__(
        self,
        channels: int | None = None,
        expansion: int = 2,
        in_channels: int | None = None,
        out_channels: int | None = None,
    ) -> None:
        super().__init__()
        channels = channels or out_channels or in_channels
        if channels is None:
            raise ValueError("EMCAM requires `channels` or `in_channels`/`out_channels`.")
        hidden = max(channels // expansion, 8)
        self.pre = ConvBNAct(channels, hidden, kernel_size=1)
        self.dw3 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False)
        self.dw5 = nn.Conv2d(hidden, hidden, kernel_size=5, padding=2, groups=hidden, bias=False)
        self.dw7 = nn.Conv2d(hidden, hidden, kernel_size=7, padding=3, groups=hidden, bias=False)
        self.mix = ConvBNAct(hidden, channels, kernel_size=1)
        self.channel_gate = EfficientChannelAttention(channels)
        self.spatial_gate = SpatialAttention(kernel_size=7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fused = self.pre(x)
        fused = self.dw3(fused) + self.dw5(fused) + self.dw7(fused)
        fused = self.mix(fused)
        fused = self.channel_gate(fused)
        fused = self.spatial_gate(fused)
        return fused
