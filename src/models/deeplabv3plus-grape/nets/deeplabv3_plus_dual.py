import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.lite_swin import LightweightSwinBranch
from nets.mobilenetv2 import mobilenetv2
from nets.xception import xception

for _parent in Path(__file__).resolve().parents:
    direct_shared = _parent / "shared_attention"
    modules_shared = _parent / "modules" / "shared_attention"
    if direct_shared.is_dir():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break
    if modules_shared.is_dir():
        modules_root = _parent / "modules"
        if str(modules_root) not in sys.path:
            sys.path.insert(0, str(modules_root))
        break

from shared_attention import build_attention


class MobileNetV2(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super().__init__()
        from functools import partial

        model = mobilenetv2(pretrained)
        self.features = model.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=4))
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        low_level_features = self.features[:4](x)
        x = self.features[4:](low_level_features)
        return low_level_features, x


class MobileNetV2Swin(nn.Module):
    """
    Lightweight dual-backbone design:
    - MobileNetV2 is the main backbone.
    - Swin Transformer is an auxiliary branch built on top of shared low-level CNN features.
    - Window attention is used to limit token interaction cost and parameter growth.
    """

    def __init__(self, downsample_factor=8, pretrained=True):
        super().__init__()
        from functools import partial

        model = mobilenetv2(pretrained)
        self.features = model.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=4))
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))

        self.swin_branch = LightweightSwinBranch(
            in_channels=24,
            embed_dim=192,
            depth=4,
            num_heads=4,
            window_size=7,
            mlp_ratio=2.0,
            patch_stride=4,
            out_channels=128,
            dropout=0.0,
        )
        self.swin_fuse = nn.Sequential(
            nn.Conv2d(320 + 128, 320, kernel_size=1, bias=False),
            nn.BatchNorm2d(320),
            nn.ReLU(inplace=True),
        )

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        low_level_features = self.features[:4](x)
        mobile_high = self.features[4:](low_level_features)
        swin_high = self.swin_branch(low_level_features, target_size=mobile_high.shape[2:])
        fused_high = self.swin_fuse(torch.cat([mobile_high, swin_high], dim=1))
        return low_level_features, fused_high


class ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        _, _, row, col = x.size()
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)

        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, "bilinear", True)

        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        return self.conv_cat(feature_cat)


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, out_channels=256, pool_sizes=(1, 2, 3, 6)):
        super().__init__()
        # The structure follows the standard PSP-style PPM design. A clean
        # reference copy is tracked under:
        # src/modules/third_party/ppm_reference/pyramid_pooling_module.py
        # Here we keep the final projection inside the module so it can be
        # inserted directly after ASPP in DeepLabV3+.
        branch_channels = max(in_channels // len(pool_sizes), 1)
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_size),
                    nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(branch_channels),
                    nn.ReLU(inplace=True),
                )
                for pool_size in pool_sizes
            ]
        )
        self.project = nn.Sequential(
            nn.Conv2d(in_channels + branch_channels * len(pool_sizes), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        pyramids = [x]
        pyramids.extend(
            [F.interpolate(stage(x), size=(h, w), mode="bilinear", align_corners=True) for stage in self.stages]
        )
        return self.project(torch.cat(pyramids, dim=1))


class DeepLab(nn.Module):
    def __init__(self, num_classes, backbone="mobilenet", pretrained=True, downsample_factor=16, attention_type="", use_ppm=False, ppm_bins=(1, 2, 3, 6)):
        super().__init__()

        if backbone == "xception":
            self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 2048
            low_level_channels = 256
        elif backbone == "mobilenet":
            self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 320
            low_level_channels = 24
        elif backbone == "mobilenet_swin":
            self.backbone = MobileNetV2Swin(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 320
            low_level_channels = 24
        else:
            raise ValueError(f"Unsupported backbone - `{backbone}`, Use mobilenet, mobilenet_swin, xception.")

        self.attention_low = build_attention(attention_type, low_level_channels)
        self.attention_high = build_attention(attention_type, in_channels)
        self.aspp = ASPP(dim_in=in_channels, dim_out=256, rate=16 // downsample_factor)
        self.attention_aspp = build_attention(attention_type, 256)
        self.use_ppm = use_ppm
        self.ppm = PyramidPoolingModule(256, out_channels=256, pool_sizes=tuple(ppm_bins)) if use_ppm else nn.Identity()

        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.cat_conv = nn.Sequential(
            nn.Conv2d(48 + 256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.attention_decoder = build_attention(attention_type, 256)
        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)

    def forward(self, x):
        height, width = x.size(2), x.size(3)
        low_level_features, x = self.backbone(x)
        low_level_features = self.attention_low(low_level_features)
        x = self.attention_high(x)
        x = self.aspp(x)
        x = self.attention_aspp(x)
        x = self.ppm(x)
        low_level_features = self.shortcut_conv(low_level_features)

        x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode="bilinear", align_corners=True)
        x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
        x = self.attention_decoder(x)
        x = self.cls_conv(x)
        x = F.interpolate(x, size=(height, width), mode="bilinear", align_corners=True)
        return x
