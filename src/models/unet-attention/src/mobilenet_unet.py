from collections import OrderedDict
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models import mobilenet_v3_large
from pathlib import Path
import sys

ATTENTION_ZOO = Path(__file__).resolve().parents[3] / "modules" / "attention_zoo"
if str(ATTENTION_ZOO) not in sys.path:
    sys.path.append(str(ATTENTION_ZOO))

from ECA import ECA_layer
from EMA import EMA
from LSK import LSKNet
from ELA import ELA
from Biformer import BiLevelRoutingAttention as BRA


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None,
                 use_eca=False):  # , use_ema=False, use_ela=False, use_bra=False):  # 其他注意力
        super(DoubleConv, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        # ====== 注意力模块（需要时取消注释）======
        self.eca = ECA_layer(out_channels) if use_eca else None
        # self.ema = EMA(channels=out_channels) if use_ema else None
        # self.ela = ELA(out_channels, phi="T") if use_ela else None
        # self.bra = BRA(out_channels) if use_bra else None

    def forward(self, x):
        x = self.double_conv(x)
        
        # ====== 注意力模块前向传播（需要时取消注释）======
        if self.eca is not None:
            x = self.eca(x)
        # if self.ema is not None:
        #     x = self.ema(x)
        # if self.ela is not None:
        #     x = self.ela(x)
        # if self.bra is not None:
        #     x = self.bra(x)
        
        return x


class Up(nn.Module):
    """上采样模块，支持可选的注意力机制"""
    def __init__(self, in_channels, out_channels, bilinear=True,
                 use_eca=False):  # , use_ema=False, use_ela=False, use_bra=False):  # 其他注意力
        super(Up, self).__init__()
        # ====== 注意力参数（需要时取消注释）======
        self.use_eca = use_eca
        # self.use_ema = use_ema
        # self.use_ela = use_ela
        # self.use_bra = use_bra
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2,
                                 use_eca=use_eca)  # , use_ema=use_ema, use_ela=use_ela, use_bra=use_bra)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels,
                                 use_eca=use_eca)  # , use_ema=use_ema, use_ela=use_ela, use_bra=use_bra)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    """输出卷积层"""
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class IntermediateLayerGetter(nn.ModuleDict):

    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}

        # 重新构建backbone，将没有使用到的模块全部删掉
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class MobileV3Unet(nn.Module):
    def __init__(self, num_classes, pretrain_backbone: bool = False,
                 use_eca: bool = False):  # , use_ema=False, use_ela=False, use_bra=False):  # 其他注意力
        super(MobileV3Unet, self).__init__()
        backbone = mobilenet_v3_large(pretrained=pretrain_backbone)

        # if pretrain_backbone:
        #     # 载入mobilenetv3 large backbone预训练权重
        #     # https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth
        #     backbone.load_state_dict(torch.load("mobilenet_v3_large.pth", map_location='cpu'))

        backbone = backbone.features

        stage_indices = [1, 3, 6, 12, 15]
        self.stage_out_channels = [backbone[i].out_channels for i in stage_indices]
        return_layers = dict([(str(j), f"stage{i}") for i, j in enumerate(stage_indices)])
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

        c = self.stage_out_channels[4] + self.stage_out_channels[3]
        self.up1 = Up(c, self.stage_out_channels[3],
                     use_eca=use_eca)  # , use_ema=use_ema, use_ela=use_ela, use_bra=use_bra)
        c = self.stage_out_channels[3] + self.stage_out_channels[2]
        self.up2 = Up(c, self.stage_out_channels[2],
                     use_eca=use_eca)  # , use_ema=use_ema, use_ela=use_ela, use_bra=use_bra)
        c = self.stage_out_channels[2] + self.stage_out_channels[1]
        self.up3 = Up(c, self.stage_out_channels[1],
                     use_eca=use_eca)  # , use_ema=use_ema, use_ela=use_ela, use_bra=use_bra)
        c = self.stage_out_channels[1] + self.stage_out_channels[0]
        self.up4 = Up(c, self.stage_out_channels[0],
                     use_eca=use_eca)  # , use_ema=use_ema, use_ela=use_ela, use_bra=use_bra)
        self.conv = OutConv(self.stage_out_channels[0], num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        input_shape = x.shape[-2:]
        backbone_out = self.backbone(x)
        x = self.up1(backbone_out['stage4'], backbone_out['stage3'])
        x = self.up2(x, backbone_out['stage2'])
        x = self.up3(x, backbone_out['stage1'])
        x = self.up4(x, backbone_out['stage0'])
        x = self.conv(x)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)

        return {"out": x}


if __name__ == "__main__":
    # 测试 ECA 注意力
    model = MobileV3Unet(num_classes=2, pretrain_backbone=False, use_eca=True)
    input_tensor = torch.randn(1, 3, 256, 256)
    output = model(input_tensor)
    print(f"输入形状: {input_tensor.shape}")
    print(f"输出形状: {output['out'].shape}")
