import torch
import torch.nn as nn
import torch.nn.functional as F


class PyramidPoolingModule(nn.Module):
    """
    Reference implementation adapted from:
    https://github.com/cheeyeo/PSPNET_tutorial
    """

    def __init__(self, in_channels, out_channels, bin_sizes):
        super(PyramidPoolingModule, self).__init__()
        self.pyramid_pool_layers = nn.ModuleList()

        for bin_sz in bin_sizes:
            ppm = nn.Sequential(
                nn.AdaptiveAvgPool2d(bin_sz),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            self.pyramid_pool_layers.append(ppm)

    def forward(self, x):
        x_size = x.size()
        out = [x]

        for layer in self.pyramid_pool_layers:
            res = F.interpolate(layer(x), x_size[2:], mode="bilinear", align_corners=True)
            out.append(res)

        return torch.cat(out, 1)
