import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.layers.res_blk import *

import math

class MultiScaleAttention(nn.Module):
    def __init__(self, dim=32, ms_dim=32, out_dim=64, num_heads=2):
        super().__init__()
        self.dim = dim
        self.ms_dim = ms_dim
        self.num_heads = num_heads
        self.keys = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        )
        self.queries = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        )
        self.values = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        )
        self.anchors = nn.Sequential(
            nn.Conv2d(ms_dim, ms_dim, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(ms_dim, dim, kernel_size=3, stride=1, padding=1, groups=math.gcd(ms_dim, dim))
        )
        self.reprojection = nn.Conv2d(dim, out_dim * 3 // 2, kernel_size=5, stride=1, padding=2)
        self.mlp = nn.Sequential(
            nn.Conv2d(out_dim * 3 // 2, out_dim * 2, kernel_size=1, stride=1),
            nn.GELU(),
            nn.Conv2d(out_dim * 2, out_dim * 2, kernel_size=3, stride=1, padding=1, groups=out_dim * 2),
            nn.GELU(),
            nn.Conv2d(out_dim * 2, out_dim, kernel_size=1, stride=1)
        )
        self.skip = nn.Conv2d(out_dim * 3 // 2, out_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        B2, C2, H2, W2 = x2.shape

        queries = self.queries(x1).reshape(B, self.dim, H * W)
        keys = self.keys(x1).reshape(B, self.dim, H * W)
        values = self.values(x1).reshape(B, self.dim, H * W)
        anchors = self.anchors(x2).reshape(B, self.dim, H2 * W2)

        head_dim = self.dim // self.num_heads

        attended_values = []
        for i in range(self.num_heads):
            anchor = anchors[:, i * head_dim: (i + 1) * head_dim, :]

            key = keys[:, i * head_dim: (i + 1) * head_dim, :]
            key_anchor = key.transpose(1, 2) @ anchor
            key_anchor = F.softmax(key_anchor, dim=1)

            query = queries[:, i * head_dim: (i + 1) * head_dim, :]
            query_anchor = anchor.transpose(1, 2) @ query
            query_anchor = F.softmax(query_anchor, dim=1)

            value = values[:, i * head_dim: (i + 1) * head_dim, :]

            context = value @ key_anchor
            attended_value = (context @ query_anchor).reshape(B, head_dim, H, W)
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        attention = self.reprojection(aggregated_values)

        return self.skip(attention) + self.mlp(attention)
    

def init_bn(module):
    if module.weight is not None:
        nn.init.ones_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return


def init_uniform(module, init_method):
    if module.weight is not None:
        if init_method == "kaiming":
            nn.init.kaiming_uniform_(module.weight)
        elif init_method == "xavier":
            nn.init.xavier_uniform_(module.weight)
    return


class MultiScaleFeatureNet(nn.Module):
    def __init__(self, base_channels, in_channels, N, CH_S, CH_NUM):
        super(MultiScaleFeatureNet, self).__init__()
        self.base_channels = base_channels

        self.conv0 = nn.Sequential(
                Conv2d(in_channels, base_channels, 3, 1, padding=1),
                Conv2d(base_channels, base_channels, 3, 1, padding=1))

        self.conv1 = nn.Sequential(
                Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2),
                Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
                Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1))

        self.conv2 = nn.Sequential(
                Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
                Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
                Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1))

        self.out1 = nn.Sequential(
            ResidualBlockWithStrideRBs(base_channels * 4, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStrideRBs(N, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStrideRBs(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, CH_S * CH_NUM[2], stride=2)
            )

        self.inner1 = nn.Conv2d(base_channels * 2, base_channels * 4, 1, bias=True)
        self.inner2 = nn.Conv2d(base_channels * 1, base_channels * 4, 1, bias=True)

        self.out2 = nn.Sequential(
            ResidualBlockWithStrideRBs(base_channels * 4, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStrideRBs(N, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStrideRBs(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, CH_S * CH_NUM[1], stride=2)
            )
        
        self.out3 = nn.Sequential(
            ResidualBlockWithStrideRBs(base_channels * 4, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStrideRBs(N, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStrideRBs(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, CH_S * CH_NUM[0], stride=2)
            )

    def forward(self, x):
        """forward.

        :param x: [B, C, H, W]

        """
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)

        intra_feat = conv2
        out = self.out1(intra_feat)
        outputs_ds4 = out

        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner1(conv1)
        out = self.out2(intra_feat)
        outputs_ds2 = out

        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner2(conv0)
        out = self.out3(intra_feat)
        outputs_ds1 = out

        return outputs_ds1, outputs_ds2, outputs_ds4


class Conv2d(nn.Module):
    """Applies a 2D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Conv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class Deconv2d(nn.Module):
    """Applies a 2D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Deconv2d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

    def forward(self, x):
        y = self.conv(x)
        if self.stride == 2:
            h, w = list(x.size())[2:]
            y = y[:, :, :2 * h, :2 * w].contiguous()
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)