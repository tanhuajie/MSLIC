import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.layers import subpel_conv3x3, AttentionBlock
from modules.layers.conv import conv, deconv, conv3x3
from modules.layers.res_blk import *


class HyperSynthesis(nn.Module):
    """
    Local Reference
    """
    def __init__(self, M=192, N=192) -> None:
        super().__init__()
        self.M = M
        self.N = N

        self.increase = nn.Sequential(
            conv3x3(N, M),
            nn.GELU(),
            subpel_conv3x3(M, M, 2),
            nn.GELU(),
            conv3x3(M, M * 3 // 2),
            nn.GELU(),
            subpel_conv3x3(M * 3 // 2, M * 3 // 2, 2),
            nn.GELU(),
            conv3x3(M * 3 // 2, M * 2),
        )

    def forward(self, x):
        x = self.increase(x)

        return x

class HyperSynthesis_MS(nn.Module):

    def __init__(self, N=192, CH_S=32, CH_NUM=[8,8,8]):
        super().__init__()
        M = CH_S*CH_NUM[0] + CH_S*CH_NUM[1]//4 + CH_S*CH_NUM[2]//16
        self.split_M_lists = [2*CH_S*CH_NUM[0], 2*CH_S*CH_NUM[1]//4, 2*CH_S*CH_NUM[2]//16]
        self.M = M
        self.N = N
        self.increase = nn.Sequential(
            conv3x3(N, M),
            nn.GELU(),
            subpel_conv3x3(M, M, 2),
            nn.GELU(),
            conv3x3(M, M * 3 // 2),
            nn.GELU(),
            subpel_conv3x3(M * 3 // 2, M * 3 // 2, 2),
            nn.GELU(),
            conv3x3(M * 3 // 2, M * 2),
        )

        self.unshuffle_x2 = nn.PixelUnshuffle(2)
        self.unshuffle_x4 = nn.PixelUnshuffle(4)

    def forward(self, x):
        x = self.increase(x)
        x1, x2, x4 = x.split(self.split_M_lists, dim=1)
        x2 = self.unshuffle_x2(x2)
        x4 = self.unshuffle_x4(x4)
        return x1, x2, x4

class SynthesisTransform(nn.Module):
    def __init__(self, N, M):
        super().__init__()
        self.synthesis_transform = nn.Sequential(
            ResidualBlock(M, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 3, 2),
        )

    def forward(self, x):
        x = self.synthesis_transform(x)

        return x


class SynthesisTransform_MS_NotSplit(nn.Module):
    def __init__(self, N=192, CH_S=32, CH_NUM=[8,8,8]):
        super().__init__()

        self.synthesis_transform_u4 = nn.Sequential(
            ResidualBlock(CH_S * CH_NUM[2], N),
            ResidualBlock(N, N),
            ResidualBlockUpsampleRBs(N, N, 2)
        )

        self.synthesis_transform_u2 = nn.Sequential(
            ResidualBlock(N + CH_S * CH_NUM[1], 2*N),
            ResidualBlock(2*N, N),
            ResidualBlockUpsampleRBs(N, N, 2)
        )

        self.synthesis_transform_u1 = nn.Sequential(
            ResidualBlock(N + CH_S * CH_NUM[0], 2*N),
            ResidualBlock(2*N, N),
            ResidualBlockUpsampleRBs(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsampleRBs(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsampleRBs(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 3, 2)
        )

    def forward(self, x1, x2, x4):
        x4_hat = self.synthesis_transform_u4(x4)

        x2_cmb = torch.cat([x4_hat, x2], dim=1)
        x2_hat = self.synthesis_transform_u2(x2_cmb)

        x1_cmb = torch.cat([x2_hat, x1], dim=1)
        x1_hat = self.synthesis_transform_u1(x1_cmb)
        
        return x1_hat


class SynthesisTransform_MS(nn.Module):
    def __init__(self, N=192, CH_S=32, CH_NUM=[8,8,8]):
        super().__init__()

        self.synthesis_transform_u4 = nn.Sequential(
            ResidualBlock(CH_S * CH_NUM[2], N//12),
            ResidualBlockUpsampleRBs(N//12, N//12, 2),
        )

        self.synthesis_transform_u2 = nn.Sequential(
            ResidualBlock(N//12 + CH_S * CH_NUM[1], N//3),
            ResidualBlockUpsampleRBs(N//3, N//3, 2),
        )

        self.synthesis_transform_u1 = nn.Sequential(
            ResidualBlock(N//3 + CH_S * CH_NUM[0], N),
            ResidualBlockUpsampleRBs(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsampleRBs(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsampleRBs(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 3, 2),
        )

    def forward(self, x1, x2, x4):
        x4_hat = self.synthesis_transform_u4(x4)

        x2_cmb = torch.cat([x4_hat, x2], dim=1)
        x2_hat = self.synthesis_transform_u2(x2_cmb)

        x1_cmb = torch.cat([x2_hat, x1], dim=1)
        x1_hat = self.synthesis_transform_u1(x1_cmb)
        
        return x1_hat


class SynthesisTransformEX(nn.Module):
    def __init__(self, N, M, act=nn.GELU) -> None:
        super().__init__()
        self.synthesis_transform = nn.Sequential(
            AttentionBlock(M),
            deconv(M, N),
            ResidualBottleneck(N, act=act, groups=N * 2),
            ResidualBottleneck(N, act=act, groups=N * 2),
            ResidualBottleneck(N, act=act, groups=N * 2),
            deconv(N, N),
            AttentionBlock(N),
            ResidualBottleneck(N, act=act, groups=N * 2),
            ResidualBottleneck(N, act=act, groups=N * 2),
            ResidualBottleneck(N, act=act),
            deconv(N, N),
            ResidualBottleneck(N, act=act, groups=N * 2),
            ResidualBottleneck(N, act=act, groups=N * 2),
            ResidualBottleneck(N, act=act, groups=N * 2),
            deconv(N, 3)
        )

    def forward(self, x):
        x = self.synthesis_transform(x)
        return x


class HyperSynthesisEX(nn.Module):
    def __init__(self, N, M, act=nn.GELU) -> None:
        super().__init__()
        self.increase = nn.Sequential(
            deconv(N, M),
            act(),
            deconv(M, M * 3 // 2),
            act(),
            deconv(M * 3 // 2, M * 2, kernel_size=3, stride=1),
        )

    def forward(self, x):
        x = self.increase(x)
        return x