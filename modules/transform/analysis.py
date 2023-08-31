import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.layers import subpel_conv3x3
from modules.layers.conv import conv3x3, conv, deconv
from modules.layers.res_blk import *


class AnalysisTransform(nn.Module):
    def __init__(self, N, M):
        super().__init__()
        self.analysis_transform = nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, M, stride=2)
        )

    def forward(self, x):
        x = self.analysis_transform(x)

        return x
    
class AnalysisTransform_MS_NotSplit(nn.Module):
    def __init__(self, N=192, CH_S=32, CH_NUM=[8,8,8]):
        super().__init__()
        self.N = N
        self.CH_S = CH_S
        self.analysis_transform_d1 = nn.Sequential(
            ResidualBlockWithStrideRBs(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStrideRBs(N, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStrideRBs(N, N, stride=2),
            ResidualBlock(N, N),
        )

        self.analysis_transform_d2 = nn.Sequential(
            ResidualBlockWithStrideRBs(N, N, stride=2),
            ResidualBlock(N, N)
        )

        self.analysis_transform_d4 = nn.Sequential(
            ResidualBlockWithStrideRBs(N, N, stride=2),
            ResidualBlock(N, N)
        )

        self.ms1 = conv3x3(N, CH_S * CH_NUM[0], stride=2)
        self.ms2 = conv3x3(N, CH_S * CH_NUM[1], stride=2)
        self.ms4 = conv3x3(N, CH_S * CH_NUM[2], stride=2)

    def forward(self, x):
        x1 = self.analysis_transform_d1(x)
        x2 = self.analysis_transform_d2(x1)
        x4 = self.analysis_transform_d4(x2)

        x1 = self.ms1(x1)
        x2 = self.ms2(x2)
        x4 = self.ms4(x4)

        return x1, x2, x4




class AnalysisTransform_MS(nn.Module):
    def __init__(self, N=192, CH_S=32, CH_NUM=[8,8,8]):
        super().__init__()
        self.N = N
        self.CH_S = CH_S
        self.analysis_transform_d1 = nn.Sequential(
            ResidualBlockWithStrideRBs(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStrideRBs(N, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStrideRBs(N, N, stride=2),
            ResidualBlock(N, N),
        )

        self.analysis_transform_d2 = nn.Sequential(
            ResidualBlockWithStrideRBs(N//3, N//3, stride=2),
            ResidualBlock(N//3, N//3)
        )

        self.analysis_transform_d4 = nn.Sequential(
            ResidualBlockWithStrideRBs(N//12, N//12, stride=2),
            ResidualBlock(N//12, N//12)
        )

        self.ms1 = conv3x3(N*2//3, CH_S * CH_NUM[0], stride=2)
        self.ms2 = conv3x3(N // 4, CH_S * CH_NUM[1], stride=2)
        self.ms4 = conv3x3(N //12, CH_S * CH_NUM[2], stride=2)

    def forward(self, x):
        x = self.analysis_transform_d1(x)
        x1, x2_x4 = x.split([self.N*2//3, self.N//3], dim=1)
        x2_x4 = self.analysis_transform_d2(x2_x4)
        x2, x4 = x2_x4.split([self.N//4, self.N//12], dim=1)
        x4 = self.analysis_transform_d4(x4)

        x1 = self.ms1(x1)
        x2 = self.ms2(x2)
        x4 = self.ms4(x4)

        return x1, x2, x4

class HyperAnalysis(nn.Module):
    """
    Local reference
    """
    def __init__(self, M=192, N=192):
        super().__init__()
        self.M = M
        self.N = N
        self.reduction = nn.Sequential(
            conv3x3(M, N),
            nn.GELU(),
            conv3x3(N, N),
            nn.GELU(),
            conv3x3(N, N, stride=2),
            nn.GELU(),
            conv3x3(N, N),
            nn.GELU(),
            conv3x3(N, N, stride=2),
        )

    def forward(self, x):
        x = self.reduction(x)

        return x
    
class HyperAnalysis_MS(nn.Module):

    def __init__(self, N=192, CH_S=32, CH_NUM=[8,8,8]):
        super().__init__()
        M = CH_S*CH_NUM[0] + CH_S*CH_NUM[1]//4 + CH_S*CH_NUM[2]//16
        self.M = M
        self.N = N
        self.reduction = nn.Sequential(
            conv3x3(M, N),
            nn.GELU(),
            conv3x3(N, N),
            nn.GELU(),
            conv3x3(N, N, stride=2),
            nn.GELU(),
            conv3x3(N, N),
            nn.GELU(),
            conv3x3(N, N, stride=2),
        )
        self.shuffle_x2 = nn.PixelShuffle(2)
        self.shuffle_x4 = nn.PixelShuffle(4)

    def forward(self, x1, x2, x4):
        x2 = self.shuffle_x2(x2)
        x4 = self.shuffle_x4(x4)
        x = torch.cat([x1, x2, x4], dim=1)
        x = self.reduction(x)
        return x

class AnalysisTransformEX(nn.Module):
    def __init__(self, N, M, act=nn.GELU):
        super().__init__()
        self.analysis_transform = nn.Sequential(
            conv(3, N),
            ResidualBottleneck(N, act=act, groups=N * 2),
            ResidualBottleneck(N, act=act, groups=N * 2),
            ResidualBottleneck(N, act=act, groups=N * 2),
            conv(N, N),
            ResidualBottleneck(N, act=act, groups=N * 2),
            ResidualBottleneck(N, act=act, groups=N * 2),
            ResidualBottleneck(N, act=act, groups=N * 2),
            AttentionBlock(N),
            conv(N, N),
            ResidualBottleneck(N, act=act, groups=N * 2),
            ResidualBottleneck(N, act=act, groups=N * 2),
            ResidualBottleneck(N, act=act, groups=N * 2),
            conv(N, M),
            AttentionBlock(M)
        )

    def forward(self, x):
        x = self.analysis_transform(x)
        return x


class HyperAnalysisEX(nn.Module):
    def __init__(self, N, M, act=nn.GELU) -> None:
        super().__init__()
        self.M = M
        self.N = N
        self.reduction = nn.Sequential(
            conv3x3(M, N),
            act(),
            conv(N, N),
            act(),
            conv(N, N)
        )

    def forward(self, x):
        x = self.reduction(x)
        return x