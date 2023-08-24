import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.layers import subpel_conv3x3
from layers.conv import conv1x1, conv3x3, conv, deconv
from layers.res_blk import *


class AnalysisTransform_L1(nn.Module):
    def __init__(self, InDim=96, N=144, M=192):
        super().__init__()
        self.analysis_transform = nn.Sequential(
            ResidualBlock(InDim, N),
            ResidualBlock(N, N),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlock(N, N),
            conv3x3(N, M, stride=1)
        )

    def forward(self, x):
        x = self.analysis_transform(x)

        return x
    
class AnalysisTransform_L2(nn.Module):
    def __init__(self, InDim=96, N=96, M=96):
        super().__init__()
        self.analysis_transform = nn.Sequential(
            ResidualBlock(InDim, N),
            ResidualBlock(N, N),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlock(N, N),
            conv3x3(N, M, stride=1)
        )

    def forward(self, x):
        x = self.analysis_transform(x)

        return x
    
class AnalysisTransform_L3(nn.Module):
    def __init__(self, InDim=192, N=96, M=48):
        super().__init__()
        self.analysis_transform = nn.Sequential(
            ResidualBlock(InDim, N),
            ResidualBlock(N, N),
            ResidualBlock(N, N),
            conv3x3(N, M, stride=1)
        )

    def forward(self, x):
        x = self.analysis_transform(x)

        return x

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


class HyperAnalysis(nn.Module):
    """
    Local reference
    """
    def __init__(self, M=336, N=192):
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