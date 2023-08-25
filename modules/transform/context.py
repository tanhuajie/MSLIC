import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.layers import subpel_conv3x3, conv3x3

class ChannelContext(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.fushion = nn.Sequential(
            nn.Conv2d(in_dim, 192, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(128, out_dim, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, channel_params):
        """
        Args:
            channel_params(Tensor): [B, C * K, H, W]
        return:
            channel_params(Tensor): [B, C * 2, H, W]
        """
        channel_params = self.fushion(channel_params)

        return channel_params


class ChannelContext_MS(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.fushion = nn.Sequential(
            nn.Conv2d(in_dim, in_dim*2, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(in_dim*2, in_dim*3//2, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(in_dim*3//2, out_dim, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, channel_params):
        
        channel_params = self.fushion(channel_params)

        return channel_params

class ScaleX2Context_MS(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.fushion = nn.Sequential(
            conv3x3(in_dim, in_dim*2),
            nn.GELU(),
            subpel_conv3x3(in_dim*2, in_dim, 2),
            nn.GELU(),
            conv3x3(in_dim, out_dim),
        )

    def forward(self, channel_params):
        
        channel_params = self.fushion(channel_params)

        return channel_params
    
class ScaleX4Context_MS(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.fushion = nn.Sequential(
            conv3x3(in_dim, in_dim*2),
            nn.GELU(),
            subpel_conv3x3(in_dim*2, in_dim, 2),
            nn.GELU(),
            subpel_conv3x3(in_dim, in_dim//2, 2),
            nn.GELU(),
            conv3x3(in_dim//2, out_dim),
        )

    def forward(self, channel_params):
        
        channel_params = self.fushion(channel_params)

        return channel_params

class ChannelContextEX(nn.Module):
    def __init__(self, in_dim, out_dim, act=nn.GELU) -> None:
        super().__init__()
        self.fushion = nn.Sequential(
            nn.Conv2d(in_dim, 224, kernel_size=3, stride=1, padding=1),
            act(),
            nn.Conv2d(224, 128, kernel_size=3, stride=1, padding=1),
            act(),
            nn.Conv2d(128, out_dim, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, channel_params):
        """
        Args:
            channel_params(Tensor): [B, C * K, H, W]
        return:
            channel_params(Tensor): [B, C * 2, H, W]
        """
        channel_params = self.fushion(channel_params)

        return channel_params