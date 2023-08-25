import torch
import torch.nn as nn
import torch.nn.functional as F

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