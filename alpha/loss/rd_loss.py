import math
import torch
import torch.nn as nn
from pytorch_msssim import ms_ssim


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, metrics='mse'):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.metrics = metrics

    def set_lmbda(self, lmbda):
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W
        eps = 1e-5
        out["bpp_loss"] = sum(
            (torch.log(torch.clamp(likelihoods, eps, 1.0)).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        if self.metrics == 'mse':
            out["mse_loss"] = self.mse(output["x_hat"], target)
            out["ms_ssim_loss"] = None
            out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]
        elif self.metrics == 'ms-ssim':
            out["mse_loss"] = None
            out["ms_ssim_loss"] = 1 - ms_ssim(output["x_hat"], target, data_range=1.0)
            out["loss"] = self.lmbda * out["ms_ssim_loss"] + out["bpp_loss"]

        return out


class RateDistortionLoss_DebugUsed(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, metrics='mse'):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.metrics = metrics

    def set_lmbda(self, lmbda):
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W
        eps = 1e-5
        out["bpp_loss"] = sum(
            (torch.log(torch.clamp(likelihoods, eps, 1.0)).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        
        for likelihoods in output["likelihoods"].values():
            if torch.isnan(likelihoods).any():
                print("nan gradient found in likelihoods")
                
        if torch.isnan(out["bpp_loss"]).any():
                print("nan gradient found in bpp_loss")
        
        if torch.isnan(output["x_hat"]).any():
                print("nan gradient found in x_hat")
        
        if torch.isnan(target).any():
                print("nan gradient found in target")
        
        if self.metrics == 'mse':
            out["mse_loss"] = self.mse(output["x_hat"], target)
            if torch.isnan(out["mse_loss"]).any():
                print("nan gradient found in mse_loss")
            out["ms_ssim_loss"] = None
            out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]
            if torch.isnan(out["loss"]).any():
                print("nan gradient found in loss")
        elif self.metrics == 'ms-ssim':
            out["mse_loss"] = None
            out["ms_ssim_loss"] = 1 - ms_ssim(output["x_hat"], target, data_range=1.0)
            out["loss"] = self.lmbda * out["ms_ssim_loss"] + out["bpp_loss"]

        return out
