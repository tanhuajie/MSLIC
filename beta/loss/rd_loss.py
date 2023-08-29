import math
import torch
import torch.nn as nn
from pytorch_msssim import ms_ssim


class RateDistortionLoss_MS(nn.Module):
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

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        if self.metrics == 'mse':
            out["mse_loss_ds1"] = self.mse(output["x_hat"], target)
            out["mse_loss_ds2"] = self.mse(output["x_hat_ds2"], target)
            out["mse_loss_ds4"] = self.mse(output["x_hat_ds4"], target)
            # out["mse_loss"    ] = out["mse_loss_ds1"]
            out["mse_loss"    ] = (out["mse_loss_ds1"] + out["mse_loss_ds2"] * 0.80 + out["mse_loss_ds4"] * 0.64) / (1+0.80+0.64)
            out["ms_ssim_loss_ds1"] = None
            out["ms_ssim_loss_ds2"] = None
            out["ms_ssim_loss_ds4"] = None
            out["ms_ssim_loss"    ] = None
            out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]
        elif self.metrics == 'ms-ssim':
            out["mse_loss_ds1"] = None
            out["mse_loss_ds2"] = None
            out["mse_loss_ds4"] = None
            out["mse_loss"    ] = None
            out["ms_ssim_loss_ds1"] = 1 - ms_ssim(output["x_hat"], target, data_range=1.0)
            out["ms_ssim_loss_ds2"] = 1 - ms_ssim(output["x_hat_ds2"], target, data_range=1.0)
            out["ms_ssim_loss_ds4"] = 1 - ms_ssim(output["x_hat_ds4"], target, data_range=1.0)
            # out["ms_ssim_loss"    ] = out["ms_ssim_loss_ds1"]
            out["ms_ssim_loss"    ] = (out["ms_ssim_loss_ds1"] + out["ms_ssim_loss_ds2"] * 0.80 + out["ms_ssim_loss_ds4"] * 0.64) / (1+0.80+0.64)
            out["loss"] = self.lmbda * out["ms_ssim_loss"] + out["bpp_loss"]

        return out
