import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from compressai.models import CompressionModel
from compressai.ops import ste_round
from compressai.ans import BufferedRansEncoder, RansDecoder
from utils.func import update_registered_buffers, get_scale_table
from modules.transform import *


class BaseLine(CompressionModel):
    def __init__(self, config, **kwargs):
        super().__init__(config.N, **kwargs)
        N = config.N
        M = config.M
        
        slice_num = config.slice_num
        slice_ch = M // slice_num
        assert slice_ch * slice_num == M

        self.N = N
        self.M = M
        
        self.slice_num = slice_num
        self.slice_ch = slice_ch

        self.g_a = AnalysisTransform(N=N, M=M)
        self.g_s = SynthesisTransform(N=N, M=M)

        self.h_a = HyperAnalysis(M=M, N=N)
        self.h_s = HyperSynthesis(M=M, N=N)

        # Gussian Conditional
        self.gaussian_conditional = GaussianConditional(None)

        self.entropy_parameters = nn.ModuleList(
            EntropyParameters(in_dim=self.M * 2 + self.slice_ch * i, out_dim=self.slice_ch * 2)
            for i in range(self.slice_num)
        )

        self.lrp_transforms = nn.ModuleList(
            LatentResidualPrediction(in_dim=self.M + (i + 1) * self.slice_ch, out_dim=self.slice_ch)
            for i in range(self.slice_num)
        )

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_hat = ste_round(z - z_offset) + z_offset

        # Hyper-parameters
        hyper_params = self.h_s(z_hat)
        hyper_scales, hyper_means = hyper_params.chunk(2, 1)

        y_slices = y.chunk(self.slice_num, dim=1)
        y_hat_slices = []
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):

            params_support = torch.cat([hyper_params] + y_hat_slices, dim=1)
            params_entropy = self.entropy_parameters[slice_index](params_support)
            scale, mu = params_entropy.chunk(2, 1)

            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_slice - mu) + mu

            lrp_support = torch.cat([hyper_means] + y_hat_slices + [y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            y_hat_slice += lrp
            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y_likelihoods": y_likelihoods, "z_likelihoods": z_likelihoods}
        }

    def compress(self, x):
        torch.cuda.synchronize()
        start_time = time.time()
        
        y = self.g_a(x)
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        hyper_params = self.h_s(z_hat)
        hyper_scales, hyper_means = hyper_params.chunk(2, 1)
        y_slices = y.chunk(self.slice_num, dim=1)
        
        y_hat_slices = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):

            params_support = torch.cat([hyper_params] + y_hat_slices, dim=1)
            params_entropy = self.entropy_parameters[slice_index](params_support)
            scale, mu = params_entropy.chunk(2, 1)

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            lrp_support = torch.cat([hyper_means] + y_hat_slices + [y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)
        torch.cuda.synchronize()
        end_time = time.time()

        cost_time = end_time - start_time
        return {
            "strings": [y_strings, z_strings],
            "shape": z.size()[-2:],
            "cost_time": cost_time
        }

    def decompress(self, strings, shape):
        torch.cuda.synchronize()
        start_time = time.time()
        y_strings = strings[0][0]
        z_strings = strings[1]
        z_hat = self.entropy_bottleneck.decompress(z_strings, shape)
        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]
        
        hyper_params = self.h_s(z_hat)
        hyper_scales, hyper_means = hyper_params.chunk(2, 1)
        y_hat_slices = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        decoder = RansDecoder()
        decoder.set_stream(y_strings)

        for slice_index in range(self.slice_num):

            params_support = torch.cat([hyper_params] + y_hat_slices, dim=1)
            params_entropy = self.entropy_parameters[slice_index](params_support)
            scale, mu = params_entropy.chunk(2, 1)

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp_support = torch.cat([hyper_means] + y_hat_slices + [y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            y_hat_slice += lrp
            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat)
        torch.cuda.synchronize()
        end_time = time.time()

        cost_time = end_time - start_time

        return {
            "x_hat": x_hat,
            "cost_time": cost_time
        }

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated
