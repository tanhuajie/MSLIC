import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from compressai.models import CompressionModel
from compressai.ops import ste_round
from compressai.ans import BufferedRansEncoder, RansDecoder
from utils.func import update_registered_buffers, get_scale_table
from modules.transform import *
from modules.layers import *

class MSLIC_V4(CompressionModel):
    def __init__(self, config, **kwargs):
        super().__init__(config.N, **kwargs)
        '''
        # "N": 192,
        # "slice_ch": 32,
        # "slice_num": [8,8,8],
        # "act": nn.GELU
        '''
        slice_ch = config.slice_ch
        slice_num = config.slice_num
        N = config.N
        M = slice_ch*slice_num[0] + slice_ch*slice_num[1]//4 + slice_ch*slice_num[2]//16

        self.N = N
        self.M = M
        self.slice_ch = slice_ch
        self.slice_num = slice_num

        self.g_a = AnalysisTransform_MS (N=N, CH_S=slice_ch, CH_NUM=slice_num)
        self.g_s = SynthesisTransform_MS(N=N, CH_S=slice_ch, CH_NUM=slice_num)

        self.h_a = HyperAnalysis_MS (N=N, CH_S=slice_ch, CH_NUM=slice_num)
        self.h_s = HyperSynthesis_MS(N=N, CH_S=slice_ch, CH_NUM=slice_num)

        # Gussian Conditional
        self.gaussian_conditional = GaussianConditional(None)

        # # Global Inter Attention
        # self.global_inter_context_x1 = nn.ModuleList(
        #     LinearGlobalInterContext(dim=slice_ch * i, out_dim=slice_ch * 2, num_heads=slice_ch * i // 32) if i else None
        #     for i in range(slice_num[0])
        # )
        # self.global_inter_context_x2 = nn.ModuleList(
        #     LinearGlobalInterContext(dim=slice_ch * i, out_dim=slice_ch * 2, num_heads=slice_ch * i // 32) if i else None
        #     for i in range(slice_num[1])
        # )
        # self.global_inter_context_x4 = nn.ModuleList(
        #     LinearGlobalInterContext(dim=slice_ch * i, out_dim=slice_ch * 2, num_heads=slice_ch * i // 32) if i else None
        #     for i in range(slice_num[2])
        # )

        # Channel Context
        self.channel_context_x1 = nn.ModuleList(
            ChannelContext_MS(in_dim=slice_ch * i, out_dim=slice_ch * i) if i else None
            for i in range(slice_num[0])
        )
        self.channel_context_x2 = nn.ModuleList(
            ChannelContext_MS(in_dim=slice_ch * i, out_dim=slice_ch * i) if i else None
            for i in range(slice_num[1])
        )
        self.channel_context_x4 = nn.ModuleList(
            ChannelContext_MS(in_dim=slice_ch * i, out_dim=slice_ch * i) if i else None
            for i in range(slice_num[2])
        )

        # Multi-Scale Attention
        self.multi_scale_context_x2_to_x1 = nn.ModuleList(
            MultiScaleAttention(dim=slice_ch * i, ms_dim=slice_ch * slice_num[1], out_dim=slice_ch * 2, num_heads=slice_ch * i // 32) if i else None
            for i in range(slice_num[0])
        )
        self.multi_scale_context_x4_to_x1 = nn.ModuleList(
            MultiScaleAttention(dim=slice_ch * i, ms_dim=slice_ch * slice_num[2], out_dim=slice_ch * 2, num_heads=slice_ch * i // 32) if i else None
            for i in range(slice_num[0])
        )
        self.multi_scale_context_x4_to_x2 = nn.ModuleList(
            MultiScaleAttention(dim=slice_ch * i, ms_dim=slice_ch * slice_num[2], out_dim=slice_ch * 2, num_heads=slice_ch * i // 32) if i else None
            for i in range(slice_num[1])
        )

        # Entropy Parameters
        self.entropy_parameters_x1 = nn.ModuleList(
            EntropyParameters(in_dim=slice_ch * slice_num[0] * 2 + slice_ch * (i + 4), out_dim=slice_ch * 2)
            if i else EntropyParameters(in_dim=slice_ch * slice_num[0] * 2, out_dim=slice_ch * 2)
            for i in range(slice_num[0])
        )
        self.entropy_parameters_x2 = nn.ModuleList(
            EntropyParameters(in_dim=slice_ch * slice_num[1] * 2 + slice_ch * (i + 2), out_dim=slice_ch * 2)
            if i else EntropyParameters(in_dim=slice_ch * slice_num[1] * 2, out_dim=slice_ch * 2)
            for i in range(slice_num[1])
        )
        self.entropy_parameters_x4 = nn.ModuleList(
            EntropyParameters(in_dim=slice_ch * slice_num[2] * 2 + slice_ch * i, out_dim=slice_ch * 2)
            if i else EntropyParameters(in_dim=slice_ch * slice_num[2] * 2, out_dim=slice_ch * 2)
            for i in range(slice_num[2])
        )

        # Latent Residual Prediction
        self.lrp_transforms_x1 = nn.ModuleList(
            LatentResidualPrediction(in_dim=slice_ch * slice_num[0] + slice_ch * (i + 1), out_dim=slice_ch)
            for i in range(slice_num[0])
        )
        self.lrp_transforms_x2 = nn.ModuleList(
            LatentResidualPrediction(in_dim=slice_ch * slice_num[1] + slice_ch * (i + 1), out_dim=slice_ch)
            for i in range(slice_num[1])
        )
        self.lrp_transforms_x4 = nn.ModuleList(
            LatentResidualPrediction(in_dim=slice_ch * slice_num[2] + slice_ch * (i + 1), out_dim=slice_ch)
            for i in range(slice_num[2])
        )

        # Shuffel
        self.shuffel_x2 = nn.PixelShuffle(2)
        self.shuffel_x4 = nn.PixelShuffle(4)

    def forward(self, x):
        y1, y2, y4 = self.g_a(x)
        z = self.h_a(y1, y2, y4)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_hat = ste_round(z - z_offset) + z_offset

        # Hyper-parameters
        hyper_params_1, hyper_params_2, hyper_params_4 = self.h_s(z_hat)
        hyper_scales_1, hyper_means_1 = hyper_params_1.chunk(2, 1)
        hyper_scales_2, hyper_means_2 = hyper_params_2.chunk(2, 1)
        hyper_scales_4, hyper_means_4 = hyper_params_4.chunk(2, 1)

        y1_slices = y1.chunk(self.slice_num[0], dim=1)
        y2_slices = y2.chunk(self.slice_num[1], dim=1)
        y4_slices = y4.chunk(self.slice_num[2], dim=1)

        # Scale = 4
        y4_hat_slices = []
        y4_likelihood = []
        for slice_index, y_slice in enumerate(y4_slices):
            if slice_index == 0:
                params_support = hyper_params_4
            else:
                # global_attn = self.global_inter_context_x4[slice_index](torch.cat(y4_hat_slices, dim=1))
                channel_ctx = self.channel_context_x4[slice_index](torch.cat(y4_hat_slices, dim=1))
                params_support = torch.cat([hyper_params_4, channel_ctx], dim=1)
                # params_support = torch.cat([hyper_params_4, global_attn, channel_ctx], dim=1)

            params_entropy = self.entropy_parameters_x4[slice_index](params_support)
            scale, mu = params_entropy.chunk(2, 1)

            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y4_likelihood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_slice - mu) + mu

            lrp_support = torch.cat([hyper_means_4] + y4_hat_slices + [y_hat_slice], dim=1)
            lrp = self.lrp_transforms_x4[slice_index](lrp_support)
            y_hat_slice += lrp
            y4_hat_slices.append(y_hat_slice)

        # Scale = 2
        y2_hat_slices = []
        y2_likelihood = []
        for slice_index, y_slice in enumerate(y2_slices):
            if slice_index == 0:
                params_support = hyper_params_2
            else:
                scale_attn = self.multi_scale_context_x4_to_x2[slice_index](torch.cat(y2_hat_slices, dim=1), torch.cat(y4_hat_slices, dim=1))
                # global_attn = self.global_inter_context_x2[slice_index](torch.cat(y2_hat_slices, dim=1))
                channel_ctx = self.channel_context_x2[slice_index](torch.cat(y2_hat_slices, dim=1))
                params_support = torch.cat([hyper_params_2, scale_attn, channel_ctx], dim=1)
                # params_support = torch.cat([scale_attn, hyper_params_2, global_attn, channel_ctx], dim=1)

            params_entropy = self.entropy_parameters_x2[slice_index](params_support)
            scale, mu = params_entropy.chunk(2, 1)

            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y2_likelihood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_slice - mu) + mu

            lrp_support = torch.cat([hyper_means_2] + y2_hat_slices + [y_hat_slice], dim=1)
            lrp = self.lrp_transforms_x2[slice_index](lrp_support)
            y_hat_slice += lrp
            y2_hat_slices.append(y_hat_slice)
        
        # Scale = 1
        y1_hat_slices = []
        y1_likelihood = []
        for slice_index, y_slice in enumerate(y1_slices):
            if slice_index == 0:
                params_support = hyper_params_1
            else:
                scale_attn_1 = self.multi_scale_context_x4_to_x1[slice_index](torch.cat(y1_hat_slices, dim=1), torch.cat(y4_hat_slices, dim=1))
                scale_attn_2 = self.multi_scale_context_x2_to_x1[slice_index](torch.cat(y1_hat_slices, dim=1), torch.cat(y2_hat_slices, dim=1))
                # global_attn = self.global_inter_context_x1[slice_index](torch.cat(y1_hat_slices, dim=1))
                channel_ctx = self.channel_context_x1[slice_index](torch.cat(y1_hat_slices, dim=1))
                params_support = torch.cat([hyper_params_1, scale_attn_1, scale_attn_2, channel_ctx], dim=1)
                # params_support = torch.cat([hyper_params_1, scale_attn_1, scale_attn_2, global_attn, channel_ctx], dim=1)

            params_entropy = self.entropy_parameters_x1[slice_index](params_support)
            scale, mu = params_entropy.chunk(2, 1)

            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y1_likelihood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_slice - mu) + mu

            lrp_support = torch.cat([hyper_means_1] + y1_hat_slices + [y_hat_slice], dim=1)
            lrp = self.lrp_transforms_x1[slice_index](lrp_support)
            y_hat_slice += lrp
            y1_hat_slices.append(y_hat_slice)

        y1_hat = torch.cat(y1_hat_slices, dim=1)
        y2_hat = torch.cat(y2_hat_slices, dim=1)
        y4_hat = torch.cat(y4_hat_slices, dim=1)

        x_hat = self.g_s(y1_hat, y2_hat, y4_hat)

        y1_likelihoods = torch.cat(y1_likelihood, dim=1)
        y2_likelihoods = torch.cat(y2_likelihood, dim=1)
        y4_likelihoods = torch.cat(y4_likelihood, dim=1)
        
        y2_likelihoods = self.shuffel_x2(y2_likelihoods)
        y4_likelihoods = self.shuffel_x4(y4_likelihoods)

        y_likelihoods = torch.cat([y1_likelihoods, y2_likelihoods, y4_likelihoods], dim=1)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y_likelihoods": y_likelihoods, "z_likelihoods": z_likelihoods}
        }

    def compress(self, x):
        torch.cuda.synchronize()
        start_time = time.time()

        y1, y2, y4 = self.g_a(x)
        z = self.h_a(y1, y2, y4)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        # Hyper-parameters
        hyper_params_1, hyper_params_2, hyper_params_4 = self.h_s(z_hat)
        hyper_scales_1, hyper_means_1 = hyper_params_1.chunk(2, 1)
        hyper_scales_2, hyper_means_2 = hyper_params_2.chunk(2, 1)
        hyper_scales_4, hyper_means_4 = hyper_params_4.chunk(2, 1)

        y1_slices = y1.chunk(self.slice_num[0], dim=1)
        y2_slices = y2.chunk(self.slice_num[1], dim=1)
        y4_slices = y4.chunk(self.slice_num[2], dim=1)
        
        y1_hat_slices = []
        y2_hat_slices = []
        y4_hat_slices = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        encoder = BufferedRansEncoder()

        symbols_list = []
        indexes_list = []
        y_strings = []

        # Scale = 4
        for slice_index, y_slice in enumerate(y4_slices):
            if slice_index == 0:
                params_support = hyper_params_4
            else:
                # global_attn = self.global_inter_context_x4[slice_index](torch.cat(y4_hat_slices, dim=1))
                channel_ctx = self.channel_context_x4[slice_index](torch.cat(y4_hat_slices, dim=1))
                params_support = torch.cat([hyper_params_4, channel_ctx], dim=1)
                # params_support = torch.cat([hyper_params_4, global_attn, channel_ctx], dim=1)

            params_entropy = self.entropy_parameters_x4[slice_index](params_support)
            scale, mu = params_entropy.chunk(2, 1)

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            lrp_support = torch.cat([hyper_means_4] + y4_hat_slices + [y_hat_slice], dim=1)
            lrp = self.lrp_transforms_x4[slice_index](lrp_support)
            y_hat_slice += lrp
            y4_hat_slices.append(y_hat_slice)

        # Scale = 2
        for slice_index, y_slice in enumerate(y2_slices):
            if slice_index == 0:
                params_support = hyper_params_2
            else:
                scale_attn = self.multi_scale_context_x4_to_x2[slice_index](torch.cat(y2_hat_slices, dim=1), torch.cat(y4_hat_slices, dim=1))
                # global_attn = self.global_inter_context_x2[slice_index](torch.cat(y2_hat_slices, dim=1))
                channel_ctx = self.channel_context_x2[slice_index](torch.cat(y2_hat_slices, dim=1))
                params_support = torch.cat([hyper_params_2, scale_attn, channel_ctx], dim=1)
                # params_support = torch.cat([scale_attn, hyper_params_2, global_attn, channel_ctx], dim=1)

            params_entropy = self.entropy_parameters_x2[slice_index](params_support)
            scale, mu = params_entropy.chunk(2, 1)

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            lrp_support = torch.cat([hyper_means_2] + y2_hat_slices + [y_hat_slice], dim=1)
            lrp = self.lrp_transforms_x2[slice_index](lrp_support)
            y_hat_slice += lrp
            y2_hat_slices.append(y_hat_slice)

        # Scale = 1
        for slice_index, y_slice in enumerate(y1_slices):
            if slice_index == 0:
                params_support = hyper_params_1
            else:
                scale_attn_1 = self.multi_scale_context_x4_to_x1[slice_index](torch.cat(y1_hat_slices, dim=1), torch.cat(y4_hat_slices, dim=1))
                scale_attn_2 = self.multi_scale_context_x2_to_x1[slice_index](torch.cat(y1_hat_slices, dim=1), torch.cat(y2_hat_slices, dim=1))
                # global_attn = self.global_inter_context_x1[slice_index](torch.cat(y1_hat_slices, dim=1))
                channel_ctx = self.channel_context_x1[slice_index](torch.cat(y1_hat_slices, dim=1))
                params_support = torch.cat([hyper_params_1, scale_attn_1, scale_attn_2, channel_ctx], dim=1)
                # params_support = torch.cat([hyper_params_1, scale_attn_1, scale_attn_2, global_attn, channel_ctx], dim=1)

            params_entropy = self.entropy_parameters_x1[slice_index](params_support)
            scale, mu = params_entropy.chunk(2, 1)

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            lrp_support = torch.cat([hyper_means_1] + y1_hat_slices + [y_hat_slice], dim=1)
            lrp = self.lrp_transforms_x1[slice_index](lrp_support)
            y_hat_slice += lrp
            y1_hat_slices.append(y_hat_slice)

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
        
        # Hyper-parameters
        hyper_params_1, hyper_params_2, hyper_params_4 = self.h_s(z_hat)
        hyper_scales_1, hyper_means_1 = hyper_params_1.chunk(2, 1)
        hyper_scales_2, hyper_means_2 = hyper_params_2.chunk(2, 1)
        hyper_scales_4, hyper_means_4 = hyper_params_4.chunk(2, 1)
        
        y1_hat_slices = []
        y2_hat_slices = []
        y4_hat_slices = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        decoder = RansDecoder()
        decoder.set_stream(y_strings)

        # Scale = 4
        for slice_index in range(self.slice_num[2]):
            if slice_index == 0:
                params_support = hyper_params_4
            else:
                # global_attn = self.global_inter_context_x4[slice_index](torch.cat(y4_hat_slices, dim=1))
                channel_ctx = self.channel_context_x4[slice_index](torch.cat(y4_hat_slices, dim=1))
                params_support = torch.cat([hyper_params_4, channel_ctx], dim=1)
                # params_support = torch.cat([hyper_params_4, global_attn, channel_ctx], dim=1)

            params_entropy = self.entropy_parameters_x4[slice_index](params_support)
            scale, mu = params_entropy.chunk(2, 1)

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0]//4, y_shape[1]//4)
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp_support = torch.cat([hyper_means_4] + y4_hat_slices + [y_hat_slice], dim=1)
            lrp = self.lrp_transforms_x4[slice_index](lrp_support)
            y_hat_slice += lrp
            y4_hat_slices.append(y_hat_slice)

        # Scale = 2
        for slice_index in range(self.slice_num[1]):
            if slice_index == 0:
                params_support = hyper_params_2
            else:
                scale_attn = self.multi_scale_context_x4_to_x2[slice_index](torch.cat(y2_hat_slices, dim=1), torch.cat(y4_hat_slices, dim=1))
                # global_attn = self.global_inter_context_x2[slice_index](torch.cat(y2_hat_slices, dim=1))
                channel_ctx = self.channel_context_x2[slice_index](torch.cat(y2_hat_slices, dim=1))
                params_support = torch.cat([hyper_params_2, scale_attn, channel_ctx], dim=1)
                # params_support = torch.cat([scale_attn, hyper_params_2, global_attn, channel_ctx], dim=1)

            params_entropy = self.entropy_parameters_x2[slice_index](params_support)
            scale, mu = params_entropy.chunk(2, 1)

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0]//2, y_shape[1]//2)
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp_support = torch.cat([hyper_means_2] + y2_hat_slices + [y_hat_slice], dim=1)
            lrp = self.lrp_transforms_x2[slice_index](lrp_support)
            y_hat_slice += lrp
            y2_hat_slices.append(y_hat_slice)

        # Scale = 1
        for slice_index in range(self.slice_num[0]):
            if slice_index == 0:
                params_support = hyper_params_1
            else:
                scale_attn_1 = self.multi_scale_context_x4_to_x1[slice_index](torch.cat(y1_hat_slices, dim=1), torch.cat(y4_hat_slices, dim=1))
                scale_attn_2 = self.multi_scale_context_x2_to_x1[slice_index](torch.cat(y1_hat_slices, dim=1), torch.cat(y2_hat_slices, dim=1))
                # global_attn = self.global_inter_context_x1[slice_index](torch.cat(y1_hat_slices, dim=1))
                channel_ctx = self.channel_context_x1[slice_index](torch.cat(y1_hat_slices, dim=1))
                params_support = torch.cat([hyper_params_1, scale_attn_1, scale_attn_2, channel_ctx], dim=1)
                # params_support = torch.cat([hyper_params_1, scale_attn_1, scale_attn_2, global_attn, channel_ctx], dim=1)

            params_entropy = self.entropy_parameters_x1[slice_index](params_support)
            scale, mu = params_entropy.chunk(2, 1)

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp_support = torch.cat([hyper_means_1] + y1_hat_slices + [y_hat_slice], dim=1)
            lrp = self.lrp_transforms_x1[slice_index](lrp_support)
            y_hat_slice += lrp
            y1_hat_slices.append(y_hat_slice)

        y1_hat = torch.cat(y1_hat_slices, dim=1)
        y2_hat = torch.cat(y2_hat_slices, dim=1)
        y4_hat = torch.cat(y4_hat_slices, dim=1)

        x_hat = self.g_s(y1_hat, y2_hat, y4_hat)

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
