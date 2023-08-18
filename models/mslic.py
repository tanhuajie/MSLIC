import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from layers.extractor import MultiScaleExtractor, MultiScaleCombiner
from transform.analysis import AnalysisTransform_L1, AnalysisTransform_L2, AnalysisTransform_L3, HyperAnalysis
from transform.synthesis import SynthesisTransform_L1, SynthesisTransform_L2, SynthesisTransform_L3, HyperSynthesis
from transform.entropy import EntropyParameters, LatentResidualPrediction

from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.ops import ste_round
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.models import CompressionModel

from utils.func import update_registered_buffers, get_scale_table

class MSLIC(CompressionModel):
    
    def __init__(self, N=192, M=336, num_slices=7, **kwargs):
        super().__init__(entropy_bottleneck_channels=N)
        self.N = int(N)
        self.M = int(M)
        self.slice_num = num_slices
        self.slice_ch = M // num_slices

        self.ms_extractor = MultiScaleExtractor(dim = 48, num_blocks = [4,6,6,8], heads = [1,2,4,8])
        self.ms_combiner  = MultiScaleCombiner (dim = 48, num_blocks = [4,6,6,8], heads = [1,2,4,8])

        self.g_a_level_1 = AnalysisTransform_L1(InDim=96 , N=144, M=192)
        self.g_a_level_2 = AnalysisTransform_L2(InDim=96 , N=96 , M=96 )
        self.g_a_level_3 = AnalysisTransform_L3(InDim=192, N=96 , M=48 )

        self.g_s_level_1 = SynthesisTransform_L1(OutDim=96 , N=144, M=192)
        self.g_s_level_2 = SynthesisTransform_L2(OutDim=96 , N=96 , M=96 )
        self.g_s_level_3 = SynthesisTransform_L3(OutDim=192, N=96 , M=48 )

        self.h_a = HyperAnalysis (M=self.M, N=self.N)
        self.h_s = HyperSynthesis(M=self.M, N=self.N)

        self.entropy_means = nn.ModuleList(
            EntropyParameters(in_dim=self.M + self.slice_ch * i, out_dim=self.slice_ch)
            for i in range(self.slice_num)
        )

        self.entropy_scales = nn.ModuleList(
            EntropyParameters(in_dim=self.M + self.slice_ch * i, out_dim=self.slice_ch)
            for i in range(self.slice_num)
        )

        self.lrp_transforms = nn.ModuleList(
            LatentResidualPrediction(in_dim=self.M + (i + 1) * self.slice_ch, out_dim=self.slice_ch)
            for i in range(self.slice_num)
        )

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)
    
    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def _likelihood(self, inputs, scales, means=None):
        half = float(0.5)
        if means is not None:
            values = inputs - means
        else:
            values = inputs

        scales = torch.max(scales, torch.tensor(0.11))
        values = torch.abs(values)
        upper = self._standardized_cumulative((half - values) / scales)
        lower = self._standardized_cumulative((-half - values) / scales)
        likelihood = upper - lower
        return likelihood

    def _standardized_cumulative(self, inputs):
        half = float(0.5)
        const = float(-(2 ** -0.5))
        # Using the complementary error function maximizes numerical precision.
        return half * torch.erfc(const * inputs)

    def forward(self, x):
        # Multi-Scales-Extractor
        ms_l1, ms_l2, ms_l3 = self.ms_extractor(x)

        y_ms_l1 = self.g_a_level_1(ms_l1)
        y_ms_l2 = self.g_a_level_2(ms_l2)
        y_ms_l3 = self.g_a_level_3(ms_l3)
        y = torch.cat([y_ms_l3, y_ms_l2, y_ms_l1], dim=1)

        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        # Hyper-Parameters
        hyper_params = self.h_s(z_hat)
        hyper_scales, hyper_means = hyper_params.chunk(2, 1)

        y_slices = y.chunk(self.slice_num, 1)
        y_hat_slices = []
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = y_hat_slices
            mean_support = torch.cat([hyper_means] + support_slices, dim=1)
            mu = self.entropy_means[slice_index](mean_support)

            scale_support = torch.cat([hyper_scales] + support_slices, dim=1)
            scale = self.entropy_means[slice_index](scale_support)

            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_slice - mu) + mu

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            y_hat_slice += lrp
            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)

        # Multi-Scales-Extractor
        y_hat_ms = torch.split(y_hat, [48,96,192], dim=1)

        x_hat_ms_l3 = self.g_s_level_3(y_hat_ms[0])
        x_hat_ms_l2 = self.g_s_level_2(y_hat_ms[1])
        x_hat_ms_l1 = self.g_s_level_1(y_hat_ms[2])

        x_hat = self.ms_combiner(x_hat_ms_l1, x_hat_ms_l2, x_hat_ms_l3)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }
    
    def compress(self, x):
        torch.cuda.synchronize()
        start_time = time.time()

        # Multi-Scales-Extractor
        ms_l1, ms_l2, ms_l3 = self.ms_extractor(x)

        y_ms_l1 = self.g_a_level_1(ms_l1)
        y_ms_l2 = self.g_a_level_2(ms_l2)
        y_ms_l3 = self.g_a_level_3(ms_l3)
        y = torch.cat([y_ms_l3, y_ms_l2, y_ms_l1], dim=1)

        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        # Hyper-Parameters
        hyper_params = self.h_s(z_hat)
        hyper_scales, hyper_means = hyper_params.chunk(2, 1)

        y_slices = y.chunk(self.slice_num, 1)

        y_hat_slices = []
        y_scales = []
        y_means = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = y_hat_slices
            mean_support = torch.cat([hyper_means] + support_slices, dim=1)
            mu = self.entropy_means[slice_index](mean_support)

            scale_support = torch.cat([hyper_scales] + support_slices, dim=1)
            scale = self.entropy_means[slice_index](scale_support)

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
            y_scales.append(scale)
            y_means.append(mu)

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

        # Hyper-Parameters
        hyper_params = self.h_s(z_hat)
        hyper_scales, hyper_means = hyper_params.chunk(2, 1)
        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]
        y_hat_slices = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_strings)

        for slice_index in range(self.slice_num):
            support_slices = y_hat_slices
            mean_support = torch.cat([hyper_means] + support_slices, dim=1)
            mu = self.entropy_means[slice_index](mean_support)

            scale_support = torch.cat([hyper_scales] + support_slices, dim=1)
            scale = self.entropy_means[slice_index](scale_support)

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            y_hat_slice += lrp
            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)

        # Multi-Scales-Extractor
        y_hat_ms = torch.split(y_hat, [48,96,192], dim=1)

        x_hat_ms_l3 = self.g_s_level_3(y_hat_ms[0])
        x_hat_ms_l2 = self.g_s_level_2(y_hat_ms[1])
        x_hat_ms_l1 = self.g_s_level_1(y_hat_ms[2])

        x_hat = self.ms_combiner(x_hat_ms_l1, x_hat_ms_l2, x_hat_ms_l3)
        x_hat = x_hat.clamp_(0, 1)

        torch.cuda.synchronize()
        end_time = time.time()

        cost_time = end_time - start_time

        return {
            "x_hat": x_hat, 
            "cost_time": cost_time
        }
