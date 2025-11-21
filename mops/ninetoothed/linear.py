import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size
import torch
import math
from icecream import ic
from mops.ninetoothed.config import MAX_NUM_CONFIG, STATIC_MODE
from mops.ninetoothed.registry import register_ninetoothed_op
from functools import lru_cache

class Linear:
    if STATIC_MODE:
        BLOCK_SIZE_M = 32
        BLOCK_SIZE_N = 32
        BLOCK_SIZE_K = 128
    else:
        BLOCK_SIZE_M = block_size()
        BLOCK_SIZE_N = block_size()
        BLOCK_SIZE_K = block_size()

    def arrangement(
        input,
        other,
        output,
        block_size_m=None,
        block_size_n=None,
        block_size_k=None,
    ):
        if block_size_m is None:
            block_size_m = Linear.BLOCK_SIZE_M

        if block_size_n is None:
            block_size_n = Linear.BLOCK_SIZE_N

        if block_size_k is None:
            block_size_k = Linear.BLOCK_SIZE_K

        output_arranged = output.tile((block_size_m, block_size_n))

        input_arranged = input.tile((block_size_m, block_size_k))
        input_arranged = input_arranged.tile((1, -1))
        input_arranged = input_arranged.expand((-1, output_arranged.shape[1]))
        input_arranged.dtype = input_arranged.dtype.squeeze(0)

        other = other.permute((1, 0))
        other_arranged = other.tile((block_size_k, block_size_n))
        other_arranged = other_arranged.tile((-1, 1))
        other_arranged = other_arranged.expand((output_arranged.shape[0], -1))
        other_arranged.dtype = other_arranged.dtype.squeeze(1)

        return input_arranged, other_arranged, output_arranged

    def arrangement_with_bias(
        input,
        other,
        output,
        bias,
        block_size_m=None,
        block_size_n=None,
        block_size_k=None,
    ):
        if block_size_m is None:
            block_size_m = Linear.BLOCK_SIZE_M

        if block_size_n is None:
            block_size_n = Linear.BLOCK_SIZE_N

        if block_size_k is None:
            block_size_k = Linear.BLOCK_SIZE_K

        input_arranged, other_arranged, output_arranged = Linear.arrangement(
            input,
            other,
            output,
            block_size_m,
            block_size_n,
            block_size_k,
        )

        # bias (1, N)
        bias_arranged = bias.tile((1, block_size_n)) # (1, N/BN) x (1, BN)
        bias_arranged = bias_arranged.expand((output_arranged.shape[0], -1)) # (M/BM, N/BN) x (1, BN)
        return input_arranged, other_arranged, output_arranged, bias_arranged

    def application(input, other, output):
        accumulator = ntl.zeros(output.shape, dtype=ntl.float32)
        for k in range(input.shape[0]):
            accumulator += ntl.dot(input[k], other[k])
        output = accumulator

    def application_with_bias(input, other, output, bias):
        Linear.application(input, other, output)
        output = output + bias

    @lru_cache(1)
    def premake(bias=None):
        if bias is None:
            return ninetoothed.make(Linear.arrangement, Linear.application, (Tensor(2) for _ in range(3)), max_num_configs=MAX_NUM_CONFIG)
        else:
            return ninetoothed.make(Linear.arrangement_with_bias, Linear.application_with_bias, (Tensor(2) for _ in range(4)), max_num_configs=MAX_NUM_CONFIG)

    def apply(input, other, bias=None):
        assert input.shape[1] == other.shape[1], "Inner dimension K must match for NT GEMM"
        output_shape = (input.shape[0], other.shape[0])
        output = torch.empty(output_shape, dtype=input.dtype, device=input.device)

        kernel = Linear.premake(bias)
        kernel(input, other, output)

        return output

@register_ninetoothed_op
def linear(input, other, bias=None):
    # A: [M, K], B: [N, K]
    # C = A @ B.T -> C: [M, N]
    return Linear.apply(input, other, bias)