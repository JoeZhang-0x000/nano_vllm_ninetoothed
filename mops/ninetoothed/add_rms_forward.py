from ninetoothed import Tensor, make, block_size, Symbol
import ninetoothed.language as ntl
import torch
from mops.ninetoothed.config import *
from functools import lru_cache
from .registry import register_ninetoothed_op

BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)

def arrangement(input, residual, weight, output, eps,
                BLOCK_SIZE = BLOCK_SIZE,
                ):
    '''
    input: (..., N)
    residual: (..., N)
    weight: (..., N)
    output: (..., N)
    ->
    input_arranged: (..., 1)x(N)
    residual: (..., 1)x(N)
    weight: (..., 1)x(N)
    output: (..., 1)x(N)
    '''
    ndim = len(input.shape)
    arrange_shape = tuple(1 for _ in range(ndim-1)) + (BLOCK_SIZE,)
    expand_shape = tuple(input.shape[:-1]) + (-1,)

    def _squeeze(x):
        for _ in range(ndim-1):
              x.dtype = x.dtype.squeeze(0)
        return x 

    input_arranged = input.tile(arrange_shape)
    input_arranged = _squeeze(input_arranged)
    residual_arranged = residual.tile(arrange_shape)
    residual_arranged = _squeeze(residual_arranged)
    weight_arranged = weight.tile(arrange_shape).expand(expand_shape)
    weight_arranged = _squeeze(weight_arranged)

    output_arranged = output.tile(arrange_shape)
    output_arranged = _squeeze(output_arranged)

    return input_arranged, residual_arranged, weight_arranged, output_arranged, eps


def application(input, residual, weight, output, eps):
    input = ntl.cast(input, ntl.float32) + ntl.cast(residual, ntl.float32)

    residual = input
    input_square = input * input
    input_square_mean = ntl.sum(input_square) / input.shape[-1]

    output = input * ntl.rsqrt(input_square_mean + eps) * weight

@lru_cache(1)
def premake(ndim):
    kernel = make(arrangement, application, (Tensor(ndim), Tensor(ndim), Tensor(ndim), Tensor(ndim), Tensor(0)), max_num_configs=MAX_NUM_CONFIG)
    return kernel

@register_ninetoothed_op
def add_rms_forward(input, residual, weight, eps, inplace=False):
    # print('add_rms_forward', input.shape)
    assert weight.dim() == 1
    ndim = input.dim()
    weight = weight.view((1,)*(ndim-1) + (-1,))
    output = input
    if not inplace:
         output = torch.empty_like(input)
    premake(ndim)(input, residual, weight, output, eps, BLOCK_SIZE=input.shape[-1])
    return output, residual

