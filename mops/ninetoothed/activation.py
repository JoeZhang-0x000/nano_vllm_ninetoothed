from mops.ninetoothed.registry import register_ninetoothed_op
from ninetoothed import Tensor, make, Symbol, block_size
import ninetoothed.language as ntl
import torch
from mops.ninetoothed.config import *
from functools import lru_cache

BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)

def arrangement(x, y, out, BLOCK_SIZE=BLOCK_SIZE):
    ndim = x.ndim
    x_arranged = x.tile((1,)*(ndim-1) + (BLOCK_SIZE,))
    y_arranged = y.tile((1,)*(ndim-1) + (BLOCK_SIZE,))
    output_arranged = out.tile((1,)*(ndim-1) + (BLOCK_SIZE,))
    return x_arranged, y_arranged, output_arranged

def application(x, y, out):
    x_fp32 = ntl.cast(x, ntl.float32)
    y_fp32 = ntl.cast(y, ntl.float32)
    out = x_fp32 * (1 / (1 + ntl.exp(-x_fp32))) * y_fp32


@lru_cache(1)
def _premake(ndim: int):
    kernel = make(arrangement, application, (Tensor(ndim), Tensor(ndim), Tensor(ndim)), max_num_configs=MAX_NUM_CONFIG)
    return kernel

@register_ninetoothed_op
def siluAndMul(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    _premake(x.ndim)(x, y, output, BLOCK_SIZE=x.shape[-1])
    return output