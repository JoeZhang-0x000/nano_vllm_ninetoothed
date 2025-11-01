import triton
import triton.language as tl
import torch
from icecream import ic
from .registry import register_triton_op

@triton.jit
def _add_rms_forward_kernel(x_base, r_base, w_base, o_base,
                 eps: tl.constexpr,
                 BLOCK_SIZE: tl.constexpr
                 ):
    '''
    x: [m, n]
    r: [m, n]
    w: [n,]
    o: [m, n]
    '''
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_base + row_idx * BLOCK_SIZE + col_offsets)
    r = tl.load(r_base + row_idx * BLOCK_SIZE + col_offsets)
    w = tl.load(w_base + col_offsets)
    x_fp32 = tl.cast(x, tl.float32)
    r_fp32 = tl.cast(r, tl.float32)
    x = x_fp32 + r_fp32
    tl.store(r_base + row_idx * BLOCK_SIZE + col_offsets, x)
    var = x * x
    x = x * tl.rsqrt(tl.sum(var) / BLOCK_SIZE + eps) * w
    tl.store(o_base + row_idx * BLOCK_SIZE + col_offsets, x)


@register_triton_op
def add_rms_forward(x, residual, weight, eps, inplace=False):
    n = x.shape[-1]
    m = x.numel() // n
    output = x
    if not inplace:
        output = torch.empty_like(x)
    assert weight.shape[-1] == n
    _add_rms_forward_kernel[(m, )](
        x, residual, weight, output,
        eps,
        n,
    )
    return output, residual
