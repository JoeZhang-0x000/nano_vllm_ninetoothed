import triton
import triton.language as tl
import torch
from icecream import ic
from .registry import register_triton_op

@triton.jit
def _rms_forward_kernel(x_base, w_base, o_base, x_stride, o_stride,
                 eps: tl.constexpr,
                 BLOCK_SIZE: tl.constexpr
                 ):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_base + row_idx * x_stride + col_offsets)
    w = tl.load(w_base + col_offsets)
    x_fp32 = tl.cast(x, tl.float32)
    var = x_fp32 * x_fp32
    x = x * tl.rsqrt(tl.sum(var) / BLOCK_SIZE + eps) * w
    tl.store(o_base + row_idx * o_stride + col_offsets, x)


@register_triton_op
def rms_forward(x, weight, eps, inplace=False):
    n = x.shape[-1]
    m = x.numel() // n
    output = x
    if not inplace:
        output = torch.empty_like(x)
    assert weight.shape[-1] == n
    _rms_forward_kernel[(m, )](
        x, weight, output,
        x.stride(0),  output.stride(0),
        eps,
        n,
    )
    return output

