import triton
import triton.language as tl
import torch
from .registry import register_triton_op

@triton.jit
def _softmax(x_ptr, o_ptr, stride_xm, stride_om, BLOCK_SIZE:tl.constexpr):
    row = tl.program_id(0)
    col_offset = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + row * stride_om + col_offset)
    x_max = tl.max(x)
    x_sub_max = x - x_max
    x_exp = tl.exp(x_sub_max)
    x_exp_sum = tl.sum(x_exp)
    o = x_exp / x_exp_sum
    tl.store(o_ptr + row * stride_om + col_offset, o)

@register_triton_op
def softmax(x: torch.Tensor):
    out = torch.empty_like(x)
    _softmax[(x.shape[0],)](x, out, x.stride(0), out.stride(0), BLOCK_SIZE=x.shape[1])
    return out

