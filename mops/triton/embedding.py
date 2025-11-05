import triton
import triton.language as tl
import torch
from mops.triton.registry import register_triton_op

@triton.jit
def _embedding_kernel(x_ptr, w_ptr, o_ptr, stride_w, stride_o, M, H, BM:tl.constexpr, BH:tl.constexpr):
    pid = tl.program_id(0)
    m = pid * BM + tl.arange(0, BM)
    m_mask = m < M
    x = tl.load(x_ptr + m, mask=m_mask)
    for h_start in tl.range(0, H, BH):
        h = h_start + tl.arange(0, BH)
        h_mask = h < H
        mask_2d = m_mask[:, None] & h_mask[None, :]
        w_idx = w_ptr + x[:, None] * stride_w + h[None, :]
        o_idx = o_ptr + m[:, None] * stride_o + h[None, :]
        w = tl.load(w_idx, mask=mask_2d)
        tl.store(o_idx, w, mask=mask_2d)

@register_triton_op
def embedding(x: torch.Tensor, weight: torch.Tensor):
    '''
    x: (B, S)
    w: (E, H)
    o: (B, S, H)
    '''
    assert x.is_cuda
    assert x.ndim == 2 or x.ndim == 1
    if x.ndim == 2:
        B, S = x.shape
        E, H = weight.shape
        output = torch.empty(B, S, H, device=x.device, dtype=weight.dtype)

        _x = x.view(-1) # (BxS)
        _w = weight #(E, H)
        _o = output.view(-1, H) #(BxS, H)
        M = B*S
    if x.ndim == 1:
        M = x.shape[0]
        E, H = weight.shape
        output = torch.empty(M, H, device=x.device, dtype=weight.dtype)
        _x = x
        _w = weight #(E, H)
        _o = output #(BxS, H)
    BM = 32
    BH = 32
    grid = (M + BM - 1) // BM
    _embedding_kernel[(grid,)](_x, _w, _o, _w.stride(0), _o.stride(0), M, H, BM, BH)
    return output

