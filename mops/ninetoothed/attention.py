from ninetoothed import Tensor, Symbol, make, block_size
import ninetoothed.language as ntl
import torch
from functools import lru_cache
from mops.ninetoothed.registry import register_ninetoothed_op
import math


BLOCK_SIZE_M = block_size()
BLOCK_SIZE_N = block_size()

def arrangement(
    q, k, v, scale, o, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N
):
    def arrange_q_or_o(input):
        arranged = input.tile((1, 1, BLOCK_SIZE_M, -1))
        arranged.dtype = arranged.dtype.squeeze((0, 1))

        return arranged

    def arrange_k_or_v(input):
        arranged = input.tile((1, 1, BLOCK_SIZE_N, -1))
        arranged = arranged.tile((1, 1, -1, -1))
        arranged = arranged.expand((-1, -1, q_arranged.shape[-2], -1))
        arranged.dtype = arranged.dtype.squeeze((0, 1, 3))
        arranged.dtype.dtype = arranged.dtype.dtype.squeeze((0, 1))

        return arranged

    q_arranged = arrange_q_or_o(q)

    return q_arranged, arrange_k_or_v(k), arrange_k_or_v(v), scale, arrange_q_or_o(o)

def application(q, k, v, scale, o):
    q_loaded = (q * scale * 1.44269504089).to(q.dtype)

    acc = ntl.zeros((q.shape[-2], q.shape[-1]), dtype=ntl.float32)
    l_i = ntl.full((q.shape[-2],), 1, dtype=ntl.float32)
    m_i = ntl.full((q.shape[-2],), float("-inf"), dtype=ntl.float32)

    for i in range(k.shape[0]):
        qk = ntl.dot(q_loaded, ntl.trans(k[i]))
        qk = ntl.where(k[i].offsets(-2) < k.source.shape[-2], qk, float("-inf"))
        # causal_mask = q.offsets(-2)[:, None] > v[i].offsets(-1)[None, :]
        # qk = ntl.where(causal_mask, qk, float("-inf"))
        m_ij = ntl.maximum(m_i, ntl.max(qk, 1))
        p = ntl.exp2(qk - m_ij[:, None])
        l_ij = ntl.sum(p, 1)

        alpha = ntl.exp2(m_i - m_ij)
        acc = acc * alpha[:, None] + ntl.dot(p.to(v.dtype.dtype), v[i])
        m_i = m_ij
        l_i = l_i * alpha + l_ij

    acc /= l_i[:, None]
    o = acc.to(o.dtype)  # noqa: F841

shape_options = (None, None, None, {"constexpr": True, "upper_bound": 128})
q, k, v, o = (Tensor(4, jagged_dim=2, shape_options=shape_options) for _ in range(4))
tensors = (q, k, v, Tensor(0), o)

kernel = make(arrangement, application, tensors, max_num_configs=2)

def _flash_attn_varlen_func(q, k, v, scale=None):
    if scale is None:
        scale = 1 / math.sqrt(q.shape[-1])

    o = torch.nested.nested_tensor_from_jagged(
        torch.empty_like(q.values()), q.offsets(), jagged_dim=2
    )

    kernel(q, k, v, scale, o)

    return o

@register_ninetoothed_op
def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, softmax_scale=None, causal=True, **kwargs):
    batch = cu_seqlens_q.numel() - 1
    head_q = q.shape[1]
    head_k = k.shape[1]
    head_dim = q.shape[2]

    num_rep = head_q // head_k
    toekns, _, _ = k.shape
    # t h d -> t h 1 d -> t h num_rep d -> t h*num_rep d
    k = k.unsqueeze(2).expand((-1, -1, num_rep, -1)).reshape(toekns, -1, head_dim)
    v = v.unsqueeze(2).expand((-1, -1, num_rep, -1)).reshape(toekns, -1, head_dim)

    # T H D -> H T D
    q = q.permute(1, 0, 2)
    k = k.permute(1, 0, 2)
    v = v.permute(1, 0, 2)

    q = torch.nested.nested_tensor_from_jagged(
        values=q, offsets=cu_seqlens_q, jagged_dim=2
    )

    k = torch.nested.nested_tensor_from_jagged(
        values=k, offsets=cu_seqlens_k, jagged_dim=2
    )

    v = torch.nested.nested_tensor_from_jagged(
        values=v, offsets=cu_seqlens_k, jagged_dim=2
    )

    o = _flash_attn_varlen_func(q, k, v, scale=softmax_scale) # (B, H, j, D)

    o = o._values

    # o = list(o) # (B, H, D)

    # o = torch.cat(o, dim=0) # (BS, H, D)

    o = o.permute((1, 0, 2))


    return o




