import triton
import triton.language as tl
import torch
from mops.triton.registry import register_triton_op
import math

@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


@register_triton_op
def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)



@triton.jit
def _flash_varlen_main_kernel(
    Q, K, V, O,
    cu_seqlens_q, cu_seqlens_k,
    stride_qt, stride_qh, stride_qd,
    stride_kt, stride_kh, stride_kd,
    stride_vt, stride_vh, stride_vd,
    stride_ot, stride_oh, stride_od,
    sm_scale,
    D_HEAD: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    start_seq_q = tl.load(cu_seqlens_q + pid_b)
    len_seq_q = tl.load(cu_seqlens_q + pid_b + 1) - start_seq_q

    start_seq_k = tl.load(cu_seqlens_k + pid_b)
    len_seq_k = tl.load(cu_seqlens_k + pid_b + 1) - start_seq_k

    offs_m = start_seq_q + pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_d = tl.arange(0, D_HEAD)

    q_ptrs = Q + offs_m[:, None] * stride_qt + pid_h * stride_qh + offs_d[None, :] * stride_qd
    q_mask = offs_m[:, None] < start_seq_q + len_seq_q
    q_i = tl.load(q_ptrs, mask=q_mask)
    m_i = tl.full((BLOCK_SIZE_M, ), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_SIZE_M, ), dtype=tl.float32)
    o_i = tl.zeros((BLOCK_SIZE_M, D_HEAD), dtype=tl.float32)

    for start_n_local in range(0, len_seq_k, BLOCK_SIZE_N):
        offs_n = start_seq_k + start_n_local + tl.arange(0, BLOCK_SIZE_N)
        kv_mask = offs_n[:, None] < (len_seq_k + start_seq_k)
        k_ptrs = K + offs_n[:, None] * stride_kt + pid_h * stride_kh + offs_d[None, :] * stride_kd
        v_ptrs = V + offs_n[:, None] * stride_vt + pid_h * stride_vh + offs_d[None, :] * stride_vd
        k_j = tl.load(k_ptrs, mask=kv_mask)
        v_j = tl.load(v_ptrs, mask=kv_mask)

        s_ij = tl.dot(q_i.to(tl.float32), tl.trans(k_j).to(tl.float32)) * sm_scale

        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            s_ij = tl.where(causal_mask, s_ij, float("-inf"))

        m_ij = tl.max(s_ij, axis=1)
        m_i_new = tl.maximum(m_ij, m_i)
        p_ij = tl.exp(s_ij - m_i_new[:, None])
        l_ij = tl.sum(p_ij, axis=1)

        exp_diff = tl.exp(m_i - m_i_new)
        l_i_new = l_i  * exp_diff + l_ij

        o_i = o_i * (l_i / l_i_new * exp_diff)[:, None] + tl.dot(p_ij.to(tl.float32), v_j.to(tl.float32)) / l_i_new[:, None]

        m_i = m_i_new
        l_i = l_i_new
    o_ptrs = O + offs_m[:, None] * stride_ot + pid_h * stride_oh + offs_d[None, :] * stride_od
    tl.store(o_ptrs, o_i.to(tl.float16), mask=q_mask)


@register_triton_op
def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, softmax_scale=None, causal=True, **kwargs):
    '''
    layout: thd
    '''
    batch = cu_seqlens_q.numel() - 1
    head_q = q.shape[1]
    head_k = k.shape[1]
    head_dim = q.shape[2]

    num_rep = head_q // head_k
    toekns, _, _ = k.shape
    # t h d -> t h 1 d -> t h num_rep d -> t h*num_rep d
    k = k.unsqueeze(2).expand((-1, -1, num_rep, -1)).reshape(toekns, -1, head_dim)
    v = v.unsqueeze(2).expand((-1, -1, num_rep, -1)).reshape(toekns, -1, head_dim)

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 32

    def strides(x):
        return x.stride(0), x.stride(1), x.stride(2)

    grid = (
        triton.cdiv(max_seqlen_q, BLOCK_SIZE_M),
        batch,
        head_q
    )

    softmax_scale = 1 / math.sqrt(head_dim) if softmax_scale is None else softmax_scale

    o = torch.empty_like(q)
    _flash_varlen_main_kernel[grid](
        q, k, v, o,
        cu_seqlens_q, cu_seqlens_k,
        *strides(q), *strides(k), *strides(v), *strides(o),
        softmax_scale,
        head_dim,
        BLOCK_SIZE_M, BLOCK_SIZE_N,
        causal
    )

    return o







