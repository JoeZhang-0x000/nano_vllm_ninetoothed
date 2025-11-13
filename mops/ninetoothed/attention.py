from ninetoothed import Tensor, Symbol, make
import ninetoothed.language as ntl
import torch
from functools import lru_cache
from mops.ninetoothed.registry import register_ninetoothed_op


# def arrangement(q, k, v, o, sm_scale):
#     '''
#     qo: B N SQ D
#     kv: B N SK D
#     ->
#     qo: (B, N, 1, 1)x(1, 1, SQ_BM, 1)x(1, 1, BM, D)
#     kv: (B, N, 1, 1)x(1, 1, SK_BN, 1)x(1, 1, BN, D)
#     '''
#     BLOCK_SIZE_M = Symbol("BLOCK_SIZE_M", constexpr=True)
#     BLOCK_SIZE_N = Symbol("BLOCK_SIZE_N", constexpr=True)
#     BLOCK_SIZE_D = q.shape[-1]
    
#     def _arrange_qo(x):
#         # (B, N, SQ, D)->(B, N, SQ//BM, 1)x(1, 1, BM, D)
#         x_arranged = x.tile((1, 1, BLOCK_SIZE_M, BLOCK_SIZE_D))
#         # ->(B, N, 1, 1)x(1, 1, SQ//BM, 1)x(1, 1, BM, D)
#         x_arranged = x_arranged.tile((1, 1, -1, -1))
#         # ->(B, N, 1, 1)x(SQ//BM,)x(1, 1, BM, D)
#         x_arranged.dtype = x_arranged.dtype.squeeze((0, 1, 3))
#         # ->(B, N, 1, 1)x(SQ//BM,)x(BM, D)
#         x_arranged.dtype.dtype = x_arranged.dtype.dtype.squeeze((0, 1))

#         return x_arranged

#     def _arrange_kv(x):
#         # (B, N, SK, D)->(B, N, SK//BN, 1)x(1, 1, BN, D)
#         x_arranged = x.tile((1, 1, BLOCK_SIZE_N, BLOCK_SIZE_D))
#         # ->(B, N, 1, 1)x(1, 1, SK//BN, 1)x(1, 1, BN, D)
#         x_arranged = x_arranged.tile((1, 1, -1, -1))
#         # ->(B, N, 1, 1)x(SK//BN,)x(1, 1, BN, D)
#         x_arranged.dtype = x_arranged.dtype.squeeze((0, 1, 3))
#         # ->(B, N, 1, 1)x(SK//BN,)x(BN, D)
#         x_arranged.dtype.dtype = x_arranged.dtype.dtype.squeeze((0, 1))

#         return x_arranged

#     q_arranged = _arrange_qo(q)
#     o_arranged = _arrange_qo(o)
#     k_arranged = _arrange_kv(k)
#     v_arranged = _arrange_kv(v)

#     # subs = {
#     #     q: Tensor(shape=(2, 10, 1024, 128)),
#     #     k: Tensor(shape=(2, 10, 1024, 128)),
#     #     BLOCK_SIZE_M: 128,
#     #     BLOCK_SIZE_N: 32,
#     # }

#     # print(q_arranged.eval(subs).shape)
#     # print(k_arranged.eval(subs).shape)

#     # # exit(0)

#     return q_arranged, k_arranged, v_arranged, o_arranged, sm_scale


# def application(q, k, v, o, sm_scale):
#     # 防止 doc string 问题所以 application 中用 ‘#’ 注释
#     # qo: (SQ//BM,)x(BM, D)
#     # kv: (SK//BN,)x(BN, D)

#     # 九齿不支持直接声明嵌套维度的数组(NK,)x(BM, D)所以需要用一个循环
#     for i in range(q.shape[0]):
#         q_i = ntl.cast(q[i], ntl.float32) * sm_scale
#         m_i = ntl.full((q_i.shape[0], ), float("-inf"), dtype=ntl.float32)
#         l_i = ntl.zeros((q_i.shape[0], ), dtype=ntl.float32)
#         o_i = ntl.zeros((q_i.shape[0], q_i.shape[1]), dtype=ntl.float32)
#         for j in range(k.shape[0]):
#             k_j = ntl.cast(k[j], ntl.float32)
#             v_j = ntl.cast(v[j], ntl.float32)

#             s_ij = ntl.dot(q_i, ntl.trans(k_j))
#             m_ij = ntl.max(s_ij, axis=1)
#             m_i_new = ntl.maximum(m_ij, m_i)
#             p_ij = ntl.exp(s_ij - m_i_new[:, None])

#             diff_exp = ntl.exp(m_i - m_i_new)
#             l_ij = ntl.sum(p_ij, axis=1)
#             l_i_new = l_i * diff_exp + l_ij

#             o_i = o_i * (l_i / l_i_new * diff_exp)[:, None] + ntl.dot(p_ij, v_j) / l_i_new[:, None]
#             m_i = m_i_new
#             l_i = l_i_new
#         o[i] = ntl.cast(o_i, o[i].dtype)


# @lru_cache(1)
# def premake():
#     kernel = make(arrangement, application, (
#         Tensor(4, shape_options=(None, None, None, {"constexpr": True})),
#         Tensor(4, shape_options=(None, None, None, {"constexpr": True})),
#         Tensor(4, shape_options=(None, None, None, {"constexpr": True})),
#         Tensor(4, shape_options=(None, None, None, {"constexpr": True})),
#         Tensor(0),
#         ))
#     return kernel

# def ntl_flash(q, k, v):
#     o = torch.zeros_like(q)

#     # chunk
#     BLOCK_SIZE_M = 128
#     BLOCK_SIZE_N = 32


#     assert q.shape[-2] % BLOCK_SIZE_M == 0 and k.shape[-2] % BLOCK_SIZE_N == 0

#     premake()(q, k, v, o, 1/math.sqrt(q.shape[-1]), BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N)
    
#     return o





