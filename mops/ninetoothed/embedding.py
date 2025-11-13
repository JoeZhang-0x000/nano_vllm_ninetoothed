from ninetoothed import Tensor, make, block_size, Symbol
import ninetoothed.language as ntl
import torch
from mops.ninetoothed.config import *
from functools import lru_cache
from mops.ninetoothed.registry import register_ninetoothed_op



def arrangement(x, weight, out):
    '''
    x: (M, )
    w: (E, H)
    o: (M, H)
    1. Only tile M with BM
    2. 
    ->
    x: (M_BM, )x(BM, )
    w: (M_BM, )x(E, H)
    o: (M_BM, )x(H_BH, )x(BM, BH)
    '''
    BLOCK_SIZE_M = Symbol("BLOCK_SIZE_M", constexpr=True)
    BLOCK_SIZE_H = Symbol("BLOCK_SIZE_H", constexpr=True)

    # (M, ) -> (M_BM, )x(BM, )
    x_arranged = x.tile((BLOCK_SIZE_M, ))

    # (E, H) -> (1, 1)x(E, H) -> (M_BM, 1)x(E, H) -> (M_BM, )x(E, H)
    w_arranged = weight.tile((-1, -1)).squeeze((0,)).expand((x_arranged.shape[0]))

    # (M, H)->(M_BM, H_BH)x(BM, BH)->(M_BM, 1)x(1, H_BH)x(BM, BH)
    o_arranged = out.tile((BLOCK_SIZE_M, BLOCK_SIZE_H)).tile((1, -1)).squeeze((1,))
    o_arranged.dtype = o_arranged.dtype.squeeze((0,))

    return x_arranged, w_arranged, o_arranged


def application(x, w, o):
    # o[i, j] = w[x[i], j]
    M = x.source.shape[0]
    E = w.source.shape[0]
    H = o.source.shape[1]
    x_load = x
    for i in range(o.shape[0]):
        _offs_m = o[i].offsets(0)
        _offs_n = o[i].offsets(1)

        o[i] = ntl.load(
            w.data_ptr() + x_load[:, None] * w.shape[1] + _offs_n[None, :],
            mask=(_offs_m[:, None] < M) & (_offs_n[None, :] < H),
        )

@lru_cache(1)
def premake():
    kernel = make(arrangement, application, (
        Tensor(1, shape_options=({"constexpr": True})),
        Tensor(2, shape_options=({"constexpr": True}, {"constexpr": True})), 
        Tensor(2, shape_options=({"constexpr": True}, {"constexpr": True}))
        ))
    return kernel
    
@register_ninetoothed_op
def embedding(x: torch.Tensor, weight: torch.Tensor):
    '''
    x: (B, S)
    weight: (E, H)
    output: (B, S, H)
    '''
    assert x.is_cuda
    assert x.ndim == 2 or x.ndim == 1
    if x.ndim == 2:
        B, S = x.shape
        E, H = weight.shape
        output = torch.empty(B, S, H, device=x.device)

        _x = x.view(-1) # (BxS)
        _w = weight #(E, H)
        _o = output.view(-1, H) #(BxS, H)
        M = B*S
    if x.ndim == 1:
        M = x.shape[0]
        M = x.shape[0]
        E, H = weight.shape
        output = torch.empty(M, H, device=x.device, dtype=weight.dtype)
        _x = x
        _w = weight #(E, H)
        _o = output #(BxS, H)
    BM = 32
    BH = H
    premake()(_x, _w, _o, BLOCK_SIZE_M=BM, BLOCK_SIZE_H=BH)
    return output