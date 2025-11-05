from ninetoothed import Tensor, make, block_size, Symbol
import ninetoothed.language as ntl
import torch
from mops.ninetoothed.config import *
from functools import lru_cache
from mops.ninetoothed.registry import register_ninetoothed_op

BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)

def arrangement(x, cos, sin, BLOCK_SIZE=BLOCK_SIZE):
    '''
    BLOCK_SZZIE = D
    x: B, H, 2*D
    cos/sin: B, 1, D
    ->
    x: (B, H, 1) x (2,) x (D,)
    cos/sin: (B, H, 1) x (D,)
    '''
    B, H, _D = x.shape
    D = BLOCK_SIZE
    # ... ->(B, H, 2)x(1, 1, D)->(B,H,1)x(1,1,2)x(1,1,D)
    x_arranged = x.tile((1, 1, D)).tile((1, 1, 2))

    # ... ->(B, H, D)->(B, H, 1)x(1, 1, D)
    cos_arranged = cos.expand((-1, H, -1)).tile((1, 1, D))
    sin_arranged = sin.expand((-1, H, -1)).tile((1, 1, D))

    def _squeeze(x_arranged):
        for _ in range(2):
            x_arranged.dtype = x_arranged.dtype.squeeze(0)
        return x_arranged

    x_arranged = _squeeze(x_arranged)
    x_arranged.dtype.dtype = x_arranged.dtype.dtype.squeeze(0)
    x_arranged.dtype.dtype = x_arranged.dtype.dtype.squeeze(0)
    cos_arranged = _squeeze(cos_arranged)
    sin_arranged = _squeeze(sin_arranged)

    return x_arranged, cos_arranged, sin_arranged

def application(x, cos, sin):
    x1 = x[0]
    x2 = x[1]
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    x[0] = y1
    x[1] = y2

@lru_cache(1)
def permake():
    kernel = make(arrangement, application, (Tensor(3), Tensor(3), Tensor(3)))
    return kernel

@register_ninetoothed_op
def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    '''
    x: B, H, 2D
    sin/cos: B, 1, D
    '''
    permake()(x, cos, sin, BLOCK_SIZE=cos.shape[-1])
    return x