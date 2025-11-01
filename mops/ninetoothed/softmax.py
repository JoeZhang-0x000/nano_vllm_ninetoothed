from ninetoothed import Tensor, make, Symbol
import ninetoothed.language as ntl
import torch
from .registry import register_ninetoothed_op

BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)

def arrangement(x, out, BLOCK_SIZE=BLOCK_SIZE):
    '''
    x: (M, N)
    out: (M, N)
    ->
    x_arranged: (M, 1)x(1, N)
    out_arranged: (M, 1)x(1, N)
    '''
    x_arranged = x.tile((1, BLOCK_SIZE))
    out_arranged = out.tile((1, BLOCK_SIZE))
    return x_arranged, out_arranged


def application(x, out):
    '''
    x: (1, N)
    out: (1, N)
    '''
    x_max = ntl.max(x)
    x_sub_max = x - x_max
    numerator = ntl.exp(x_sub_max)
    denominator = ntl.sum(numerator, axis=-1)
    out = numerator / denominator

kernel = make(arrangement, application, (Tensor(2), Tensor(2)))

@register_ninetoothed_op
def softmax(x):
    out = torch.empty_like(x)
    kernel(x, out, BLOCK_SIZE=out.shape[1])
    return out

    