from ninetoothed import Tensor, make, Symbol
import ninetoothed.language as ntl
import torch
from mops.ninetoothed.registry import register_ninetoothed_op


def arrangement(x, out):
    '''
    x: (M, N)
    out: (M, N)
    ->
    x: (M, 1)x(1, N/BN)x(1, BN)
    out: (M, 1)x(1, N/BN)x(1, BN)
    '''
    BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)
    # -> (M, N/BN)x(1, BN) -> (M, 1)x(1, N/BN)x(1, BN)
    x_arranged = x.tile((1, BLOCK_SIZE)).tile((1, -1))
    x_arranged = x_arranged.squeeze(1)
    x_arranged.dtype = x_arranged.dtype.squeeze(0)
    x_arranged.dtype.dtype = x_arranged.dtype.dtype.squeeze(0)

    out_arranged = out.tile((1, BLOCK_SIZE)).tile((1, -1))
    out_arranged = out_arranged.squeeze(1)
    out_arranged.dtype = out_arranged.dtype.squeeze(0)
    out_arranged.dtype.dtype = out_arranged.dtype.dtype.squeeze(0)
    return x_arranged, out_arranged


def application(x, out):
    running_max = float("-inf")
    running_denominator = float(0)
    for i in range(x.shape[0]):
        x_i = ntl.cast(x[i], ntl.float32)
        local_max = ntl.max(x_i)
        global_max = ntl.maximum(local_max, running_max)
        l_correct = ntl.exp(-global_max + running_max)
        numerator = ntl.exp(x_i - global_max)
        running_denominator = running_denominator * l_correct + ntl.sum(numerator)
        running_max = global_max

    for i in range(x.shape[0]):
        x_i = ntl.cast(x[i], ntl.float32)
        numerator  = ntl.exp(x_i - running_max)
        out[i] = numerator / running_denominator 


kernel = make(arrangement, application, (Tensor(2), Tensor(2)))

@register_ninetoothed_op
def softmax(x):
    out = torch.empty_like(x)
    kernel(x, out, BLOCK_SIZE=128)
    return out