from ninetoothed import Tensor, make, block_size, Symbol
import ninetoothed.language as ntl
import torch
from mops.ninetoothed.config import *
from functools import lru_cache
from .registry import register_ninetoothed_op

BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)

def arrangement(logits, temperatures, output, H=BLOCK_SIZE):
    '''
    logits: (B, H)
    temperatures: (B, 1)
    output: (B, 1)
    ->
    logits: (B, 1) x (1, H)
    temperatures: (B, 1) x (1, H)
    output: (B, 1) x (1, 1)
    '''
    logits_arranged = logits.tile((1, H))
    temperatures_arranged = temperatures.expand((-1, H)).tile((1, H))

    # temperatures_arranged = temperatures.tile((1, H))
    # temperatures_arranged.dtype = temperatures_arranged.dtype.expand((-1, H))
    output_arranged = output.tile((1, H))

    subs = {
        logits: Tensor(shape=(2, 3)),
        temperatures: Tensor(shape=(2, 1)),
        output: Tensor(shape=(2, 1)),
        H: 3
    }

    print('logits \n', logits_arranged.eval(subs))
    print('temperatures \n', temperatures_arranged.eval(subs))
    print('outputk \n', output_arranged.eval(subs))

    exit(0)
    return logits_arranged, temperatures_arranged, output_arranged


# TODO: gumbel max
def application(logits, temperatures, output):
    logits_fp32 = ntl.cast(logits, ntl.float32)
    logits_fp32 = logits_fp32 / temperatures
    var = logits_fp32 - ntl.max(logits_fp32)
    probs = ntl.exp(var) / (ntl.sum(ntl.exp(logits_fp32)))

    max_val = ntl.max(probs, axis=-1)
    is_max = max_val == probs
    cur_idx = logits.offsets(-1)
    cur_idx = ntl.where(is_max, cur_idx, -1)
    max_idx = ntl.max(cur_idx)
    output = max_idx


kernel = make(arrangement, application, (Tensor(2), Tensor(2), Tensor(2)))

@register_ninetoothed_op
def sampler_forward(logits: torch.Tensor, temperatures: torch.Tensor):
    assert logits.dim() == 2, "dimension should be 2!"
    B, H = logits.shape
    output = torch.empty((B, 1), dtype=torch.int32, device=logits.device)
    kernel(logits, temperatures.view(-1, 1), output, BLOCK_SIZE=H)
    return output.ravel()

    