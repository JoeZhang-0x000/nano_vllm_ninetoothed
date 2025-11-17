from ninetoothed import Tensor, make, block_size, Symbol
import ninetoothed.language as ntl
import torch
from mops.ninetoothed.config import *
from functools import lru_cache
from mops.ninetoothed.registry import register_ninetoothed_op


def arrangement(logits, temperatures, softmax_output, output):
    '''
    logits: (B, N)
    temperatures: (B, )
    softmax: (B, N)
    output: (B,)
    ->
    logits: (B,)x(N//BN,)x(BN,)
    temperatures: (B,)x(1,)
    softmax: (B,)x(N//BN,)x(BN,)
    output: (B,)x(1, )
    '''
    BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)

    # (B, N)->(B, N/BN)x(1, BN)->(B, 1)x(1, N/BN)x(1, BN)->(B, )x(1, N/BN)x(1, BN)
    logits_arranged = logits.tile((1, BLOCK_SIZE)).tile((1, -1)).squeeze(1)
    # ->(B, )x(N/BN, )x(1, BN)
    logits_arranged.dtype = logits_arranged.dtype.squeeze(0)
    # ->(B, )x(N/BN, )x(BN, )
    logits_arranged.dtype.dtype = logits_arranged.dtype.dtype.squeeze(0)

    # (B, N)->(B, N/BN)x(1, BN)->(B, 1)x(1, N/BN)x(1, BN)->(B, )x(1, N/BN)x(1, BN)
    softmax_arranged = softmax_output.tile((1, BLOCK_SIZE)).tile((1, -1)).squeeze(1)
    # ->(B, )x(N/BN, )x(1, BN)
    softmax_arranged.dtype = softmax_arranged.dtype.squeeze(0)
    # ->(B, )x(N/BN, )x(BN, )
    softmax_arranged.dtype.dtype = softmax_arranged.dtype.dtype.squeeze(0)

    # (B,) ->  (B,)x(1,)
    temperatures_arranged = temperatures.tile((1,))

    # (B,) -> (B, )x(1, )
    output_arranged = output.tile((1,))

    return logits_arranged, temperatures_arranged, softmax_arranged, output_arranged


# TODO: gumbel max
def application(logits, temperatures, softmax_out, output):
    prev_max = ntl.cast(float("-inf"), ntl.float32)
    denominator = ntl.cast(0, ntl.float32)

    # calculate probs
    for k in range(logits.shape[0]):
        # logits / temperatures
        logits_fp32 = ntl.cast(logits[k], ntl.float32)
        temperatures_fp32 = ntl.cast(temperatures, ntl.float32)
        logits[k] = logits_fp32 / temperatures_fp32

    # online softmax
    for k in range(logits.shape[0]):
        local_max = ntl.maximum(prev_max, ntl.max(logits[k]))
        probs_max_diff_exp = ntl.exp(logits[k] - local_max)
        prev_local_max_diff_exp = ntl.exp(prev_max - local_max)
        denominator = denominator * prev_local_max_diff_exp + ntl.sum(probs_max_diff_exp)
        prev_max = local_max
        
    for k in range(logits.shape[0]):
        numerator = ntl.exp(logits[k] - prev_max)
        softmax_out[k] = numerator / denominator

    # argmax
    global_max_val = float("-inf")
    global_max_idx = -1
    for k in range(softmax_out.shape[0]):
        local_max_val = ntl.max(softmax_out[k])
        take_local = local_max_val > global_max_val
        global_max_val = ntl.where(take_local, local_max_val, global_max_val)
        is_max_in_block = softmax_out[k] == local_max_val
        indices_in_block = ntl.where(is_max_in_block, softmax_out[k].offsets(-1), -1)
        local_max_idx = ntl.max(indices_in_block)
        global_max_idx = ntl.where(take_local, local_max_idx, global_max_idx)
    output = global_max_idx
    

       
kernel = make(arrangement, application, (Tensor(2), Tensor(1), Tensor(2), Tensor(1)))

@register_ninetoothed_op
def sampler_forward(logits: torch.Tensor, temperatures: torch.Tensor):
    assert logits.dim() == 2, "dimension should be 2!"
    B, H = logits.shape
    softmax_output = torch.empty_like(logits)
    output = torch.empty((B, ), dtype=torch.int64, device=logits.device)
    kernel(logits, temperatures, softmax_output, output, BLOCK_SIZE=1024)
    return output