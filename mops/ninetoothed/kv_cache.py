from ninetoothed import Tensor, make, block_size, Symbol
import ninetoothed.language as ntl
import torch
from mops.ninetoothed.config import *
from functools import lru_cache
from mops.ninetoothed.registry import register_ninetoothed_op



def arrangement(key, value, k_cache, v_cache, slot_mapping):
    '''
    key/value: (M, H)
    k_cache/v_cache: (E, H)
    slot_mapping: (M,)
    ->
    key/value: (M, 1)x(H,)
    k_cache/v_cache: (M, 1)x(E, H)
    slot_mapping: (M,)x(1,)
    '''
    M = key.shape[0]
    H = key.shape[1]
    E = k_cache.shape[0]

    def _kv_aranged(x):
        # (M, H)->(M, 1)x(1, H)
        x_arranged = x.tile((1, H))
        return x_arranged


    def _cache_arranged(x):
        # (E, H)->(1, 1)x(E, H)->(M, 1)x(E, H)
        x_arranged = x.tile((-1, -1)).expand((key_arranged.shape[0], -1))
        return x_arranged

    key_arranged = _kv_aranged(key)
    value_arranged = _kv_aranged(value)
    k_cache_arranged = _cache_arranged(k_cache)
    v_cache_arranged = _cache_arranged(v_cache)

    # (M,)->(M,)x(1,)
    slot_mapping_arranged = slot_mapping.tile((1,))

    return key_arranged, value_arranged, k_cache_arranged, v_cache_arranged, slot_mapping_arranged

# 没有stride导致的？？？ 乱码
def application(key, value, k_cache, v_cache, slot_mapping):
    '''
    key/vale: (1, H)
    cache: (E, H)
    slot: (1,)
    '''
    slot = slot_mapping
    k = key
    v = value
    offs_h = key.offsets(-1)
    cache_offsets = slot[:, None] * k_cache.shape[1] + offs_h[None, :]
    ntl.store(
        k_cache.data_ptr() + cache_offsets, k,
        mask=offs_h[None, :] < key.shape[1]
    )

    ntl.store(
        v_cache.data_ptr() + cache_offsets, v,
        mask=offs_h[None, :] < value.shape[1]
    )



@lru_cache(1)
def premake():
    tensors = (
        Tensor(2, shape_options=(
            {"constexpr": True},
            {"constexpr": True},
        )),
        Tensor(2, shape_options=(
            {"constexpr": True},
            {"constexpr": True},
        )),
        Tensor(2, shape_options=(
            {"constexpr": True},
            {"constexpr": True},
        )),
        Tensor(2, shape_options=(
            {"constexpr": True},
            {"constexpr": True},
        )),
        Tensor(1,shape_options=(
            {"constexpr": True},
        ))
    )
    kernel = make(arrangement, application, tensors)
    return kernel

def appaly(key, value, k_cache, v_cache, slot_mapping):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    premake()(key, value, k_cache, v_cache, slot_mapping)


@register_ninetoothed_op
def store_kvcache(*args):
    appaly(*args)