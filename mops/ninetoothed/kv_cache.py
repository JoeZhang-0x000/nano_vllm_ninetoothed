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
    key/value: (M,)x(H,)
    k_cache/v_cache: (M,)x(E, H)
    slot_mapping: (M,)x(1,)
    '''
    M = key.shape[0]
    H = key.shape[1]
    E = k_cache.shape[0]

    def _kv_aranged(x):
        # (M, H)->(M, 1)x(1, H)->(M, )x(H, )
        x_arranged = x.tile((1, H)).squeeze((1, ))
        x_arranged.dtype = x_arranged.dtype.squeeze((0, ))
        return x_arranged

    # (M,)(E,)(H,)
    def _cache_arranged(x):
        x_arranged = (
            x # (E, H)
            .tile((1, -1)) # (E, 1)x(1, H)
            .squeeze((1,)) # (E, )x(1, H)
            .tile((-1,)) # (1,)x(E,)x(1, H)
            .expand((M,)) # (M,)x(E,)x(1,H)
        )
        x_arranged.dtype.dtype = x_arranged.dtype.dtype.squeeze((0,)) # (M,)x(E,)x(H,)
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
    key/vale: (H,)
    cache: (E,)x(H,)
    slot: (1,)
    '''
    slot = slot_mapping
    k = key
    v = value

    k_cache[slot] = k[None, :]
    v_cache[slot] = v[None, :]



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
    # print(key.shape, k_cache.shape)
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    premake()(key.view(-1, D), value.view(-1, D), k_cache.view(-1, D), v_cache.view(-1, D), slot_mapping)


@register_ninetoothed_op
def store_kvcache(*args):
    appaly(*args)