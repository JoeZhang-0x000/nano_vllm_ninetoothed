import os
from .registry import set_current_backend, get_op


_BACKEND = os.environ.get("BACKEND", "ninetoothed").lower()
print(f"Current Backend: {_BACKEND}")
set_current_backend(_BACKEND)

from . import triton 
from . import ninetoothed
from . import torch

# set_current_backend("torch")

linear = get_op("linear")
rms_forward = get_op("rms_forward")
add_rms_forward = get_op("add_rms_forward")
softmax = get_op("softmax")
siluAndMul = get_op("siluAndMul")
sampler_forward = get_op("sampler_forward")
apply_rotary_emb = get_op("apply_rotary_emb")
embedding = get_op("embedding")
store_kvcache = get_op("store_kvcache")
flash_attn_varlen_func = get_op("flash_attn_varlen_func")
flash_attn_with_kvcache = get_op("flash_attn_with_kvcache")
