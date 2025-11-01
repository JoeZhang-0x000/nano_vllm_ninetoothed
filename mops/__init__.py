import os
from .registry import set_current_backend, get_op


_BACKEND = os.environ.get("BACKEND", "ninetoothed").lower()
print(f"Current Backend: {_BACKEND}")
set_current_backend(_BACKEND)

from . import triton 
from . import ninetoothed
from . import torch


linear = get_op("linear")
rms_forward = get_op("rms_forward")
add_rms_forward = get_op("add_rms_forward")
softmax = get_op("softmax")

