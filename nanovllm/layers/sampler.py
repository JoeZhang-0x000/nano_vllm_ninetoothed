import torch
from torch import nn
from mops import sampler_forward

class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        return sampler_forward(logits, temperatures)
