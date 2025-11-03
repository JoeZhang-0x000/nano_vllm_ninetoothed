import torch
from .registry import register_torch_op


@register_torch_op
def linear(x, weight, bias=None):
        return torch.nn.functional.linear(x, weight, bias)

@register_torch_op
def rms_forward(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + eps))
        x = x.to(orig_dtype).mul_(weight)
        return x

@register_torch_op
def add_rms_forward(x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + eps))
        x = x.to(orig_dtype).mul_(weight)
        return x, residual

@register_torch_op
def siluAndMul(x: torch.Tensor, y: torch.Tensor):
        return torch.nn.functional.silu(x) * y


@register_torch_op
def sampler_forward(logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        # sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        sample_tokens = torch.argmax(probs, dim=-1)
        return sample_tokens