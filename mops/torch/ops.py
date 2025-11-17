import torch
from mops.torch.registry import register_torch_op


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

@register_torch_op
def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
        x1, x2 = torch.chunk(x.float(), 2, dim=-1)
        y1 = x1 * cos - x2 * sin
        y2 = x2 * cos + x1 * sin
        return torch.cat((y1, y2), dim=-1).to(x.dtype)


@register_torch_op
def embedding(x: torch.Tensor, weight: torch.Tensor):
        return torch.nn.functional.embedding(x, weight)


@register_torch_op
def store_kvcache(*args, **kwargs):
        from mops.triton.attention import store_kvcache
        return store_kvcache(*args, **kwargs) 


import flash_attn

@register_torch_op
def flash_attn_varlen_func(*args, **kwargs):
        return flash_attn.flash_attn_varlen_func(*args, **kwargs)

@register_torch_op
def flash_attn_with_kvcache(*args, **kwargs):
        return flash_attn.flash_attn_with_kvcache(*args, **kwargs)

# @register_torch_op
# def flash_attn_with_kvcache(q, k_cache, v_cache, cache_seqlens, block_table, softmax_scale, causal):
#         '''
#         q: b 1 hq d
#         '''
#         o = torch.empty_like(q)
#         for b in range(q.shape[0]):
#                 k = torch.empty((1,
#                         cache_seqlens[b],
#                         k_cache.shape[2],
#                         q.shape[3],),
#                         device=q.device, dtype=q.dtype)
#                 v = torch.empty((
#                         1,
#                         cache_seqlens[b],
#                         v_cache.shape[2],
#                         q.shape[3],),
#                         device=q.device, dtype=q.dtype)
#                 BLOCK_SIZE = k_cache.shape[1]
#                 for i in range(cache_seqlens[b]):
#                         k[0, i, ...] = k_cache[block_table[b, i // BLOCK_SIZE], i % BLOCK_SIZE, ...]
#                         v[0, i, ...] = v_cache[block_table[b, i // BLOCK_SIZE], i % BLOCK_SIZE, ...]
#                 # b s h d
#                 q_b = q[[b]]
                
#                 # gqa
#                 hq = q_b.shape[2]
#                 hk = k.shape[2]
#                 rep = hq // hk
#                 k = (k # b s h d
#                 .unsqueeze(-2) # b s h 1 d
#                 .expand((-1, -1, -1, rep, -1)) # b s h rep d
#                 .reshape(k.shape[0], k.shape[1], hq, k.shape[-1]) # b s hq d
#                 )
#                 v = (v # b s h d
#                 .unsqueeze(-2) # b s h 1 d
#                 .expand((-1, -1, -1, rep, -1)) # b s h rep d
#                 .reshape(k.shape[0], k.shape[1], hq, k.shape[-1]) # b s hq d
#                 )

#                 # b h s d
#                 q_b = q_b.permute(0, 2, 1, 3)
#                 k_b = k.permute(0, 2, 1, 3)
#                 v_b = v.permute(0, 2, 1, 3)
#                 # b s h d
#                 o_b = torch.nn.functional.scaled_dot_product_attention(q_b, k_b, v_b, scale=softmax_scale, is_causal=False).permute(0, 2, 1, 3)
#                 o[[b]] = o_b
#         return o