# Stand in for flash_attn.modules.mha.MHA, for the one way UniRig uses it:
#     MHA(embed_dim=feat_dim, num_heads=num_heads, cross_attn=True)(q, x_kv=kv)
# UniRig leaves use_flash_attn at its default (False), so the real class already runs plain PyTorch
# attention (its CrossAttention). This repeats that arithmetic with the same parameter names (Wq, Wkv,
# out_proj), so UniRig's checkpoint loads into it unchanged. Anything else raises.
import math

import torch
from torch import nn


class MHA(nn.Module):
    def __init__(self, embed_dim, num_heads, num_heads_kv=None, cross_attn=False, qkv_proj_bias=True,
                 out_proj_bias=True, dropout=0.0, softmax_scale=None, causal=False, use_flash_attn=False,
                 device=None, dtype=None, **unsupported):
        super().__init__()
        if not cross_attn or causal or dropout or use_flash_attn or (num_heads_kv not in (None, num_heads)):
            raise NotImplementedError("the stand in covers only UniRig's use: non causal cross attention")
        if any(unsupported.values()):
            raise NotImplementedError(f"the stand in doesn't support {sorted(k for k, v in unsupported.items() if v)}")
        kwargs = {"device": device, "dtype": dtype}
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.softmax_scale = softmax_scale
        self.Wq = nn.Linear(embed_dim, embed_dim, bias=qkv_proj_bias, **kwargs)
        self.Wkv = nn.Linear(embed_dim, 2 * embed_dim, bias=qkv_proj_bias, **kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=out_proj_bias, **kwargs)

    def forward(self, x, x_kv=None, **kwargs):
        if kwargs.get("key_padding_mask") is not None or kwargs.get("inference_params") is not None:
            raise NotImplementedError("the stand in doesn't support masks or a key value cache")
        q = self.Wq(x)
        kv = self.Wkv(x_kv if x_kv is not None else x)
        q = q.unflatten(-1, (self.num_heads, self.head_dim))  # ... (h d) -> ... h d
        kv = kv.unflatten(-1, (2, self.num_heads, self.head_dim))  # ... (two h d) -> ... two h d
        k, v = kv.unbind(dim=-3)
        # flash-attn's CrossAttention, step for step
        scale = self.softmax_scale or 1.0 / math.sqrt(q.shape[-1])
        scores = torch.einsum("bthd,bshd->bhts", q, k * scale)
        attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
        context = torch.einsum("bhts,bshd->bthd", attention, v)
        return self.out_proj(context.flatten(-2))
