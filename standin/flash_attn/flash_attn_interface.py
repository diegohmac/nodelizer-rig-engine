import torch.nn.functional as F


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, **_):
    """flash-attn's call, with PyTorch's attention. Tensors are (batch, length, heads, head size)."""
    if dropout_p:
        raise NotImplementedError("the stand in is for inference only")
    k = k.to(q.dtype)
    v = v.to(q.dtype)
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=causal, scale=softmax_scale
    )
    return out.transpose(1, 2)
