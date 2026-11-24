# A stand in for the flash-attn package, which has no official Windows build. It gives the two things
# the models use, done with PyTorch's own attention: flash_attn_func (SkinTokens) and
# flash_attn.modules.mha.MHA (UniRig's skin model). Both were checked against the real package in
# Docker on Linux.
from .flash_attn_interface import flash_attn_func

__version__ = "0.0.0+kit.torch"
