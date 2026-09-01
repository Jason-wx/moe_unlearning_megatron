"""Locally maintained, pure-text XTuner compatibility helpers."""

from .pretrain import prepare_pretrain_text, proxy_attention_flops
from .soft_pack import build_expand_soft_pack_infos

__all__ = [
    "build_expand_soft_pack_infos",
    "prepare_pretrain_text",
    "proxy_attention_flops",
]
