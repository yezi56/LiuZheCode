"""Shared pluggable attention modules for all models."""

from .factory import AVAILABLE_ATTENTIONS, build_attention
from .injector import AttentionHookSpec, attach_attention_hooks

__all__ = [
    "AVAILABLE_ATTENTIONS",
    "AttentionHookSpec",
    "attach_attention_hooks",
    "build_attention",
]
