from __future__ import annotations

import torch.nn as nn

from .modules import (
    CAA,
    CBAMBlock,
    CPCA,
    EMCAM,
    EfficientChannelAttention,
    SEAttention,
    ShuffleAttention,
    TripletAttention,
)


AVAILABLE_ATTENTIONS = {
    "cbam": CBAMBlock,
    "se": SEAttention,
    "caa": CAA,
    "eca": EfficientChannelAttention,
    "cpca": CPCA,
    "ta": TripletAttention,
    "triplet": TripletAttention,
    "sa": ShuffleAttention,
    "shuffle": ShuffleAttention,
    "emcam": EMCAM,
}


def build_attention(attention_type: str | None, channels: int, **kwargs) -> nn.Module:
    if not attention_type:
        return nn.Identity()
    attention_type = attention_type.lower().strip()
    if attention_type in {"none", "identity"}:
        return nn.Identity()
    if attention_type not in AVAILABLE_ATTENTIONS:
        raise ValueError(
            f"Unsupported attention `{attention_type}`. Available: {', '.join(sorted(AVAILABLE_ATTENTIONS))}"
        )
    return AVAILABLE_ATTENTIONS[attention_type](channels, **kwargs)
