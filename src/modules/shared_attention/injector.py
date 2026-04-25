from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .factory import build_attention


@dataclass(frozen=True)
class AttentionHookSpec:
    module_name: str
    channels: int
    output_index: int | None = None


def _resolve_module(model: nn.Module, module_name: str) -> nn.Module:
    module = model
    for part in module_name.split("."):
        module = getattr(module, part)
    return module


def attach_attention_hooks(model: nn.Module, attention_type: str | None, specs: list[AttentionHookSpec]) -> list:
    handles = []
    if not attention_type:
        return handles

    for index, spec in enumerate(specs):
        attention = build_attention(attention_type, spec.channels)
        model.add_module(f"_shared_attention_hook_{index}", attention)
        target = _resolve_module(model, spec.module_name)

        def _hook(_module: nn.Module, _inputs, output, _attention=attention, _spec=spec):
            if _spec.output_index is None:
                if isinstance(output, torch.Tensor):
                    return _attention(output)
                return output
            if not isinstance(output, (list, tuple)):
                return output
            items = list(output)
            items[_spec.output_index] = _attention(items[_spec.output_index])
            return type(output)(items) if isinstance(output, tuple) else items

        handles.append(target.register_forward_hook(_hook))
    return handles
