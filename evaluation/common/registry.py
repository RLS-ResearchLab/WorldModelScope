"""Map a model name + checkpoint -> a constructed adapter.

The runner never imports an adapter module directly; it asks here. Keeps the
metric/runner code free of any model dependency.
"""
from __future__ import annotations

from typing import Callable

_BUILDERS: dict[str, Callable] = {}


def register(name: str):
    def deco(fn: Callable) -> Callable:
        _BUILDERS[name] = fn
        return fn
    return deco


def build_adapter(name: str, ckpt: str, device: str = "cuda", **kw):
    if name not in _BUILDERS:
        raise KeyError(f"unknown model '{name}'. registered: {sorted(_BUILDERS)}")
    return _BUILDERS[name](ckpt=ckpt, device=device, **kw)


@register("vjepa2_ac")
def _build_vjepa2_ac(ckpt: str, device: str, **kw):
    from evaluation.adapters.vjepa2_ac import VJEPA2ACAdapter

    return VJEPA2ACAdapter(predictor_ckpt=ckpt, device=device, **kw)


@register("dino_wm")
def _build_dino_wm(ckpt: str, device: str, **kw):
    from evaluation.adapters.dino_wm import DINOWMAdapter

    return DINOWMAdapter(ckpt=ckpt, device=device, **kw)


@register("lewm")
def _build_lewm(ckpt: str, device: str, **kw):
    from evaluation.adapters.lewm import LeWMAdapter

    return LeWMAdapter(ckpt=ckpt, device=device, **kw)
