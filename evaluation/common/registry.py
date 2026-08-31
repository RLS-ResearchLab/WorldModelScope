"""Map a model name + checkpoint -> a constructed adapter.

The runner never imports an adapter module directly; it asks here. Keeps the
metric/runner code free of any model dependency.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

_BUILDERS: dict[str, Callable] = {}


def resolved_name(model: str, ckpt: str) -> str:
    """The results-folder / W&B-run stem for a (model, checkpoint), known without
    building the adapter -- so a run can be named at init. DINO-WM's encoder type
    lives in the checkpoint's embedded config; the others map 1:1."""
    if model != "dino_wm":
        return {"lewm": "lewm_bridge"}.get(model, model)
    try:
        import torch

        cfg = torch.load(ckpt, map_location="cpu", weights_only=False)["config"]
        return f"dino_wm_{cfg['model']['encoder']['type']}"
    except Exception:
        for tag in ("dinov2", "eupe"):
            if tag in str(ckpt):
                return f"dino_wm_{tag}"
        return model


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
