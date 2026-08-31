"""Model-agnostic metric functions.

Every function takes plain tensors in the canonical ``(B, K, P, D)`` shape (or a
shape that flattens to ``(N, D)``) and returns Python floats / plain dicts. No
function here imports a model or an adapter.

Comparability tags (used by ``report.py`` to decide what may be ranked):
    ratio   -- dimensionless; cancels the latent's units; cross-model comparable
    shared  -- computed in a shared space (pixels / frozen judge / task success)
    raw     -- in the model's own latent units; logged for debugging, never ranked
"""
