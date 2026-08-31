"""Model-agnostic world-model evaluation harness.

Every metric is computed against a canonical latent tensor shape ``(B, T, P, D)``
so a number from one model's latent space is comparable to another's. Metric
code never imports a model; models are wrapped to the ``WorldModelAdapter``
contract in ``evaluation/adapters/``.
"""
