"""Output metrics for causal interpretability experiments."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def logit_margin(logits: jax.Array, target_id: int, foil_id: int) -> jax.Array:
    """Target-minus-foil logit margin at the final prompt position.

    Args:
        logits: Unbatched model output with shape (sequence, vocabulary).
        target_id: Token ID of the desired answer (one token).
        foil_id: Token ID of the alternative answer (one token).
    """
    return logits[-1, target_id] - logits[-1, foil_id]


def top_positive_neurons(scores: jax.Array, k: int) -> list[tuple[int, int, float]]:
    """Return up to k layer/neuron pairs with the largest positive scores.

    The score array has shape (layers, mlp_hidden_dim). The results are ordinary
    Python values intended for reporting, not for JIT compilation.
    """
    if scores.ndim != 2 or k < 1:
        raise ValueError("scores must have shape (layers, neurons), and k >= 1")

    n_neurons = scores.shape[1]
    flattened = jnp.asarray(scores).reshape(-1)
    descending = jnp.argsort(flattened)[::-1]
    result = []
    for flat_index in descending[:k].tolist():
        score = float(flattened[flat_index])
        if score <= 0:
            break
        result.append((flat_index // n_neurons, flat_index % n_neurons, score))
    return result
