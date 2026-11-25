"""Logit margins, gradient attribution, and causal ablation with JAX."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from penzai import pz

from mechinterp.interventions import GatedModel, with_scales
from mechinterp.metrics import logit_margin


def margin_function(
    gated: GatedModel,
    token_ids: jax.Array,
    target_id: int,
    foil_id: int,
) -> Callable[[jax.Array], jax.Array]:
    """Build a differentiable margin function of the intervention scales.

    Since the gate multiplies h_i, dM/dscale_i at scale_i=1 equals
    h_i * dM/dh_i (gradient x activation), at that prompt position.
    """
    if token_ids.ndim != 1:
        raise ValueError("This experiment expects one unbatched token sequence")
    tokens = pz.nx.wrap(jnp.asarray(token_ids, dtype=jnp.int32), "seq")

    def margin(scales: jax.Array) -> jax.Array:
        modified_model = with_scales(gated, scales)
        logits = modified_model(tokens).unwrap("seq", "vocabulary")
        return logit_margin(logits, target_id, foil_id)

    return margin


def gradient_times_activation(
    margin_fn: Callable[[jax.Array], jax.Array],
    baseline_scales: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Baseline margin and gradient x activation for every gated neuron."""
    return jax.value_and_grad(margin_fn)(baseline_scales)


def ablated_scales(
    baseline_scales: jax.Array, neurons: list[tuple[int, int]]
) -> jax.Array:
    """Return a new scale array with the selected neurons set to zero."""
    if not neurons:
        return baseline_scales
    layer_ids, neuron_ids = zip(*neurons)
    return baseline_scales.at[
        jnp.asarray(layer_ids), jnp.asarray(neuron_ids)
    ].set(0.0)



def next_token_logits_function(
    gated: GatedModel,
    token_ids: jax.Array,
) -> Callable[[jax.Array], jax.Array]:
    tokens = pz.nx.wrap(
        jnp.asarray(token_ids, dtype=jnp.int32),
        "seq",
    )

    def logits(scales):
        model = with_scales(gated, scales)
        return model(tokens).unwrap("seq", "vocabulary")[-1]

    return logits



from mechinterp.activations import (
    capture_final_mlp_activations,
    capture_swiglu_activations,
)


def test_capture_final_mlp_activations():
    gated = instrument_mlp_neurons(
        tiny_llama_model(),
        expected_layers=2,
    )

    ids = jnp.array([1, 2, 3], dtype=jnp.int32)

    captured = capture_final_mlp_activations(
        gated.model,
        ids,
        (0, 1),
    )

    assert captured.shape == (
        2, gated.num_neurons
    )

    for layer in range(2):
        expected = capture_swiglu_activations(
            gated, ids, layer
        ).activation[-1]

        assert jnp.allclose(
            captured[layer],
            expected,
            atol=1e-5,
            rtol=1e-5,
        )
