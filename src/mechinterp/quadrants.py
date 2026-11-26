import dataclasses

import jax
import jax.numpy as jnp

from penzai import pz
from penzai.models.transformer import model_parts

from mechinterp.interventions import (
    GatedModel,
    FinalTokenNeuronGate,
)


PP = 0
PN = 1
NP = 2
NN = 3

QUADRANTS = ("++", "+-", "-+", "--")

BASELINE = 0
DROP = 1
ONLY = 2
TOP_K = 3
RESTORE_QUADRANT = 4
RESTORE_RANDOM = 5
RESTORE_LARGEST = 6


def quadrant_ids(gate, value):
    return (
        2 * (gate < 0).astype(jnp.int32)
        + (value < 0).astype(jnp.int32)
    )


def ranked_mask(scores, count):
    seq_len, num_neurons = scores.shape

    count = jnp.broadcast_to(
        jnp.asarray(count), (seq_len,)
    )

    order = jnp.argsort(
        -scores,
        axis=-1,
        stable=True,
    )

    retained = (
        jnp.arange(num_neurons)[None, :]
        < count[:, None]
    )

    rows = jnp.arange(seq_len)[:, None]

    return jnp.zeros_like(
        scores, dtype=jnp.bool_
    ).at[rows, order].set(retained)


def quadrant_mask(
    gate,
    value,
    activation,
    mode,
    quadrant,
    keep,
    key,
):
    ids = quadrant_ids(gate, value)
    selected = ids == quadrant

    def full_model_mask():
        return jax.lax.switch(
            mode,
            (
                lambda: jnp.ones_like(
                    selected, dtype=jnp.bool_
                ),
                lambda: ~selected,
                lambda: selected,
            ),
        )

    def sparse_model_mask():
        magnitude = jnp.abs(activation)

        top = ranked_mask(magnitude, keep)

        discarded = ~top

        restore = discarded & selected

        count = restore.sum(axis=-1)

        def random_restore():
            scores = jax.random.uniform(
                key,
                activation.shape,
                dtype=jnp.float32,
            )

            scores = jnp.where(
                discarded, scores, -jnp.inf
            )

            return top | ranked_mask(scores, count)

        def largest_restore():
            scores = jnp.where(
                discarded, magnitude, -jnp.inf
            )

            return top | ranked_mask(scores, count)

        return jax.lax.switch(
            mode - TOP_K,
            (
                lambda: top,
                lambda: top | restore,
                random_restore,
                largest_restore,
            ),
        )

    return jax.lax.cond(
        mode < TOP_K,
        full_model_mask,
        sparse_model_mask,
    )


def quadrant_statistics(gate, value, activation, mask):
    ids = quadrant_ids(gate, value)

    membership = jax.nn.one_hot(
        ids, 4, dtype=jnp.float32
    )

    h = activation.astype(jnp.float32)

    def aggregate(weights):
        return jnp.sum(
            membership * weights[..., None],
            axis=(0, 1),
        )

    return jnp.stack([
        aggregate(jnp.ones_like(h)),
        aggregate(jnp.abs(h)),
        aggregate(jnp.square(h)),
        aggregate(mask.astype(jnp.float32)),
        aggregate(jnp.abs(h) * mask),
    ], axis=-1)


@pz.pytree_dataclass
class QuadrantSwiGLU(pz.nn.Layer):
    gate_linear: pz.nn.Layer
    gate_nonlinearity: pz.nn.Layer
    value_branch: pz.nn.Layer

    mode: jax.Array
    quadrant: jax.Array
    keep: jax.Array
    key: jax.Array

    saved: pz.StateVariable

    def __call__(self, x, **kwargs):
        gate = self.gate_linear(x, **kwargs)

        value = self.value_branch(x, **kwargs)

        activated_gate = self.gate_nonlinearity(
            gate, **kwargs
        )

        g = gate.unwrap("seq", "neurons")
        v = value.unwrap("seq", "neurons")

        h = (
            activated_gate.unwrap("seq", "neurons")
            * v
        )

        mask = quadrant_mask(
            g,
            v,
            h,
            self.mode,
            self.quadrant,
            self.keep,
            self.key,
        )

        self.saved.value = quadrant_statistics(
            g, v, h, mask
        )

        edited = jnp.where(mask, h, 0)

        return pz.nx.wrap(
            edited, "seq", "neurons"
        )


def with_quadrant_intervention(
    gated: GatedModel,
    mode,
    quadrant,
    keep,
    key,
):
    saved = {
        layer: pz.StateVariable(None)
        for layer in range(gated.num_layers)
    }

    instrumented_layers = set()

    def instrument(ff):
        if len(ff.sublayers) != 3:
            return ff

        glu, intervention, down = ff.sublayers

        if not isinstance(
            intervention, FinalTokenNeuronGate
        ):
            return ff

        if not isinstance(
            glu, pz.nn.BranchAndMultiplyTogether
        ):
            raise ValueError("Expected SwiGLU layer")

        layer = intervention.layer_index

        gate_branch, value_branch = glu.branches

        gate_linear, gate_nonlinearity = (
            gate_branch.sublayers
        )

        replacement = QuadrantSwiGLU(
            gate_linear=gate_linear,
            gate_nonlinearity=gate_nonlinearity,
            value_branch=value_branch,
            mode=mode,
            quadrant=quadrant,
            keep=keep,
            key=jax.random.fold_in(key, layer),
            saved=saved[layer],
        )

        instrumented_layers.add(layer)

        return dataclasses.replace(
            ff,
            sublayers=[
                replacement,
                intervention,
                down,
            ],
        )

    model = (
        pz.select(gated.model)
        .at_instances_of(
            model_parts.TransformerFeedForward
        )
        .apply(instrument)
    )

    if len(instrumented_layers) != gated.num_layers:
        raise ValueError(
            "Failed to instrument every MLP layer"
        )

    return model, tuple(
        saved[layer]
        for layer in range(gated.num_layers)
    )
