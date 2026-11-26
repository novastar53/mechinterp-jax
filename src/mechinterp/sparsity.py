import jax
import jax.numpy as jnp

from penzai import pz

from mechinterp.interventions import (
    GatedModel,
    FinalTokenNeuronGate,
)


TOP = 0
RANDOM = 1
BOTTOM = 2


def sparsify_activations(
    activations: jax.Array,
    keep: jax.Array,
    mode: jax.Array,
    key: jax.Array,
) -> jax.Array:
    num_neurons = activations.shape[-1]

    def sparsify(x):
        scores = jax.lax.switch(
            mode,
            (
                lambda: jnp.abs(x),
                lambda: jax.random.uniform(
                    key, x.shape, dtype=jnp.float32
                ),
                lambda: -jnp.abs(x),
            ),
        )

        order = jnp.argsort(
            -scores, axis=-1, stable=True
        )

        retained = jnp.broadcast_to(
            jnp.arange(num_neurons) < keep,
            x.shape,
        )

        rows = jnp.arange(x.shape[0])[:, None]

        mask = jnp.zeros_like(
            x, dtype=jnp.bool_
        ).at[rows, order].set(retained)

        return jnp.where(mask, x, 0)

    return jax.lax.cond(
        keep >= num_neurons,
        lambda x: x,
        sparsify,
        activations,
    )


@pz.pytree_dataclass
class TokenwiseSparsity(pz.nn.Layer):
    keep: jax.Array
    mode: jax.Array
    key: jax.Array

    def __call__(self, x, **kwargs):
        activations = x.unwrap("seq", "neurons")

        sparse = sparsify_activations(
            activations,
            self.keep,
            self.mode,
            self.key,
        )

        return pz.nx.wrap(
            sparse, "seq", "neurons"
        )


def with_mlp_sparsity(
    gated: GatedModel,
    keep: jax.Array,
    mode: jax.Array,
    key: jax.Array,
):
    def instrument(gate):
        return pz.nn.Sequential([
            gate,
            TokenwiseSparsity(
                keep=keep,
                mode=mode,
                key=jax.random.fold_in(
                    key, gate.layer_index
                ),
            ),
        ])

    return (
        pz.select(gated.model)
        .at_instances_of(FinalTokenNeuronGate)
        .apply(instrument)
    )
