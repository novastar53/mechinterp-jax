from typing import NamedTuple

import jax
import jax.numpy as jnp

from jax.scipy.fft import dct, idct
from penzai import pz

from mechinterp.interventions import (
    GatedModel,
    FinalTokenNeuronGate,
)


class Rotation(NamedTuple):
    permutations: jax.Array
    inverse_permutations: jax.Array
    signs: jax.Array


def make_rotation(key, num_neurons, rounds=2):
    keys = jax.random.split(key, 2 * rounds)

    permutations = jnp.stack([
        jax.random.permutation(keys[2 * i], num_neurons)
        for i in range(rounds)
    ])

    signs = jnp.stack([
        jax.random.rademacher(
            keys[2 * i + 1],
            shape=(num_neurons,),
            dtype=jnp.float32,
        )
        for i in range(rounds)
    ])

    return Rotation(
        permutations=permutations,
        inverse_permutations=jnp.argsort(
            permutations, axis=-1
        ),
        signs=signs,
    )


def rotate(x, rotation):
    for i in range(rotation.signs.shape[0]):
        x = x[..., rotation.permutations[i]]
        x = x * rotation.signs[i]
        x = dct(x, type=2, axis=-1, norm="ortho")

    return x


def inverse_rotate(x, rotation):
    for i in reversed(range(rotation.signs.shape[0])):
        x = idct(x, type=2, axis=-1, norm="ortho")
        x = x * rotation.signs[i]
        x = x[..., rotation.inverse_permutations[i]]

    return x


def topk_mask(x, k):
    order = jnp.argsort(
        -jnp.abs(x), axis=-1, stable=True
    )
    ranks = jnp.argsort(order, axis=-1)

    return ranks < k


def ablate_coordinates(x, k, retain):
    mask = topk_mask(x, k)
    mask = jnp.where(retain, mask, ~mask)

    return jnp.where(mask, x, 0)


def basis_ablation(x, rotation, rotated, k, retain):
    dtype = x.dtype
    x = x.astype(jnp.float32)

    def native_branch(x):
        return ablate_coordinates(x, k, retain)

    def rotated_branch(x):
        z = rotate(x, rotation)
        z = ablate_coordinates(z, k, retain)

        return inverse_rotate(z, rotation)

    result = jax.lax.cond(
        rotated,
        rotated_branch,
        native_branch,
        x,
    )

    return result.astype(dtype)


@pz.pytree_dataclass
class BasisAblation(pz.nn.Layer):
    rotation: Rotation
    rotated: jax.Array
    k: jax.Array
    retain: jax.Array

    def __call__(self, x, **kwargs):
        activations = x.unwrap("seq", "neurons")

        activations = basis_ablation(
            activations,
            self.rotation,
            self.rotated,
            self.k,
            self.retain,
        )

        return pz.nx.wrap(
            activations, "seq", "neurons"
        )


def make_layer_rotations(key, num_layers, num_neurons):
    keys = jax.random.split(key, num_layers)

    return jax.vmap(
        lambda key: make_rotation(key, num_neurons)
    )(keys)


def with_basis_ablation(
    gated: GatedModel,
    rotations: Rotation,
    rotated,
    k,
    retain,
):
    def instrument(gate):
        layer_index = gate.layer_index

        rotation = jax.tree_util.tree_map(
            lambda x: x[layer_index],
            rotations,
        )

        return pz.nn.Sequential([
            gate,
            BasisAblation(
                rotation=rotation,
                rotated=rotated,
                k=k,
                retain=retain,
            ),
        ])

    return (
        pz.select(gated.model)
        .at_instances_of(FinalTokenNeuronGate)
        .apply(instrument)
    )