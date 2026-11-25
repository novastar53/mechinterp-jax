"""Explicit Penzai model surgery for native MLP neurons."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from penzai import pz
from penzai.models.transformer import model_parts


def scale_final_position(activations: jax.Array, scales: jax.Array) -> jax.Array:
    """Scale MLP neuron outputs only at the last token position.

    activations has shape (sequence, neurons), scales has shape (neurons,).
    A scale of 1 is the original model; a scale of 0 ablates that neuron.
    """
    if activations.ndim != 2 or scales.ndim != 1:
        raise ValueError("Expected (sequence, neurons) and (neurons,) arrays")
    if activations.shape[1] != scales.shape[0]:
        raise ValueError("The number of neuron scales must match activations")
    return activations.at[-1, :].multiply(scales)


@pz.pytree_dataclass
class FinalTokenNeuronGate(pz.nn.Layer):
    """Multiply intermediate MLP neuron outputs by a vector of intervention scales.

    This small first experiment accepts unbatched prompts only. Penzai names
    the relevant activation axes "seq" and "neurons".
    """

    layer_index: int = dataclasses.field(metadata={"pytree_node": False})
    scales: jax.Array

    def __call__(self, x: pz.nx.NamedArray, **_unused_side_inputs) -> pz.nx.NamedArray:
        values = x.unwrap("seq", "neurons")
        edited = scale_final_position(values, self.scales)
        return pz.nx.wrap(edited, "seq", "neurons")


@dataclass(frozen=True)
class GatedModel:
    model: model_parts.TransformerLM
    num_layers: int
    num_neurons: int

    def ones(self) -> jax.Array:
        return jnp.ones((self.num_layers, self.num_neurons), dtype=jnp.float32)


def instrument_mlp_neurons(
    model: model_parts.TransformerLM, *, expected_layers: int | None = None
) -> GatedModel:
    """Install gates immediately before each MLP's output projection.

    Supports non-stacked GPT-NeoX (post-GELU) and Llama-like (post-GLU
    multiplication) MLPs. Layer indices follow decoder-block traversal order.
    """
    found_layers: list[int] = []
    n_neurons = int(model.metadata.mlp_hidden_dim)

    def add_gate(ff: model_parts.TransformerFeedForward):
        layers = ff.sublayers
        is_neox = len(layers) == 3 and isinstance(layers[1], pz.nn.Elementwise)
        is_glu = (
            len(layers) == 2
            and isinstance(layers[0], pz.nn.BranchAndMultiplyTogether)
            and isinstance(layers[1], pz.nn.Linear)
        )
        if not (is_neox or is_glu):
            raise ValueError("Unsupported MLP structure; expected NeoX or Llama-like")
        layer_index = len(found_layers)
        found_layers.append(layer_index)
        gate = FinalTokenNeuronGate(
            layer_index=layer_index,
            scales=jnp.ones((n_neurons,), dtype=jnp.float32),
        )
        return dataclasses.replace(
            ff, sublayers=[*layers[:-1], gate, layers[-1]]
        )

    patched = (
        pz.select(model)
        .at_instances_of(model_parts.TransformerFeedForward)
        .apply(add_gate)
    )
    if not found_layers:
        raise ValueError("No supported MLP blocks found; check model architecture")
    if expected_layers is not None and len(found_layers) != expected_layers:
        raise ValueError(
            f"Expected {expected_layers} MLP blocks, got {len(found_layers)}"
        )
    return GatedModel(
        model=patched, num_layers=len(found_layers), num_neurons=n_neurons
    )


def with_scales(gated: GatedModel, scales: jax.Array) -> model_parts.TransformerLM:
    """Return a modified *copy* of the model with the supplied neuron gates."""
    if scales.shape != (gated.num_layers, gated.num_neurons):
        raise ValueError("Expected gate scales with shape (layers, neurons)")
    return (
        pz.select(gated.model)
        .at_instances_of(FinalTokenNeuronGate)
        .apply(
            lambda gate: dataclasses.replace(
                gate, scales=scales[gate.layer_index]
            )
        )
    )



def patch_final_position(
    activations: jax.Array,
    indices: jax.Array,
    values: jax.Array,
) -> jax.Array:
    return activations.at[-1, indices].set(
        values.astype(activations.dtype)
    )


@pz.pytree_dataclass
class FinalTokenNeuronPatch(pz.nn.Layer):
    indices: jax.Array
    values: jax.Array

    def __call__(self, x, **kwargs):
        activations = x.unwrap("seq", "neurons")

        edited = patch_final_position(
            activations,
            self.indices,
            self.values,
        )

        return pz.nx.wrap(edited, "seq", "neurons")


def with_activation_patch(
    gated: GatedModel,
    layer_index: int,
    neuron_indices: tuple[int, ...],
    values: jax.Array,
):
    if not 0 <= layer_index < gated.num_layers:
        raise ValueError("Invalid layer index")

    if any(
        not 0 <= i < gated.num_neurons
        for i in neuron_indices
    ):
        raise ValueError("Invalid neuron index")

    if len(set(neuron_indices)) != len(neuron_indices):
        raise ValueError("Duplicate neuron indices")

    if values.shape != (len(neuron_indices),):
        raise ValueError("Incorrect replacement shape")

    indices = jnp.asarray(
        neuron_indices, dtype=jnp.int32
    )

    def replace(gate):
        if gate.layer_index != layer_index:
            return gate

        return pz.nn.Sequential([
            gate,
            FinalTokenNeuronPatch(
                indices=indices,
                values=values,
            ),
        ])

    return (
        pz.select(gated.model)
        .at_instances_of(FinalTokenNeuronGate)
        .apply(replace)
    )



@pz.pytree_dataclass
class FinalTokenNeuronOffset(pz.nn.Layer):
    offsets: jax.Array

    def __call__(self, x, **kwargs):
        activations = x.unwrap("seq", "neurons")

        edited = activations.at[-1].add(
            self.offsets.astype(activations.dtype)
        )

        return pz.nx.wrap(edited, "seq", "neurons")


def with_activation_offsets(
    gated: GatedModel,
    layer_indices: tuple[int, ...],
    offsets: jax.Array,
):
    if offsets.shape != (
        len(layer_indices), gated.num_neurons
    ):
        raise ValueError("Incorrect offset shape")

    if len(set(layer_indices)) != len(layer_indices):
        raise ValueError("Duplicate layer indices")

    if any(
        not 0 <= layer < gated.num_layers
        for layer in layer_indices
    ):
        raise ValueError("Invalid layer index")

    lookup = {
        layer: i
        for i, layer in enumerate(layer_indices)
    }

    def replace(gate):
        if gate.layer_index not in lookup:
            return gate

        return pz.nn.Sequential([
            gate,
            FinalTokenNeuronOffset(
                offsets=offsets[
                    lookup[gate.layer_index]
                ]
            ),
        ])

    return (
        pz.select(gated.model)
        .at_instances_of(FinalTokenNeuronGate)
        .apply(replace)
    )
