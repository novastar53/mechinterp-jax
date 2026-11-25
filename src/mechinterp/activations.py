import dataclasses
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from penzai import pz
from penzai.models.transformer import model_parts

from mechinterp.interventions import (
    GatedModel,
    FinalTokenNeuronGate,
)


@pz.pytree_dataclass
class SaveActivation(pz.nn.Layer):
    saved: pz.StateVariable

    def __call__(self, x, **kwargs):
        self.saved.value = x
        return x


@dataclass(frozen=True)
class SwiGLUActivations:
    gate: jax.Array
    value: jax.Array
    activation: jax.Array


def capture_swiglu_activations(
    gated: GatedModel,
    token_ids: jax.Array,
    layer_index: int,
) -> SwiGLUActivations:
    if not 0 <= layer_index < gated.num_layers:
        raise ValueError("Invalid layer index")

    gate_saved = pz.StateVariable(None)
    value_saved = pz.StateVariable(None)
    activation_saved = pz.StateVariable(None)

    def instrument(ff):
        layers = ff.sublayers

        if len(layers) != 3:
            return ff

        glu, intervention, down = layers

        if not isinstance(intervention, FinalTokenNeuronGate):
            return ff

        if intervention.layer_index != layer_index:
            return ff

        if not isinstance(glu, pz.nn.BranchAndMultiplyTogether):
            raise ValueError("Expected a gated MLP")

        gate_branch, value_branch = glu.branches
        gate_linear, nonlinearity = gate_branch.sublayers

        gate_branch = dataclasses.replace(
            gate_branch,
            sublayers=[
                gate_linear,
                SaveActivation(gate_saved),
                nonlinearity,
            ],
        )

        value_branch = pz.nn.Sequential([
            value_branch,
            SaveActivation(value_saved),
        ])

        glu = dataclasses.replace(
            glu,
            branches=[gate_branch, value_branch],
        )

        return dataclasses.replace(
            ff,
            sublayers=[
                glu,
                SaveActivation(activation_saved),
                intervention,
                down,
            ],
        )

    model = (
        pz.select(gated.model)
        .at_instances_of(model_parts.TransformerFeedForward)
        .apply(instrument)
    )

    tokens = pz.nx.wrap(token_ids, "seq")
    _ = model(tokens)

    if activation_saved.value is None:
        raise ValueError("No matching MLP was instrumented")

    return SwiGLUActivations(
        gate=gate_saved.value.unwrap("seq", "neurons"),
        value=value_saved.value.unwrap("seq", "neurons"),
        activation=activation_saved.value.unwrap(
            "seq", "neurons"
        ),
    )