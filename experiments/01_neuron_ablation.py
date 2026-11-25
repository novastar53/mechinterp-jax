"""Experiment 1: native MLP neuron attribution and causal ablation.

Run: uv run python experiments/01_neuron_ablation.py
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp

from mechinterp.attribution import (
    ablated_scales,
    gradient_times_activation,
    margin_function,
)
from mechinterp.interventions import instrument_mlp_neurons
from mechinterp.metrics import top_positive_neurons
from mechinterp.models import load_pythia, load_smollm2, single_token_id


def run(
    seed: int, top_k: int, num_controls: int, prompt: str, model: str = "smollm2"
) -> None:
    loaded = load_smollm2() if model == "smollm2" else load_pythia()
    tokenizer = loaded.tokenizer
    target = " Paris"
    foil = " Rome"
    target_id = single_token_id(tokenizer, target)
    foil_id = single_token_id(tokenizer, foil)
    ids = jnp.asarray(tokenizer.encode(prompt, add_special_tokens=False))

    gated = instrument_mlp_neurons(loaded.model, expected_layers=loaded.num_layers)
    scales = gated.ones()
    # The expensive graph is compiled once; subsequent ablations reuse it.
    margin = jax.jit(margin_function(gated, ids, target_id, foil_id))
    baseline, attribution = gradient_times_activation(margin, scales)
    baseline = float(baseline)
    selected = top_positive_neurons(attribution, top_k)

    print(f"Model: {loaded.model_id}")
    print(f"Prompt: {prompt!r}")
    print(f"Target/foil: {target!r} / {foil!r}")
    print(f"MLP gates: {gated.num_layers} x {gated.num_neurons}")
    print(f"Baseline margin (Paris - Rome): {baseline:+.5f}")
    print("\nTop positive gradient x activation candidates:")
    for layer, neuron, score in selected:
        actual = float(margin(ablated_scales(scales, [(layer, neuron)])))
        print(
            f"  layer={layer:02d} neuron={neuron:04d} "
            f"attribution={score:+.5f} actual_delta={actual - baseline:+.5f}"
        )

    if not selected:
        print("No positive attributions were found; try a different prompt.")
        return

    selected_indices = [(layer, neuron) for layer, neuron, _ in selected]
    after_group = float(margin(ablated_scales(scales, selected_indices)))
    print(f"\nSelected-group delta: {after_group - baseline:+.5f}")

    # Random sets of the same size, for a simple causal control.
    key = jax.random.key(seed)
    n_total = scales.size
    random_deltas = []
    for _ in range(num_controls):
        key, subkey = jax.random.split(key)
        flat_ids = jax.random.choice(
            subkey, n_total, shape=(len(selected_indices),), replace=False
        )
        control = [
            (int(i) // gated.num_neurons, int(i) % gated.num_neurons)
            for i in flat_ids.tolist()
        ]
        result = float(margin(ablated_scales(scales, control)))
        random_deltas.append(result - baseline)
    if random_deltas:
        values = jnp.asarray(random_deltas)
        print(
            f"Random-group delta, n={num_controls}: "
            f"mean={float(jnp.mean(values)):+.5f} "
            f"std={float(jnp.std(values)):.5f}"
        )
        print("Random deltas:", [round(float(x), 5) for x in values])

    print("\nNote: large effects do not identify a neuron's semantic meaning.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--num-controls", type=int, default=8)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--model", choices=["smollm2", "pythia"], default="smollm2")
    args = parser.parse_args()
    run(args.seed, args.top_k, args.num_controls, args.prompt, args.model)
