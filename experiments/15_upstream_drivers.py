import csv
from itertools import combinations
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from mechinterp.models import load_smollm2
from mechinterp.interventions import (
    instrument_mlp_neurons,
    with_activation_offsets,
)
from mechinterp.activations import (
    capture_swiglu_activations,
    capture_final_mlp_activations,
)


TARGET_LAYER = 26
TARGET_NEURONS = (829, 920)
COMPONENTS = ("gate", "value")

TOP_K = 50
EPSILON = 0.25

PROMPTS = {
    "Paris capital": "Paris is the capital of",
    "Paris language": "Paris is a city where people speak",
    "Paris population": "Paris has a population of",
    "Rome capital": "Rome is the capital of",
    "Rome language": "Rome is a city where people speak",
    "Rome population": "Rome has a population of",
}


def save_csv(path, records):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(records[0])
        )
        writer.writeheader()
        writer.writerows(records)


def cosine_similarity(a, b):
    a = a.ravel()
    b = b.ravel()

    denominator = (
        jnp.linalg.norm(a) * jnp.linalg.norm(b)
    )

    return float(
        jnp.dot(a, b)
        / jnp.maximum(denominator, 1e-12)
    )


def top_indices(values, k=TOP_K):
    return jax.lax.top_k(
        jnp.abs(values).ravel(), k
    )[1].tolist()


def overlap(a, b):
    return len(
        set(top_indices(a)) & set(top_indices(b))
    )


def main():
    loaded = load_smollm2("HuggingFaceTB/SmolLM2-360M")
    tokenizer = loaded.tokenizer

    gated = instrument_mlp_neurons(
        loaded.model,
        expected_layers=loaded.num_layers,
    )

    upstream_layers = tuple(range(TARGET_LAYER))
    target_indices = jnp.asarray(TARGET_NEURONS)

    zero_offsets = jnp.zeros(
        (len(upstream_layers), gated.num_neurons),
        dtype=jnp.float32,
    )

    @jax.jit
    def upstream_activations(tokens):
        return capture_final_mlp_activations(
            gated.model,
            tokens,
            upstream_layers,
        ).astype(jnp.float32)

    @jax.jit
    def target_projections(tokens, offsets):
        modified = with_activation_offsets(
            gated,
            upstream_layers,
            offsets,
        )

        captured = capture_swiglu_activations(
            gated,
            tokens,
            TARGET_LAYER,
            model=modified,
        )

        return jnp.stack([
            captured.gate[-1, target_indices],
            captured.value[-1, target_indices],
        ], axis=-1).astype(jnp.float32)

    @jax.jit
    def sensitivities(tokens, offsets):
        outputs, pullback = jax.vjp(
            lambda delta: target_projections(
                tokens, delta
            ),
            offsets,
        )

        basis = jnp.eye(
            outputs.size,
            dtype=outputs.dtype,
        ).reshape((-1,) + outputs.shape)

        gradients = jax.lax.map(
            lambda cotangent: pullback(cotangent)[0],
            basis,
        )

        gradients = gradients.reshape(
            outputs.shape + offsets.shape
        )

        return outputs, gradients

    output_dir = Path("results/experiment_15")
    output_dir.mkdir(parents=True, exist_ok=True)

    driver_records = []
    validation_records = []
    summary_records = []
    vectors = {}

    for context, prompt in PROMPTS.items():
        print(f"\n{prompt}\n", flush=True)

        tokens = jnp.asarray(
            tokenizer.encode(
                prompt, add_special_tokens=False
            ),
            dtype=jnp.int32,
        )

        upstream = upstream_activations(tokens)

        outputs, gradients = sensitivities(
            tokens, zero_offsets
        )

        original = capture_swiglu_activations(
            gated, tokens, TARGET_LAYER
        )

        expected = jnp.stack([
            original.gate[-1, target_indices],
            original.value[-1, target_indices],
        ], axis=-1)

        if not jnp.allclose(
            outputs, expected,
            atol=1e-4, rtol=1e-4,
        ):
            raise ValueError(
                "Zero-offset intervention changed projections"
            )

        contributions = (
            gradients * upstream[None, None, :, :]
        )

        np.savez_compressed(
            output_dir / (
                context.replace(" ", "_").lower()
                + ".npz"
            ),
            upstream=np.asarray(upstream),
            outputs=np.asarray(outputs),
            gradients=np.asarray(gradients),
            contributions=np.asarray(contributions),
            upstream_layers=np.asarray(upstream_layers),
            target_neurons=np.asarray(TARGET_NEURONS),
            components=np.asarray(COMPONENTS),
        )

        validation_candidates = set()

        for i, target in enumerate(TARGET_NEURONS):
            gate, value = map(float, outputs[i])

            activation = float(
                jax.nn.silu(outputs[i, 0])
                * outputs[i, 1]
            )

            print(
                f"L26:{target} "
                f"g={gate:+.4f} "
                f"v={value:+.4f} "
                f"h={activation:+.4f}"
            )

            for j, component in enumerate(COMPONENTS):
                gradient = gradients[i, j]
                contribution = contributions[i, j]

                vectors[
                    context, target, component
                ] = (gradient, contribution)

                summary_records.append({
                    "context": context,
                    "target_neuron": target,
                    "component": component,
                    "projection": float(outputs[i, j]),
                    "gradient_norm": float(
                        jnp.linalg.norm(gradient)
                    ),
                    "contribution_norm": float(
                        jnp.linalg.norm(contribution)
                    ),
                })

                for ranking, scores in [
                    ("sensitivity", gradient),
                    ("contribution", contribution),
                ]:
                    ranked = top_indices(scores)
                    validation_candidates.add(ranked[0])

                    print(
                        f"\n{component.upper()} — "
                        f"{ranking.upper()}"
                    )

                    print(
                        f"{'Driver':<14}"
                        f"{'Activation':>12}"
                        f"{'Gradient':>12}"
                        f"{'Contribution':>14}"
                    )

                    for rank, idx in enumerate(ranked, 1):
                        layer, neuron = divmod(
                            idx, gated.num_neurons
                        )

                        h = float(upstream[layer, neuron])
                        g = float(gradient[layer, neuron])
                        c = float(
                            contribution[layer, neuron]
                        )

                        driver_records.append({
                            "context": context,
                            "prompt": prompt,
                            "target_layer": TARGET_LAYER,
                            "target_neuron": target,
                            "component": component,
                            "ranking": ranking,
                            "rank": rank,
                            "driver_layer": layer,
                            "driver_neuron": neuron,
                            "driver_activation": h,
                            "gradient": g,
                            "contribution": c,
                        })

                        if rank <= 5:
                            print(
                                f"L{layer}:{neuron:04d}     "
                                f"{h:+12.4f}"
                                f"{g:+12.4f}"
                                f"{c:+14.4f}"
                            )

            print()

        for idx in sorted(validation_candidates):
            layer, neuron = divmod(
                idx, gated.num_neurons
            )

            direction = zero_offsets.at[
                layer, neuron
            ].set(1.)

            plus = target_projections(
                tokens,
                EPSILON * direction,
            )

            minus = target_projections(
                tokens,
                -EPSILON * direction,
            )

            finite_difference = (
                plus - minus
            ) / (2 * EPSILON)

            ablated = target_projections(
                tokens,
                -upstream[layer, neuron] * direction,
            )

            actual_effect = ablated - outputs

            predicted_effect = (
                -contributions[:, :, layer, neuron]
            )

            for i, target in enumerate(TARGET_NEURONS):
                for j, component in enumerate(COMPONENTS):
                    validation_records.append({
                        "context": context,
                        "target_neuron": target,
                        "component": component,
                        "driver_layer": layer,
                        "driver_neuron": neuron,
                        "gradient": float(
                            gradients[i, j, layer, neuron]
                        ),
                        "finite_difference": float(
                            finite_difference[i, j]
                        ),
                        "actual_ablation_effect": float(
                            actual_effect[i, j]
                        ),
                        "predicted_ablation_effect": float(
                            predicted_effect[i, j]
                        ),
                    })

    save_csv(
        output_dir / "upstream_drivers.csv",
        driver_records,
    )

    save_csv(
        output_dir / "driver_validation.csv",
        validation_records,
    )

    save_csv(
        output_dir / "projection_summary.csv",
        summary_records,
    )

    compare_drivers(vectors, output_dir)


def compare_drivers(vectors, output_dir):
    records = []

    def compare(target, a, b):
        va = vectors[a[0], target, a[1]]
        vb = vectors[b[0], target, b[1]]

        for ranking, x, y in [
            ("sensitivity", va[0], vb[0]),
            ("contribution", va[1], vb[1]),
        ]:
            records.append({
                "target_neuron": target,
                "context_a": a[0],
                "component_a": a[1],
                "context_b": b[0],
                "component_b": b[1],
                "ranking": ranking,
                "cosine_similarity": cosine_similarity(x, y),
                "shared_top_k": overlap(x, y),
                "top_k": TOP_K,
            })

    for target in TARGET_NEURONS:
        print(f"\nL26:{target} GATE / VALUE COMPARISON\n")

        for context in PROMPTS:
            compare(
                target,
                (context, "gate"),
                (context, "value"),
            )

            gate = vectors[context, target, "gate"][0]
            value = vectors[context, target, "value"][0]

            print(
                f"{context:<20}"
                f"cos={cosine_similarity(gate, value):+.4f} "
                f"overlap={overlap(gate, value):2d}/{TOP_K}"
            )

        for component in COMPONENTS:
            for a, b in combinations(PROMPTS, 2):
                compare(
                    target,
                    (a, component),
                    (b, component),
                )

    save_csv(
        output_dir / "driver_similarity.csv",
        records,
    )


if __name__ == "__main__":
    main()
