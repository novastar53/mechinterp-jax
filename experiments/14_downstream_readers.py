import csv
from itertools import combinations
from pathlib import Path

import jax
import jax.numpy as jnp

from mechinterp.models import load_smollm2
from mechinterp.interventions import (
    instrument_mlp_neurons,
    with_activation_patch,
)
from mechinterp.activations import (
    capture_final_mlp_activations,
)


SOURCE_LAYER = 26
SOURCE_NEURONS = (829, 920)

EPSILON = 0.25
TOP_K = 50

PROMPTS = {
    "Paris capital": "Paris is the capital of",
    "Paris language": "Paris is a city where people speak",
    "Paris population": "Paris has a population of",
    "Rome capital": "Rome is the capital of",
    "Rome language": "Rome is a city where people speak",
    "Rome population": "Rome has a population of",
}


def cosine_similarity(a, b):
    a = a.ravel()
    b = b.ravel()

    denominator = jnp.linalg.norm(a) * jnp.linalg.norm(b)

    return float(
        jnp.dot(a, b) / jnp.maximum(denominator, 1e-12)
    )


def relative_error(actual, predicted):
    return float(
        jnp.linalg.norm(actual - predicted)
        / jnp.maximum(jnp.linalg.norm(actual), 1e-12)
    )


def top_indices(values, k):
    return jax.lax.top_k(
        jnp.abs(values).ravel(), k
    )[1]


def main():
    loaded = load_smollm2("HuggingFaceTB/SmolLM2-360M")
    tokenizer = loaded.tokenizer

    gated = instrument_mlp_neurons(
        loaded.model,
        expected_layers=loaded.num_layers,
    )

    reader_layers = tuple(
        range(SOURCE_LAYER + 1, gated.num_layers)
    )

    source_indices = jnp.asarray(SOURCE_NEURONS)

    @jax.jit
    def baseline_activations(tokens):
        return capture_final_mlp_activations(
            gated.model,
            tokens,
            (SOURCE_LAYER,) + reader_layers,
        ).astype(jnp.float32)

    @jax.jit
    def readers(tokens, source_values):
        model = with_activation_patch(
            gated,
            SOURCE_LAYER,
            SOURCE_NEURONS,
            source_values,
        )

        return capture_final_mlp_activations(
            model,
            tokens,
            reader_layers,
        ).astype(jnp.float32)

    @jax.jit
    def sensitivities(tokens, source_values, direction):
        return jax.jvp(
            lambda values: readers(tokens, values),
            (source_values,),
            (direction,),
        )

    output_dir = Path("results/experiment_14")
    output_dir.mkdir(parents=True, exist_ok=True)

    sensitivity_vectors = {}

    path = output_dir / "reader_sensitivities.csv"

    with path.open("w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "context",
            "prompt",
            "source_neuron",
            "source_activation",
            "reader_layer",
            "reader_neuron",
            "reader_activation",
            "derivative",
            "finite_difference",
            "ablation_effect",
            "predicted_ablation_effect",
        ])

        for context, prompt in PROMPTS.items():
            tokens = jnp.asarray(
                tokenizer.encode(
                    prompt, add_special_tokens=False
                ),
                dtype=jnp.int32,
            )

            original = baseline_activations(tokens)

            source_values = original[0, source_indices]
            baseline = original[1:]

            print(f"\n{prompt}\n", flush=True)

            for i, source in enumerate(SOURCE_NEURONS):
                direction = jnp.zeros_like(
                    source_values
                ).at[i].set(1.)

                restored, derivative = sensitivities(
                    tokens,
                    source_values,
                    direction,
                )

                if not jnp.allclose(
                    baseline,
                    restored,
                    atol=1e-4,
                    rtol=1e-4,
                ):
                    raise ValueError(
                        "Self-patching changed activations"
                    )

                plus = readers(
                    tokens,
                    source_values + EPSILON * direction,
                )

                minus = readers(
                    tokens,
                    source_values - EPSILON * direction,
                )

                finite_difference = (
                    plus - minus
                ) / (2 * EPSILON)

                ablated = readers(
                    tokens,
                    source_values.at[i].set(0.),
                )

                ablation_effect = ablated - baseline

                predicted_ablation_effect = (
                    -source_values[i] * derivative
                )

                sensitivity_vectors[
                    source, context
                ] = derivative

                print(f"\nSource: L26:{source}")

                print(
                    f"Activation: "
                    f"{float(source_values[i]):+.4f}"
                )

                print(
                    f"JVP/FD cosine similarity: "
                    f"{cosine_similarity(derivative, finite_difference):.6f}"
                )

                print(
                    f"JVP/FD relative error: "
                    f"{relative_error(finite_difference, derivative):.6f}"
                )

                print(
                    f"Ablation prediction cosine: "
                    f"{cosine_similarity(ablation_effect, predicted_ablation_effect):.6f}"
                )

                print(
                    f"Ablation prediction relative error: "
                    f"{relative_error(ablation_effect, predicted_ablation_effect):.6f}"
                )

                print(
                    f"\n{'Reader':<14}"
                    f"{'Activation':>12}"
                    f"{'JVP':>12}"
                    f"{'FD':>12}"
                    f"{'Ablation':>12}"
                )

                arrays = [
                    x.ravel().tolist()
                    for x in jax.device_get((
                        baseline,
                        derivative,
                        finite_difference,
                        ablation_effect,
                        predicted_ablation_effect,
                    ))
                ]

                h, jac, fd, effect, predicted = arrays

                ranked = top_indices(
                    derivative, TOP_K
                ).tolist()

                for idx in ranked[:10]:
                    layer_offset, neuron = divmod(
                        idx, gated.num_neurons
                    )

                    layer = reader_layers[layer_offset]

                    print(
                        f"L{layer}:{neuron:04d}     "
                        f"{h[idx]:+12.4f}"
                        f"{jac[idx]:+12.4f}"
                        f"{fd[idx]:+12.4f}"
                        f"{effect[idx]:+12.4f}"
                    )

                for idx, values in enumerate(zip(*arrays)):
                    layer_offset, neuron = divmod(
                        idx, gated.num_neurons
                    )

                    writer.writerow([
                        context,
                        prompt,
                        source,
                        float(source_values[i]),
                        reader_layers[layer_offset],
                        neuron,
                        *values,
                    ])

                f.flush()

    compare_contexts(
        sensitivity_vectors,
        output_dir,
    )


def compare_contexts(vectors, output_dir):
    path = output_dir / "reader_similarity.csv"

    with path.open("w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "source_neuron",
            "context_a",
            "context_b",
            "cosine_similarity",
            "top_k",
            "shared_readers",
            "overlap_fraction",
        ])

        for source in SOURCE_NEURONS:
            print(f"\nL26:{source} CONTEXT COMPARISON\n")

            for a, b in combinations(PROMPTS, 2):
                va = vectors[source, a]
                vb = vectors[source, b]

                top_a = set(
                    top_indices(va, TOP_K).tolist()
                )

                top_b = set(
                    top_indices(vb, TOP_K).tolist()
                )

                shared = len(top_a & top_b)
                cosine = cosine_similarity(va, vb)

                writer.writerow([
                    source,
                    a,
                    b,
                    cosine,
                    TOP_K,
                    shared,
                    shared / TOP_K,
                ])

                print(
                    f"{a:<20} {b:<20} "
                    f"cos={cosine:+.4f} "
                    f"overlap={shared:2d}/{TOP_K}"
                )


if __name__ == "__main__":
    main()
