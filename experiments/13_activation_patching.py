import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from penzai import pz

from mechinterp.models import load_smollm2, single_token_id
from mechinterp.interventions import (
    instrument_mlp_neurons,
    with_activation_patch,
)
from mechinterp.activations import capture_swiglu_activations
from mechinterp.attribution import next_token_logits_function


LAYER = 26
NEURONS = (829, 920)
ALPHAS = [0., 0.25, 0.5, 0.75, 1.]

PROMPTS = {
    "Paris": "Paris is the capital of",
    "Rome": "Rome is the capital of",
}


def main():
    loaded = load_smollm2("HuggingFaceTB/SmolLM2-360M")
    tokenizer = loaded.tokenizer

    gated = instrument_mlp_neurons(
        loaded.model,
        expected_layers=loaded.num_layers,
    )

    france_id = single_token_id(tokenizer, " France")
    italy_id = single_token_id(tokenizer, " Italy")

    indices = jnp.asarray(NEURONS)

    @jax.jit
    def capture(tokens):
        captured = capture_swiglu_activations(
            gated, tokens, LAYER
        )
        return captured.activation[-1, indices]

    @jax.jit
    def patched_logits(tokens, values):
        model = with_activation_patch(
            gated,
            LAYER,
            NEURONS,
            values,
        )

        tokens = pz.nx.wrap(tokens, "seq")

        return model(tokens).unwrap(
            "seq", "vocabulary"
        )[-1].astype(jnp.float32)

    @jax.jit
    def original_logits(tokens):
        return next_token_logits_function(
            gated, tokens
        )(gated.ones()).astype(jnp.float32)

    token_ids = {
        city: jnp.asarray(
            tokenizer.encode(
                prompt, add_special_tokens=False
            ),
            dtype=jnp.int32,
        )
        for city, prompt in PROMPTS.items()
    }

    activations = {
        city: capture(tokens)
        for city, tokens in token_ids.items()
    }

    print("Original activations:")

    for city, values in activations.items():
        print(
            f"{city}: "
            f"829={float(values[0]):+.4f}, "
            f"920={float(values[1]):+.4f}"
        )

        baseline = original_logits(token_ids[city])
        restored = patched_logits(token_ids[city], values)

        if not jnp.allclose(
            baseline, restored, atol=1e-4, rtol=1e-4
        ):
            raise ValueError("Self-patching changed logits")

    records = []

    for destination, source in [
        ("Rome", "Paris"),
        ("Paris", "Rome"),
    ]:
        tokens = token_ids[destination]

        dest_h = activations[destination]
        source_h = activations[source]

        print(f"\n{source} -> {destination}\n")

        for neuron_idx, neuron in enumerate(NEURONS):
            print(f"Neuron {neuron}")

            for alpha in ALPHAS:
                interpolated = (
                    dest_h[neuron_idx]
                    + alpha * (
                        source_h[neuron_idx]
                        - dest_h[neuron_idx]
                    )
                )

                replacements = dest_h.at[
                    neuron_idx
                ].set(interpolated)

                logits = patched_logits(
                    tokens, replacements
                )

                probs = jax.nn.softmax(logits)

                p_france = float(probs[france_id])
                p_italy = float(probs[italy_id])

                margin = float(
                    logits[france_id] - logits[italy_id]
                )

                prediction = tokenizer.decode([
                    int(jnp.argmax(logits))
                ])

                records.append({
                    "source": source,
                    "destination": destination,
                    "neuron": neuron,
                    "alpha": alpha,
                    "activation": float(interpolated),
                    "p_france": p_france,
                    "p_italy": p_italy,
                    "margin": margin,
                    "prediction": prediction,
                })

                print(
                    f"alpha={alpha:.2f} "
                    f"h={float(interpolated):+8.3f} "
                    f"France={p_france:8.3%} "
                    f"Italy={p_italy:8.3%} "
                    f"margin={margin:+8.3f} "
                    f"pred={prediction!r}",
                    flush=True,
                )

            print()

    output_dir = Path("results/experiment_13")
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / "activation_patching.csv"

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(records[0])
        )
        writer.writeheader()
        writer.writerows(records)

    plot_results(records, output_dir)


def plot_results(records, output_dir):
    fig, axes = plt.subplots(
        2, 2, figsize=(12, 9)
    )

    for i, destination in enumerate(["Rome", "Paris"]):
        for j, neuron in enumerate(NEURONS):
            rows = [
                r for r in records
                if r["destination"] == destination
                and r["neuron"] == neuron
            ]

            ax = axes[i, j]

            for country in ["france", "italy"]:
                ax.plot(
                    [r["alpha"] for r in rows],
                    [100 * r[f"p_{country}"] for r in rows],
                    marker="o",
                    label=country.title(),
                )

            ax.set_title(
                f"{rows[0]['source']} → {destination}, "
                f"L26:{neuron}"
            )

            ax.set_xlabel("Interpolation coefficient")
            ax.set_ylabel("Probability (%)")
            ax.set_ylim(0, 100)
            ax.legend()
            ax.grid(alpha=0.2)

    plt.tight_layout()

    fig.savefig(
        output_dir / "activation_patching.png",
        dpi=200,
    )

    plt.close(fig)


if __name__ == "__main__":
    main()
