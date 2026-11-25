import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from penzai import pz

from mechinterp.models import load_smollm2
from mechinterp.interventions import instrument_mlp_neurons
from mechinterp.activations import capture_swiglu_activations


CITIES = {
    "Paris": ("France", "French"),
    "Rome": ("Italy", "Italian"),
    "Berlin": ("Germany", "German"),
    "Madrid": ("Spain", "Spanish"),
    "Lisbon": ("Portugal", "Portuguese"),
    "London": ("United Kingdom", "English"),
    "Tokyo": ("Japan", "Japanese"),
}

TEMPLATES = {
    "Entity": "{}",
    "Copula": "{} is",
    "Description": "{} is a",
    "Capital": "{} is the capital of",
    "Location": "{} is located in",
    "Language": "{} is a city where people speak",
    "Population": "{} has a population of",
    "Explanation": "{} is beautiful because",
    "Spelling": "{} is spelled",
    "Neutral": "{} is mentioned in detail",
}

LAYER = 26
NEURONS = [829, 920]
SEQ_LEN = 32


def save_csv(path, records):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(records[0])
        )
        writer.writeheader()
        writer.writerows(records)


def main():
    loaded = load_smollm2("HuggingFaceTB/SmolLM2-360M")
    tokenizer = loaded.tokenizer

    gated = instrument_mlp_neurons(
        loaded.model,
        expected_layers=loaded.num_layers,
    )

    indices = jnp.asarray(NEURONS)

    @jax.jit
    def capture(tokens):
        captured = capture_swiglu_activations(
            gated, tokens, LAYER
        )

        return jnp.stack(
            [
                captured.gate[:, indices],
                captured.value[:, indices],
                captured.activation[:, indices],
            ],
            axis=-1,
        )

    @jax.jit
    def predict(tokens, position):
        named_tokens = pz.nx.wrap(tokens, "seq")

        logits = gated.model(named_tokens).unwrap(
            "seq", "vocabulary"
        )

        return jax.nn.softmax(
            logits[position].astype(jnp.float32)
        )

    token_records = []
    summary_records = []

    for city, (country, language) in CITIES.items():
        print(f"\n{city.upper()}\n")

        for category, template in TEMPLATES.items():
            prompt = template.format(city)

            ids = tokenizer.encode(
                prompt, add_special_tokens=False
            )

            length = len(ids)

            if length > SEQ_LEN:
                raise ValueError("Sequence too long")

            padded = ids + [
                tokenizer.eos_token_id
            ] * (SEQ_LEN - length)

            tokens = jnp.asarray(
                padded, dtype=jnp.int32
            )

            values = jax.device_get(
                capture(tokens)
            )[:length]

            probs = predict(
                tokens, jnp.int32(length - 1)
            )

            top_probs, top_ids = jax.lax.top_k(probs, 3)

            predictions = [
                tokenizer.decode([int(i)])
                for i in top_ids
            ]

            expected = {
                "Capital": country,
                "Language": language,
            }.get(category)

            expected_prob = None

            if expected is not None:
                target_ids = tokenizer.encode(
                    " " + expected,
                    add_special_tokens=False,
                )

                if len(target_ids) == 1:
                    expected_prob = float(
                        probs[target_ids[0]]
                    )

            for pos, token_id in enumerate(ids):
                token = tokenizer.decode([token_id])

                for j, neuron in enumerate(NEURONS):
                    g, v, h = map(float, values[pos, j])

                    token_records.append({
                        "city": city,
                        "category": category,
                        "prompt": prompt,
                        "position": pos,
                        "token_id": token_id,
                        "token": token,
                        "layer": LAYER,
                        "neuron": neuron,
                        "gate": g,
                        "value": v,
                        "activation": h,
                    })

            for j, neuron in enumerate(NEURONS):
                g, v, h = map(float, values[-1, j])

                summary_records.append({
                    "city": city,
                    "category": category,
                    "prompt": prompt,
                    "neuron": neuron,
                    "gate": g,
                    "value": v,
                    "activation": h,
                    "prediction_1": predictions[0],
                    "probability_1": float(top_probs[0]),
                    "prediction_2": predictions[1],
                    "probability_2": float(top_probs[1]),
                    "prediction_3": predictions[2],
                    "probability_3": float(top_probs[2]),
                    "expected": expected,
                    "expected_probability": expected_prob,
                })

            h829 = float(values[-1, 0, 2])
            h920 = float(values[-1, 1, 2])

            print(
                f"{category:<14} "
                f"829={h829:+8.3f} "
                f"920={h920:+8.3f} "
                f"Prediction={predictions[0]!r} "
                f"({float(top_probs[0]):.2%})",
                flush=True,
            )

    output_dir = Path("results/experiment_12")
    output_dir.mkdir(parents=True, exist_ok=True)

    save_csv(
        output_dir / "token_activations.csv",
        token_records,
    )

    save_csv(
        output_dir / "final_activations.csv",
        summary_records,
    )

    plot_summary(summary_records, output_dir)
    plot_trajectories(token_records, output_dir)


def plot_summary(records, output_dir):
    lookup = {
        (r["neuron"], r["city"], r["category"]): r
        for r in records
    }

    cities = list(CITIES)
    categories = list(TEMPLATES)
    components = ["gate", "value", "activation"]

    fig, axes = plt.subplots(
        2, 3, figsize=(16, 10)
    )

    for i, neuron in enumerate(NEURONS):
        for j, component in enumerate(components):
            matrix = jnp.asarray([
                [
                    lookup[
                        neuron, city, category
                    ][component]
                    for city in cities
                ]
                for category in categories
            ])

            vmax = max(
                float(jnp.max(jnp.abs(matrix))),
                1e-8,
            )

            ax = axes[i, j]

            im = ax.imshow(
                matrix,
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
                aspect="auto",
            )

            ax.set_title(f"L26:{neuron} — {component}")

            ax.set_xticks(
                range(len(cities)),
                cities,
                rotation=45,
                ha="right",
            )

            ax.set_yticks(
                range(len(categories)),
                categories,
            )

            for y in range(len(categories)):
                for x in range(len(cities)):
                    val = float(matrix[y, x])

                    ax.text(
                        x, y, f"{val:+.1f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=(
                            "white"
                            if abs(val) > 0.6 * vmax
                            else "black"
                        ),
                    )

            fig.colorbar(im, ax=ax)

    plt.tight_layout()

    fig.savefig(
        output_dir / "final_activation_heatmaps.png",
        dpi=200,
    )

    plt.close(fig)


def plot_trajectories(records, output_dir):
    for city, neuron in [
        ("Paris", 829),
        ("Rome", 920),
    ]:
        categories = [
            "Capital",
            "Language",
            "Population",
            "Neutral",
        ]

        fig, axes = plt.subplots(
            len(categories), 1,
            figsize=(11, 11),
        )

        for ax, category in zip(axes, categories):
            rows = [
                r for r in records
                if r["city"] == city
                and r["neuron"] == neuron
                and r["category"] == category
            ]

            positions = [r["position"] for r in rows]

            for component in [
                "gate", "value", "activation"
            ]:
                ax.plot(
                    positions,
                    [r[component] for r in rows],
                    marker="o",
                    label=component,
                )

            ax.set_xticks(
                positions,
                [repr(r["token"]) for r in rows],
            )

            ax.axhline(
                0, color="gray", linestyle="--"
            )

            ax.set_title(rows[0]["prompt"])
            ax.legend()

        fig.suptitle(f"L26:{neuron}")
        plt.tight_layout()

        fig.savefig(
            output_dir / f"trajectory_{neuron}.png",
            dpi=200,
        )

        plt.close(fig)


if __name__ == "__main__":
    main()
