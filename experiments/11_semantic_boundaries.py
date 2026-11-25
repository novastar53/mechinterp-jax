import csv
from pathlib import Path
from statistics import median

import jax
import jax.numpy as jnp

from mechinterp.models import load_smollm2
from mechinterp.interventions import instrument_mlp_neurons
from mechinterp.activations import capture_swiglu_activations


CITIES = {
    "French": [
        "Paris", "Lyon", "Marseille", "Bordeaux", "Nice"
    ],
    "Belgian": [
        "Liège", "Namur", "Charleroi", "Mons"
    ],
    "Canadian": [
        "Quebec", "Montreal", "Trois-Rivières", "Sherbrooke"
    ],
    "Italian": [
        "Rome", "Milan", "Naples", "Florence", "Venice"
    ],
    "Valencian": [
        "Favara", "Otos", "Gandia", "Xàtiva", "Valencia"
    ],
    "Spanish": [
        "Madrid", "Seville", "Granada", "Toledo"
    ],
    "Portuguese": [
        "Lisbon", "Porto", "Coimbra", "Braga"
    ],
    "Romanian": [
        "Bucharest", "Iași", "Brașov"
    ],
    "German": [
        "Berlin", "Munich", "Hamburg"
    ],
    "English": [
        "London", "Manchester", "Oxford"
    ],
    "Japanese": [
        "Tokyo", "Kyoto", "Osaka"
    ],
}


PEOPLE = {
    "French": [
        "Napoleon Bonaparte",
        "Victor Hugo",
        "Claude Monet",
        "Jean-Paul Sartre",
    ],
    "Belgian": [
        "Hergé",
        "Georges Simenon",
        "Jacques Brel",
    ],
    "Canadian": [
        "Jean Chrétien",
        "Céline Dion",
        "Pierre Trudeau",
    ],
    "Italian": [
        "Dante Alighieri",
        "Leonardo da Vinci",
        "Ignazio Danti",
        "Giuseppe Verdi",
    ],
    "Italian-American": [
        "Michael Imperioli",
        "Robert De Niro",
        "Al Pacino",
        "Francis Ford Coppola",
    ],
    "Spanish": [
        "Miguel de Cervantes",
        "Pablo Picasso",
        "Diego Velázquez",
    ],
    "Portuguese": [
        "Fernando Pessoa",
        "Luís de Camões",
        "Cristiano Ronaldo",
    ],
    "German": [
        "Johann Wolfgang von Goethe",
        "Ludwig van Beethoven",
        "Albert Einstein",
    ],
}


TEMPLATES = [
    "The article mentions {}",
    "We read about {}",
]

SUFFIX = " in detail"


def main():
    loaded = load_smollm2("HuggingFaceTB/SmolLM2-360M")
    tokenizer = loaded.tokenizer

    gated = instrument_mlp_neurons(
        loaded.model,
        expected_layers=loaded.num_layers,
    )

    layer = 26
    neurons = jnp.array([829, 920])
    seq_len = 32

    @jax.jit
    def capture(tokens):
        captured = capture_swiglu_activations(
            gated, tokens, layer
        )

        return jnp.stack(
            [
                captured.gate[:, neurons],
                captured.value[:, neurons],
                captured.activation[:, neurons],
            ],
            axis=-1,
        )

    def tokenize(text):
        return tokenizer.encode(
            text, add_special_tokens=False
        )

    def evaluate(entity, template):
        prefix = template.format(entity)

        prefix_ids = tokenize(prefix)
        ids = tokenize(prefix + SUFFIX)

        if ids[:len(prefix_ids)] != prefix_ids:
            raise ValueError("Tokenization boundary changed")

        if len(ids) > seq_len:
            raise ValueError("Sequence exceeds maximum length")

        positions = {
            "entity": len(prefix_ids) - 1,
            "suffix": len(ids) - 1,
        }

        padded = ids + [
            tokenizer.eos_token_id
        ] * (seq_len - len(ids))

        tokens = jnp.asarray(padded, dtype=jnp.int32)

        values = jax.device_get(capture(tokens))

        return {
            name: values[pos]
            for name, pos in positions.items()
        }

    records = []

    for kind, groups in [
        ("city", CITIES),
        ("person", PEOPLE),
    ]:
        print(f"\n{kind.upper()} ACTIVATIONS\n")

        print(
            f"{'Category':<18}"
            f"{'Entity':<32}"
            f"{'829 entity':>12}"
            f"{'829 suffix':>12}"
            f"{'920 entity':>12}"
            f"{'920 suffix':>12}"
        )

        for category, entities in groups.items():
            for entity in entities:
                results = []

                for template in TEMPLATES:
                    result = evaluate(entity, template)
                    results.append(result)

                    for position, values in result.items():
                        for j, neuron in enumerate([829, 920]):
                            g, v, h = map(float, values[j])

                            records.append({
                                "kind": kind,
                                "category": category,
                                "entity": entity,
                                "template": template,
                                "position": position,
                                "layer": layer,
                                "neuron": neuron,
                                "gate": g,
                                "value": v,
                                "activation": h,
                            })

                def mean_activation(position, neuron_idx):
                    return sum(
                        float(r[position][neuron_idx, 2])
                        for r in results
                    ) / len(results)

                print(
                    f"{category:<18}"
                    f"{entity:<32}"
                    f"{mean_activation('entity', 0):+12.4f}"
                    f"{mean_activation('suffix', 0):+12.4f}"
                    f"{mean_activation('entity', 1):+12.4f}"
                    f"{mean_activation('suffix', 1):+12.4f}",
                    flush=True,
                )

    print("\nCATEGORY MEDIANS AT SHARED SUFFIX\n")

    print(
        f"{'Kind':<10}"
        f"{'Category':<20}"
        f"{'829':>12}"
        f"{'920':>12}"
    )

    for kind, groups in [
        ("city", CITIES),
        ("person", PEOPLE),
    ]:
        for category in groups:
            medians = []

            for neuron in [829, 920]:
                values = [
                    r["activation"]
                    for r in records
                    if r["kind"] == kind
                    and r["category"] == category
                    and r["neuron"] == neuron
                    and r["position"] == "suffix"
                ]

                medians.append(median(values))

            print(
                f"{kind:<10}"
                f"{category:<20}"
                f"{medians[0]:+12.4f}"
                f"{medians[1]:+12.4f}"
            )

    output_dir = Path("results/experiment_11")
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / "semantic_boundaries.csv"

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(records[0])
        )
        writer.writeheader()
        writer.writerows(records)

    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
