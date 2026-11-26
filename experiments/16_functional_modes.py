import csv
from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, cut_tree, dendrogram
from scipy.spatial.distance import squareform

from mechinterp.models import load_smollm2
from mechinterp.interventions import (
    instrument_mlp_neurons,
    with_activation_offsets,
    with_activation_patch,
)
from mechinterp.activations import (
    capture_final_mlp_activations,
    capture_swiglu_activations,
)


TARGET_LAYER = 26
TARGET_NEURONS = (829, 920)

CITIES = [
    "Paris",
    "Rome",
    "Berlin",
    "Madrid",
    "Lisbon",
    "London",
]

TEMPLATES = {
    "capital": [
        "{} is the capital of",
        "The country with {} as its capital is",
        "{} serves as the capital of",
    ],
    "language": [
        "{} is a city where people speak",
        "The main language spoken in {} is",
        "In {}, the primary language is",
    ],
    "population": [
        "{} has a population of",
        "The population of {} is approximately",
        "The number of people living in {} is approximately",
    ],
}


def normalize(x):
    norm = jnp.linalg.norm(
        x, axis=-1, keepdims=True
    )

    return x / jnp.maximum(norm, 1e-12)


def save_csv(path, records):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(records[0])
        )
        writer.writeheader()
        writer.writerows(records)


def main():
    loaded = load_smollm2(
        "HuggingFaceTB/SmolLM2-360M"
    )

    tokenizer = loaded.tokenizer

    gated = instrument_mlp_neurons(
        loaded.model,
        expected_layers=loaded.num_layers,
    )

    upstream_layers = tuple(range(TARGET_LAYER))

    downstream_layers = tuple(
        range(TARGET_LAYER + 1, gated.num_layers)
    )

    target_indices = jnp.asarray(TARGET_NEURONS)

    zero_offsets = jnp.zeros(
        (len(upstream_layers), gated.num_neurons),
        dtype=jnp.float32,
    )

    @jax.jit
    def baseline_activations(tokens):
        return capture_final_mlp_activations(
            gated.model,
            tokens,
            upstream_layers + (TARGET_LAYER,),
        ).astype(jnp.float32)

    @jax.jit
    def upstream_sensitivities(tokens):
        def projections(offsets):
            model = with_activation_offsets(
                gated,
                upstream_layers,
                offsets,
            )

            captured = capture_swiglu_activations(
                gated,
                tokens,
                TARGET_LAYER,
                model=model,
            )

            return jnp.stack([
                captured.gate[-1, target_indices],
                captured.value[-1, target_indices],
            ], axis=-1).astype(jnp.float32)

        outputs, pullback = jax.vjp(
            projections,
            zero_offsets,
        )

        basis = jnp.eye(
            outputs.size,
            dtype=outputs.dtype,
        ).reshape((-1,) + outputs.shape)

        gradients = jax.lax.map(
            lambda c: pullback(c)[0],
            basis,
        )

        return outputs, gradients.reshape(
            outputs.shape + zero_offsets.shape
        )

    @jax.jit
    def downstream_sensitivities(tokens, source_values):
        def readers(values):
            model = with_activation_patch(
                gated,
                TARGET_LAYER,
                TARGET_NEURONS,
                values,
            )

            return capture_final_mlp_activations(
                model,
                tokens,
                downstream_layers,
            ).astype(jnp.float32)

        directions = jnp.eye(
            len(TARGET_NEURONS),
            dtype=source_values.dtype,
        )

        def derivative(direction):
            return jax.jvp(
                readers,
                (source_values,),
                (direction,),
            )[1]

        return jax.lax.map(
            derivative,
            directions,
        )

    records = []

    for city in CITIES:
        for task, templates in TEMPLATES.items():
            for template_id, template in enumerate(templates):
                prompt = template.format(city)

                ids = tokenizer.encode(
                    prompt,
                    add_special_tokens=False,
                )

                records.append({
                    "city": city,
                    "task": task,
                    "template": template_id,
                    "prompt": prompt,
                    "ids": ids,
                })

    # Group by sequence length to limit recompilation.
    buckets = defaultdict(list)

    for i, record in enumerate(records):
        buckets[len(record["ids"])].append(i)

    gradients = [None] * len(records)
    contributions = [None] * len(records)
    readers = [None] * len(records)
    activations = [None] * len(records)
    projections = [None] * len(records)

    completed = 0

    for length, indices in sorted(buckets.items()):
        print(f"\nSequence length: {length}", flush=True)

        for i in indices:
            record = records[i]

            tokens = jnp.asarray(
                record["ids"],
                dtype=jnp.int32,
            )

            baseline = baseline_activations(tokens)

            upstream = baseline[:-1]
            source_values = baseline[-1, target_indices]

            outputs, grads = upstream_sensitivities(tokens)

            reconstructed = (
                jax.nn.silu(outputs[:, 0])
                * outputs[:, 1]
            )

            if not jnp.allclose(
                source_values,
                reconstructed,
                atol=1e-4,
                rtol=1e-4,
            ):
                raise ValueError(
                    "Reconstructed activation mismatch"
                )

            contrib = grads * upstream[None, None, :, :]

            downstream = downstream_sensitivities(
                tokens,
                source_values,
            )

            results = jax.device_get((
                grads,
                contrib,
                downstream,
                source_values,
                outputs,
            ))

            (
                gradients[i],
                contributions[i],
                readers[i],
                activations[i],
                projections[i],
            ) = results

            completed += 1

            print(
                f"[{completed}/{len(records)}] "
                f"{record['prompt']} "
                f"829={float(source_values[0]):+.3f} "
                f"920={float(source_values[1]):+.3f}",
                flush=True,
            )

        # Control compiled executable memory on the MacBook.
        jax.clear_caches()

    output_dir = Path("results/experiment_16")
    output_dir.mkdir(parents=True, exist_ok=True)

    gradients = np.stack(gradients)
    contributions = np.stack(contributions)
    readers = np.stack(readers)
    activations = np.stack(activations)
    projections = np.stack(projections)

    np.savez_compressed(
        output_dir / "interaction_fingerprints.npz",
        gradients=gradients,
        contributions=contributions,
        readers=readers,
        activations=activations,
        projections=projections,
        target_neurons=np.asarray(TARGET_NEURONS),
        upstream_layers=np.asarray(upstream_layers),
        downstream_layers=np.asarray(downstream_layers),
    )

    save_csv(
        output_dir / "prompts.csv",
        [
            {
                "index": i,
                "city": r["city"],
                "task": r["task"],
                "template": r["template"],
                "prompt": r["prompt"],
                "activation_829": float(activations[i, 0]),
                "activation_920": float(activations[i, 1]),
            }
            for i, r in enumerate(records)
        ],
    )

    analyze(
        records,
        contributions,
        readers,
        output_dir,
    )


def analyze(records, contributions, readers, output_dir):
    n = len(records)

    cluster_records = []
    comparison_records = []
    generalization_records = []

    for i, neuron in enumerate(TARGET_NEURONS):
        gate = normalize(
            jnp.asarray(contributions[:, i, 0]).reshape(n, -1)
        )

        value = normalize(
            jnp.asarray(contributions[:, i, 1]).reshape(n, -1)
        )

        downstream = normalize(
            jnp.asarray(readers[:, i]).reshape(n, -1)
        )

        similarities = {
            "gate": gate @ gate.T,
            "value": value @ value.T,
            "readers": downstream @ downstream.T,
        }

        similarities["combined"] = (
            similarities["gate"]
            + similarities["value"]
            + similarities["readers"]
        ) / 3

        for component, similarity in similarities.items():
            similarity = np.asarray(similarity)

            distance = np.clip(
                1 - similarity,
                0,
                2,
            )

            distance = (distance + distance.T) / 2
            np.fill_diagonal(distance, 0)

            tree = linkage(
                squareform(distance),
                method="average",
            )

            clusters = cut_tree(
                tree,
                n_clusters=3,
            ).ravel()

            print(f"\nL26:{neuron} — {component.upper()}")

            for cluster in sorted(set(clusters)):
                members = [
                    r["task"]
                    for j, r in enumerate(records)
                    if clusters[j] == cluster
                ]

                counts = {
                    task: members.count(task)
                    for task in TEMPLATES
                }

                print(f"Cluster {cluster}: {counts}")

            for j, record in enumerate(records):
                cluster_records.append({
                    "neuron": neuron,
                    "component": component,
                    "prompt_index": j,
                    "city": record["city"],
                    "task": record["task"],
                    "template": record["template"],
                    "cluster": int(clusters[j]),
                })

            same_task = []
            different_task = []

            for a in range(n):
                for b in range(a + 1, n):
                    ra = records[a]
                    rb = records[b]

                    if ra["city"] == rb["city"]:
                        continue

                    if ra["task"] == rb["task"]:
                        if ra["template"] == rb["template"]:
                            continue

                        same_task.append(similarity[a, b])
                    else:
                        different_task.append(similarity[a, b])

            within = float(np.mean(same_task))
            between = float(np.mean(different_task))

            comparison_records.append({
                "neuron": neuron,
                "component": component,
                "same_task_similarity": within,
                "different_task_similarity": between,
                "separation": within - between,
            })

            print(f"Same-task similarity: {within:+.4f}")
            print(f"Different-task similarity: {between:+.4f}")

            predictions = cross_generalization(
                records,
                similarity,
                neuron,
                component,
            )

            generalization_records.extend(predictions)

            accuracy = np.mean([
                r["correct"] for r in predictions
            ])

            print(f"Held-out accuracy: {accuracy:.2%}")

            if component == "combined":
                np.save(
                    output_dir / f"similarity_{neuron}.npy",
                    similarity,
                )

                plot_modes(
                    records,
                    similarity,
                    tree,
                    neuron,
                    output_dir,
                )

    save_csv(
        output_dir / "clusters.csv",
        cluster_records,
    )

    save_csv(
        output_dir / "mode_similarity.csv",
        comparison_records,
    )

    save_csv(
        output_dir / "generalization.csv",
        generalization_records,
    )


def cross_generalization(
    records,
    similarity,
    neuron,
    component,
):
    results = []

    for i, test in enumerate(records):
        scores = {}

        for task in TEMPLATES:
            train = [
                j for j, r in enumerate(records)
                if r["task"] == task
                and r["city"] != test["city"]
                and r["template"] != test["template"]
            ]

            scores[task] = float(
                np.mean(similarity[i, train])
            )

        predicted = max(scores, key=scores.get)

        results.append({
            "neuron": neuron,
            "component": component,
            "prompt_index": i,
            "city": test["city"],
            "task": test["task"],
            "template": test["template"],
            "predicted_task": predicted,
            "correct": int(predicted == test["task"]),
        })

    return results


def plot_modes(
    records,
    similarity,
    tree,
    neuron,
    output_dir,
):
    labels = [
        f"{r['city']} | {r['task']} | T{r['template']}"
        for r in records
    ]

    fig, ax = plt.subplots(figsize=(11, 15))

    dendrogram(
        tree,
        labels=labels,
        orientation="right",
        leaf_font_size=8,
        ax=ax,
    )

    ax.set_title(f"L26:{neuron} — Functional modes")
    ax.set_xlabel("Cosine distance")

    plt.tight_layout()

    fig.savefig(
        output_dir / f"dendrogram_{neuron}.png",
        dpi=200,
    )

    plt.close(fig)

    order = sorted(
        range(len(records)),
        key=lambda i: (
            records[i]["task"],
            records[i]["city"],
            records[i]["template"],
        ),
    )

    ordered = similarity[np.ix_(order, order)]
    ordered_labels = [labels[i] for i in order]

    fig, ax = plt.subplots(figsize=(14, 12))

    im = ax.imshow(
        ordered,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        aspect="equal",
    )

    ax.set_xticks(
        range(len(order)),
        ordered_labels,
        rotation=90,
        fontsize=6,
    )

    ax.set_yticks(
        range(len(order)),
        ordered_labels,
        fontsize=6,
    )

    group_size = len(CITIES) * 3

    for boundary in [group_size, 2 * group_size]:
        ax.axhline(boundary - 0.5, color="black")
        ax.axvline(boundary - 0.5, color="black")

    ax.set_title(f"L26:{neuron} — Fingerprint similarity")

    fig.colorbar(im, ax=ax, label="Cosine similarity")

    plt.tight_layout()

    fig.savefig(
        output_dir / f"heatmap_{neuron}.png",
        dpi=200,
    )

    plt.close(fig)


if __name__ == "__main__":
    main()
