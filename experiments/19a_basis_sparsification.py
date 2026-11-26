import argparse
import csv

from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from datasets import load_dataset
from penzai import pz

from mechinterp.models import load_smollm2
from mechinterp.interventions import instrument_mlp_neurons
from mechinterp.basis import (
    make_layer_rotations,
    with_basis_ablation,
)


MODEL_ID = "HuggingFaceTB/SmolLM2-360M"

K_VALUES = [
    64, 128, 200, 320, 640, 1280, 1920
]

METRICS = (
    "loss",
    "kl",
    "agreement",
    "accuracy",
)


def corpus_blocks(tokenizer, num_tokens, seq_len):
    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
        split="test",
        streaming=True,
    )

    buffer = []
    num_blocks = num_tokens // seq_len
    width = seq_len + 1
    count = 0

    for example in dataset:
        buffer.extend(
            tokenizer.encode(
                example["text"],
                add_special_tokens=False,
            )
        )

        while len(buffer) >= width:
            block = jnp.asarray(
                buffer[:width],
                dtype=jnp.int32,
            )

            buffer = buffer[width:]

            yield block[:-1], block[1:]

            count += 1

            if count >= num_blocks:
                return


def make_conditions(num_neurons, num_rotations):
    conditions = [
        ("baseline", "retain", num_neurons, -1),
        ("native", "retain", 0, -1),
    ]

    for intervention in ("retain", "remove"):
        for k in K_VALUES:
            conditions.append(
                ("native", intervention, k, -1)
            )

            for seed in range(num_rotations):
                conditions.append(
                    ("rotated", intervention, k, seed)
                )

    return conditions


@jax.jit
def compute_metrics(clean, intervened, targets):
    loss = -jnp.mean(
        jnp.take_along_axis(
            intervened,
            targets[:, None],
            axis=-1,
        )
    )

    kl = jnp.mean(
        jnp.sum(
            jnp.exp(clean) * (clean - intervened),
            axis=-1,
        )
    )

    predictions = jnp.argmax(intervened, axis=-1)
    clean_predictions = jnp.argmax(clean, axis=-1)

    agreement = jnp.mean(
        predictions == clean_predictions
    )

    accuracy = jnp.mean(predictions == targets)

    return loss, kl, agreement, accuracy


def summarize(rows, num_neurons):
    grouped = defaultdict(list)

    for row in rows:
        key = (
            row["basis"],
            row["intervention"],
            row["k"],
            row["seed"],
        )

        grouped[key].append(row)

    seed_rows = []

    for key, group in grouped.items():
        basis, intervention, k, seed = key

        row = {
            "basis": basis,
            "intervention": intervention,
            "k": k,
            "seed": seed,
            "num_blocks": len(group),
        }

        for metric in METRICS:
            row[metric] = mean(
                r[metric] for r in group
            )

        seed_rows.append(row)

    grouped_seeds = defaultdict(list)

    for row in seed_rows:
        key = (
            row["basis"],
            row["intervention"],
            row["k"],
        )

        grouped_seeds[key].append(row)

    baseline_loss = next(
        r["loss"]
        for r in seed_rows
        if r["basis"] == "baseline"
    )

    ablated_kl = next(
        r["kl"]
        for r in seed_rows
        if r["basis"] == "native"
        and r["intervention"] == "retain"
        and r["k"] == 0
    )

    summary = []

    for key, group in grouped_seeds.items():
        basis, intervention, k = key

        row = {
            "basis": basis,
            "intervention": intervention,
            "k": k,
            "selected_fraction": k / num_neurons,
            "retained_fraction": (
                k / num_neurons
                if intervention == "retain"
                else 1 - k / num_neurons
            ),
            "num_rotations": (
                len(group) if basis == "rotated" else 0
            ),
        }

        for metric in METRICS:
            values = [r[metric] for r in group]

            row[metric] = mean(values)
            row[f"{metric}_std"] = (
                stdev(values) if len(values) > 1 else 0.0
            )

        row["relative_loss_increase"] = (
            100 * (row["loss"] / baseline_loss - 1)
        )

        row["relative_loss_increase_std"] = (
            100 * row["loss_std"] / baseline_loss
        )

        row["recovery"] = (
            1 - row["kl"] / ablated_kl
        )

        summary.append(row)

    return seed_rows, summary


def write_csv(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(rows[0])
        )

        writer.writeheader()
        writer.writerows(rows)


def plot_results(summary, output_dir):
    metrics = [
        ("relative_loss_increase", "Loss increase (%)"),
        ("kl", "KL divergence (nats)"),
    ]

    for metric, ylabel in metrics:
        fig, axes = plt.subplots(
            1, 2, figsize=(11, 4.5)
        )

        for ax, intervention in zip(
            axes, ("retain", "remove")
        ):
            for basis in ("native", "rotated"):
                rows = sorted(
                    [
                        r for r in summary
                        if r["basis"] == basis
                        and r["intervention"] == intervention
                        and r["k"] in K_VALUES
                    ],
                    key=lambda r: r["k"],
                )

                x = jnp.array([
                    100 * r["selected_fraction"]
                    for r in rows
                ])

                y = jnp.array([
                    r[metric] for r in rows
                ])

                std = jnp.array([
                    r[f"{metric}_std"]
                    for r in rows
                ])

                line, = ax.plot(
                    x,
                    y,
                    marker="o",
                    label=basis.capitalize(),
                )

                if basis == "rotated":
                    ax.fill_between(
                        x,
                        y - std,
                        y + std,
                        alpha=0.15,
                        color=line.get_color(),
                    )

            ax.set_xscale("log")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.2)
            ax.legend()

            if intervention == "retain":
                ax.set_title("Sufficiency")
                ax.set_xlabel("Strongest coordinates retained (%)")
            else:
                ax.set_title("Necessity")
                ax.set_xlabel("Strongest coordinates removed (%)")

        fig.tight_layout()

        fig.savefig(
            output_dir / f"{metric}.png",
            dpi=180,
            bbox_inches="tight",
        )

        plt.close(fig)


def main(args):
    if args.num_tokens < args.seq_len:
        raise ValueError("Not enough evaluation tokens")

    if args.rotations < 1:
        raise ValueError("At least one rotation required")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_smollm2(MODEL_ID)

    gated = instrument_mlp_neurons(
        loaded.model,
        expected_layers=loaded.num_layers,
    )

    num_neurons = gated.num_neurons

    rotation_sets = [
        make_layer_rotations(
            jax.random.fold_in(
                jax.random.key(args.seed), seed
            ),
            gated.num_layers,
            num_neurons,
        )
        for seed in range(args.rotations)
    ]

    @jax.jit
    def clean_forward(tokens):
        logits = gated.model(
            pz.nx.wrap(tokens, "seq")
        ).unwrap("seq", "vocabulary")

        return jax.nn.log_softmax(
            logits.astype(jnp.float32), axis=-1
        )

    @jax.jit
    def forward(tokens, rotations, rotated, k, retain):
        model = with_basis_ablation(
            gated,
            rotations,
            rotated,
            k,
            retain,
        )

        logits = model(
            pz.nx.wrap(tokens, "seq")
        ).unwrap("seq", "vocabulary")

        return jax.nn.log_softmax(
            logits.astype(jnp.float32), axis=-1
        )

    conditions = make_conditions(
        num_neurons, args.rotations
    )

    rows = []

    blocks = corpus_blocks(
        loaded.tokenizer,
        args.num_tokens,
        args.seq_len,
    )

    num_blocks = args.num_tokens // args.seq_len

    print(f"Layers: {gated.num_layers}")
    print(f"Neurons per layer: {num_neurons}")
    print(f"Rotations: {args.rotations}")
    print(f"Blocks: {num_blocks}")
    print(f"Conditions: {len(conditions)}")

    for block_index, (tokens, targets) in enumerate(blocks):
        clean = clean_forward(tokens)

        if block_index == 0:
            for seed, rotations in enumerate(rotation_sets):
                reconstructed = forward(
                    tokens,
                    rotations,
                    jnp.asarray(True),
                    jnp.asarray(num_neurons),
                    jnp.asarray(True),
                )

                _, kl, agreement, _ = compute_metrics(
                    clean, reconstructed, targets
                )

                max_error = jnp.max(
                    jnp.abs(clean - reconstructed)
                )

                print(
                    f"Rotation {seed}: "
                    f"round-trip KL={float(kl):.8f}, "
                    f"agreement={float(agreement):.6f}, "
                    f"max log-prob error={float(max_error):.6f}"
                )

                if float(kl) > 1e-5:
                    raise RuntimeError(
                        "Full rotation changes model behavior"
                    )

        for basis, intervention, k, seed in conditions:
            if basis == "baseline":
                intervened = clean
            else:
                rotations = rotation_sets[max(seed, 0)]

                intervened = forward(
                    tokens,
                    rotations,
                    jnp.asarray(basis == "rotated"),
                    jnp.asarray(k),
                    jnp.asarray(intervention == "retain"),
                )

            values = compute_metrics(
                clean, intervened, targets
            )

            row = {
                "block": block_index,
                "basis": basis,
                "intervention": intervention,
                "k": k,
                "seed": seed,
            }

            row.update({
                metric: float(value)
                for metric, value in zip(METRICS, values)
            })

            rows.append(row)

        print(
            f"Block {block_index + 1}/{num_blocks} complete",
            flush=True,
        )

        write_csv(
            output_dir / "corpus_blocks.csv", rows
        )

    seed_rows, summary = summarize(
        rows, num_neurons
    )

    write_csv(output_dir / "per_seed.csv", seed_rows)
    write_csv(output_dir / "summary.csv", summary)

    plot_results(summary, output_dir)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num-tokens", type=int, default=2048
    )

    parser.add_argument(
        "--seq-len", type=int, default=128
    )

    parser.add_argument(
        "--rotations", type=int, default=5
    )

    parser.add_argument(
        "--seed", type=int, default=42
    )

    parser.add_argument(
        "--output-dir",
        default="results/experiment_19a",
    )

    main(parser.parse_args())