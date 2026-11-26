import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from datasets import load_dataset
from penzai import pz

from mechinterp.models import load_smollm2
from mechinterp.interventions import instrument_mlp_neurons
from mechinterp.quadrants import (
    QUADRANTS,
    BASELINE,
    DROP,
    ONLY,
    TOP_K,
    RESTORE_QUADRANT,
    RESTORE_RANDOM,
    RESTORE_LARGEST,
    with_quadrant_intervention,
)


MODEL_ID = "HuggingFaceTB/SmolLM2-360M"

METHODS = {
    BASELINE: "baseline",
    DROP: "drop",
    ONLY: "only",
    TOP_K: "top_k",
    RESTORE_QUADRANT: "restore_quadrant",
    RESTORE_RANDOM: "restore_random",
    RESTORE_LARGEST: "restore_largest",
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num-tokens", type=int, default=8192
    )

    parser.add_argument(
        "--seq-len", type=int, default=128
    )

    parser.add_argument(
        "--keep", type=int, default=200
    )

    parser.add_argument(
        "--seed", type=int, default=42
    )

    return parser.parse_args()


def save_csv(path, records):
    if not records:
        return

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(records[0])
        )

        writer.writeheader()
        writer.writerows(records)


def corpus_blocks(tokenizer, seq_len, num_tokens):
    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
        split="test",
        streaming=True,
    )

    buffer = []
    remaining = num_tokens

    for row in dataset:
        text = row["text"]

        if not text.strip():
            continue

        if not text.endswith("\n"):
            text += "\n"

        buffer.extend(
            tokenizer.encode(
                text,
                add_special_tokens=False,
            )
        )

        while len(buffer) >= seq_len + 1:
            block = jnp.asarray(
                buffer[:seq_len + 1],
                dtype=jnp.int32,
            )

            del buffer[:seq_len]

            yield block

            remaining -= seq_len

            if remaining == 0:
                return

    raise ValueError("Insufficient corpus tokens")


def make_conditions():
    conditions = [
        (BASELINE, 0),
        (TOP_K, 0),
    ]

    for mode in [
        DROP,
        ONLY,
        RESTORE_QUADRANT,
        RESTORE_RANDOM,
        RESTORE_LARGEST,
    ]:
        for quadrant in range(4):
            conditions.append((mode, quadrant))

    return conditions


@jax.jit
def compare_distributions(
    original,
    modified,
    targets,
):
    target_log_probs = jnp.take_along_axis(
        modified,
        targets[:, None],
        axis=-1,
    ).squeeze(-1)

    loss = -target_log_probs.mean()

    kl = jnp.sum(
        jnp.exp(original)
        * (original - modified),
        axis=-1,
    ).mean()

    original_predictions = jnp.argmax(
        original, axis=-1
    )

    modified_predictions = jnp.argmax(
        modified, axis=-1
    )

    agreement = jnp.mean(
        original_predictions == modified_predictions
    )

    accuracy = jnp.mean(
        modified_predictions == targets
    )

    return jnp.stack([
        loss, kl, agreement, accuracy
    ])


def main():
    args = parse_args()

    if (
        args.seq_len <= 0
        or args.num_tokens <= 0
        or args.num_tokens % args.seq_len != 0
    ):
        raise ValueError(
            "num_tokens must be a positive "
            "multiple of seq_len"
        )

    output_dir = Path(
        f"results/experiment_18/keep_{args.keep}"
    )

    output_dir.mkdir(
        parents=True, exist_ok=True
    )

    loaded = load_smollm2(MODEL_ID)
    tokenizer = loaded.tokenizer

    gated = instrument_mlp_neurons(
        loaded.model,
        expected_layers=loaded.num_layers,
    )

    if not 0 <= args.keep <= gated.num_neurons:
        raise ValueError("Invalid retention budget")

    conditions = make_conditions()

    @jax.jit
    def forward(tokens, mode, quadrant, key):
        model, saved = with_quadrant_intervention(
            gated,
            mode,
            quadrant,
            jnp.int32(args.keep),
            key,
        )

        logits = model(
            pz.nx.wrap(tokens, "seq")
        ).unwrap(
            "seq", "vocabulary"
        )

        statistics = jnp.stack([
            state.value
            for state in saved
        ])

        return (
            jax.nn.log_softmax(
                logits.astype(jnp.float32),
                axis=-1,
            ),
            statistics,
        )

    def evaluate(tokens, mode, quadrant, key):
        return forward(
            tokens,
            jnp.int32(mode),
            jnp.int32(quadrant),
            key,
        )

    base_key = jax.random.key(args.seed)

    baseline_stats = jnp.zeros(
        (gated.num_layers, 4, 5),
        dtype=jnp.float32,
    )

    records = []

    for block_id, block in enumerate(
        corpus_blocks(
            tokenizer,
            args.seq_len,
            args.num_tokens,
        )
    ):
        tokens = block[:-1]
        targets = block[1:]

        key = jax.random.fold_in(
            base_key, block_id
        )

        original, statistics = evaluate(
            tokens, BASELINE, 0, key
        )

        baseline_stats += statistics

        if block_id == 0:
            reference_logits = gated.model(
                pz.nx.wrap(tokens, "seq")
            ).unwrap(
                "seq", "vocabulary"
            )

            reference = jax.nn.log_softmax(
                reference_logits.astype(jnp.float32),
                axis=-1,
            )

            if not jnp.allclose(
                original,
                reference,
                atol=1e-4,
                rtol=1e-4,
            ):
                raise ValueError(
                    "Baseline instrumentation changed logits"
                )

        for mode, quadrant in conditions:
            if mode == BASELINE:
                modified = original
                stats = statistics
            else:
                modified, stats = evaluate(
                    tokens, mode, quadrant, key
                )

            loss, kl, agreement, accuracy = map(
                float,
                compare_distributions(
                    original,
                    modified,
                    targets,
                ),
            )

            totals = stats.sum(axis=(0, 1))

            total_count = float(totals[0])
            retained_count = float(totals[3])

            records.append({
                "block": block_id,
                "num_tokens": args.seq_len,
                "keep": args.keep,
                "seed": args.seed,
                "method": METHODS[mode],
                "quadrant": (
                    QUADRANTS[quadrant]
                    if mode not in (BASELINE, TOP_K)
                    else "all"
                ),
                "loss": loss,
                "kl": kl,
                "agreement": agreement,
                "accuracy": accuracy,
                "retained_fraction": (
                    retained_count / total_count
                ),
                "retained_abs_mass_fraction": (
                    float(totals[4])
                    / max(float(totals[1]), 1e-12)
                ),
            })

        processed = (
            (block_id + 1) * args.seq_len
        )

        print(
            f"Evaluated {processed}/"
            f"{args.num_tokens} tokens",
            flush=True,
        )

        if (block_id + 1) % 4 == 0:
            save_csv(
                output_dir / "corpus_blocks.csv",
                records,
            )

    save_csv(
        output_dir / "corpus_blocks.csv",
        records,
    )

    save_quadrant_statistics(
        baseline_stats,
        output_dir,
    )

    summaries = summarize(
        records,
        output_dir,
    )

    plot_results(
        summaries,
        baseline_stats,
        output_dir,
    )


def save_quadrant_statistics(stats, output_dir):
    stats = jax.device_get(stats)

    records = []

    for layer, layer_stats in enumerate(stats):
        totals = layer_stats.sum(axis=0)

        for quadrant, row in enumerate(layer_stats):
            count, l1, l2, _, _ = map(float, row)

            records.append({
                "layer": layer,
                "quadrant": QUADRANTS[quadrant],
                "count": count,
                "count_fraction": count / totals[0],
                "absolute_mass": l1,
                "absolute_mass_fraction": l1 / max(
                    totals[1], 1e-12
                ),
                "squared_mass": l2,
                "squared_mass_fraction": l2 / max(
                    totals[2], 1e-12
                ),
                "mean_abs_activation": (
                    l1 / max(count, 1)
                ),
                "rms_activation": math.sqrt(
                    l2 / max(count, 1)
                ),
            })

    save_csv(
        output_dir / "quadrant_statistics.csv",
        records,
    )

    totals = stats.sum(axis=0)
    grand_totals = totals.sum(axis=0)

    print("\nGLOBAL QUADRANT STATISTICS\n")

    print(
        f"{'Quadrant':<12}"
        f"{'Frequency':>14}"
        f"{'Abs mass':>14}"
        f"{'Squared mass':>16}"
        f"{'Mean |h|':>14}"
    )

    for quadrant, row in enumerate(totals):
        count, l1, l2, _, _ = map(float, row)

        print(
            f"{QUADRANTS[quadrant]:<12}"
            f"{count / grand_totals[0]:>14.2%}"
            f"{l1 / grand_totals[1]:>14.2%}"
            f"{l2 / grand_totals[2]:>16.2%}"
            f"{l1 / max(count, 1):>14.5f}"
        )


def summarize(records, output_dir):
    groups = defaultdict(list)

    for record in records:
        key = (
            record["method"],
            record["quadrant"],
        )

        groups[key].append(record)

    def mean(rows, field):
        return sum(
            r[field] for r in rows
        ) / len(rows)

    baseline_loss = mean(
        groups["baseline", "all"],
        "loss",
    )

    sparse_loss = mean(
        groups["top_k", "all"],
        "loss",
    )

    sparse_damage = sparse_loss - baseline_loss

    summaries = []

    print("\nINTERVENTION RESULTS\n")

    print(
        f"{'Method':<22}"
        f"{'Quadrant':<10}"
        f"{'Retained':>12}"
        f"{'Loss':>12}"
        f"{'Delta loss':>14}"
        f"{'Recovered':>14}"
        f"{'Agreement':>14}"
    )

    for (method, quadrant), rows in groups.items():
        loss = mean(rows, "loss")
        retained = mean(
            rows, "retained_fraction"
        )

        relative_increase = (
            loss - baseline_loss
        ) / baseline_loss

        recovery = None

        if (
            method.startswith("restore")
            and sparse_damage > 1e-8
        ):
            recovery = (
                sparse_loss - loss
            ) / sparse_damage

        summaries.append({
            "method": method,
            "quadrant": quadrant,
            "num_blocks": len(rows),
            "retained_fraction": retained,
            "retained_abs_mass_fraction": mean(
                rows,
                "retained_abs_mass_fraction",
            ),
            "loss": loss,
            "delta_loss": loss - baseline_loss,
            "relative_loss_increase": relative_increase,
            "perplexity": math.exp(loss),
            "kl": mean(rows, "kl"),
            "agreement": mean(rows, "agreement"),
            "accuracy": mean(rows, "accuracy"),
            "recovery_fraction": recovery,
        })

        recovered_text = (
            f"{recovery:.2%}"
            if recovery is not None
            else "-"
        )

        print(
            f"{method:<22}"
            f"{quadrant:<10}"
            f"{retained:>12.2%}"
            f"{loss:>12.4f}"
            f"{relative_increase:>14.2%}"
            f"{recovered_text:>14}"
            f"{mean(rows, 'agreement'):>14.2%}"
        )

    save_csv(
        output_dir / "summary.csv",
        summaries,
    )

    return summaries


def plot_results(summaries, stats, output_dir):
    lookup = {
        (r["method"], r["quadrant"]): r
        for r in summaries
    }

    totals = jax.device_get(
        stats.sum(axis=0)
    )

    x = jnp.arange(4)

    fig, ax = plt.subplots(figsize=(9, 5))

    metrics = [
        (0, "Frequency"),
        (1, "Absolute activation mass"),
        (2, "Squared activation mass"),
    ]

    for i, (column, label) in enumerate(metrics):
        values = totals[:, column]
        fractions = 100 * values / values.sum()

        ax.bar(
            x + (i - 1) * 0.25,
            fractions,
            width=0.25,
            label=label,
        )

    ax.set_xticks(x, QUADRANTS)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Global SwiGLU quadrant statistics")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()

    fig.savefig(
        output_dir / "quadrant_statistics.png",
        dpi=200,
    )

    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, method in enumerate(["drop", "only"]):
        values = [
            100 * lookup[
                method, q
            ]["relative_loss_increase"]
            for q in QUADRANTS
        ]

        ax.bar(
            x + (i - 0.5) * 0.35,
            values,
            width=0.35,
            label=method.title(),
        )

    ax.set_xticks(x, QUADRANTS)
    ax.set_ylabel("Relative loss increase (%)")
    ax.set_title("Selective quadrant ablation")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()

    fig.savefig(
        output_dir / "quadrant_ablation.png",
        dpi=200,
    )

    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))

    methods = [
        ("restore_quadrant", "Quadrant"),
        ("restore_random", "Matched random"),
        ("restore_largest", "Matched largest"),
    ]

    for i, (method, label) in enumerate(methods):
        values = [
            100 * (
                lookup[
                    method, q
                ]["recovery_fraction"]
                or 0
            )
            for q in QUADRANTS
        ]

        ax.bar(
            x + (i - 1) * 0.25,
            values,
            width=0.25,
            label=label,
        )

    ax.axhline(
        100,
        linestyle="--",
        alpha=0.5,
        label="Full recovery",
    )

    ax.set_xticks(x, QUADRANTS)
    ax.set_ylabel("Loss recovered (%)")
    ax.set_title("Quadrant restoration")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()

    fig.savefig(
        output_dir / "quadrant_restoration.png",
        dpi=200,
    )

    plt.close(fig)


if __name__ == "__main__":
    main()
