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
from mechinterp.sparsity import (
    TOP,
    RANDOM,
    BOTTOM,
    with_mlp_sparsity,
)


MODEL_ID = "HuggingFaceTB/SmolLM2-360M"

RETENTION_FRACTIONS = [
    0.75,
    0.50,
    0.25,
    0.125,
    0.10,
    0.078125,
    0.05,
    0.025,
    0.,
]

RANDOM_FRACTIONS = [0.25, 0.078125]

FACTS = [
    ("Paris is the capital of", " France"),
    ("Rome is the capital of", " Italy"),
    ("Berlin is the capital of", " Germany"),
    ("Madrid is the capital of", " Spain"),
    ("Lisbon is the capital of", " Portugal"),
    ("Tokyo is the capital of", " Japan"),
    ("Paris is a city where people speak", " French"),
    ("Rome is a city where people speak", " Italian"),
    ("Berlin is a city where people speak", " German"),
    ("Madrid is a city where people speak", " Spanish"),
    ("Lisbon is a city where people speak", " Portuguese"),
    ("Tokyo is a city where people speak", " Japanese"),
    ("The largest planet in the solar system is", " Jupiter"),
    ("The opposite of hot is", " cold"),
    ("The chemical symbol for gold is", " Au"),
]

OUTPUT_DIR = Path("results/experiment_17")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num-tokens", type=int, default=8192
    )

    parser.add_argument(
        "--seq-len", type=int, default=128
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

    raise ValueError("Insufficient evaluation tokens")


def make_conditions(num_neurons):
    conditions = [
        ("baseline", num_neurons, TOP, 0)
    ]

    for fraction in RETENTION_FRACTIONS:
        keep = round(fraction * num_neurons)

        conditions.append(
            ("top", keep, TOP, 0)
        )

    for fraction in RANDOM_FRACTIONS:
        keep = round(fraction * num_neurons)

        for trial in range(2):
            conditions.append(
                ("random", keep, RANDOM, trial)
            )

    excluded = round(
        0.078125 * num_neurons
    )

    conditions.append(
        (
            "bottom",
            num_neurons - excluded,
            BOTTOM,
            0,
        )
    )

    return conditions


@jax.jit
def compare_distributions(
    original_log_probs,
    sparse_log_probs,
    targets,
):
    original_probs = jnp.exp(
        original_log_probs
    )

    target_log_probs = jnp.take_along_axis(
        sparse_log_probs,
        targets[:, None],
        axis=-1,
    ).squeeze(-1)

    nll = -target_log_probs

    kl = jnp.sum(
        original_probs
        * (original_log_probs - sparse_log_probs),
        axis=-1,
    )

    original_predictions = jnp.argmax(
        original_log_probs, axis=-1
    )

    sparse_predictions = jnp.argmax(
        sparse_log_probs, axis=-1
    )

    agreement = (
        original_predictions == sparse_predictions
    )

    accuracy = sparse_predictions == targets

    return jnp.stack([
        jnp.mean(nll),
        jnp.mean(kl),
        jnp.mean(agreement.astype(jnp.float32)),
        jnp.mean(accuracy.astype(jnp.float32)),
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

    OUTPUT_DIR.mkdir(
        parents=True, exist_ok=True
    )

    loaded = load_smollm2(MODEL_ID)
    tokenizer = loaded.tokenizer

    gated = instrument_mlp_neurons(
        loaded.model,
        expected_layers=loaded.num_layers,
    )

    num_neurons = gated.num_neurons
    conditions = make_conditions(num_neurons)

    @jax.jit
    def forward(tokens, keep, mode, key):
        model = with_mlp_sparsity(
            gated,
            keep,
            mode,
            key,
        )

        logits = model(
            pz.nx.wrap(tokens, "seq")
        ).unwrap(
            "seq", "vocabulary"
        )

        return jax.nn.log_softmax(
            logits.astype(jnp.float32),
            axis=-1,
        )

    def evaluate(tokens, keep, mode, key):
        return forward(
            tokens,
            jnp.int32(keep),
            jnp.int32(mode),
            key,
        )

    base_key = jax.random.key(args.seed)

    block_records = []

    for block_id, block in enumerate(
        corpus_blocks(
            tokenizer,
            args.seq_len,
            args.num_tokens,
        )
    ):
        tokens = block[:-1]
        targets = block[1:]

        block_key = jax.random.fold_in(
            base_key, block_id
        )

        original = evaluate(
            tokens,
            num_neurons,
            TOP,
            block_key,
        )

        for method, keep, mode, trial in conditions:
            key = jax.random.fold_in(
                block_key, trial
            )

            sparse = (
                original
                if method == "baseline"
                else evaluate(
                    tokens, keep, mode, key
                )
            )

            metrics = compare_distributions(
                original,
                sparse,
                targets,
            )

            loss, kl, agreement, accuracy = map(
                float, metrics
            )

            block_records.append({
                "block": block_id,
                "num_tokens": args.seq_len,
                "method": method,
                "keep": keep,
                "trial": trial,
                "retained_fraction": (
                    keep / num_neurons
                ),
                "sparsity": (
                    1 - keep / num_neurons
                ),
                "loss": loss,
                "kl": kl,
                "agreement": agreement,
                "accuracy": accuracy,
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
                OUTPUT_DIR / "corpus_blocks.csv",
                block_records,
            )

    save_csv(
        OUTPUT_DIR / "corpus_blocks.csv",
        block_records,
    )

    summary = summarize_corpus(block_records)

    save_csv(
        OUTPUT_DIR / "corpus_summary.csv",
        summary,
    )

    plot_corpus(summary)

    facts = evaluate_facts(
        tokenizer,
        evaluate,
        conditions,
        num_neurons,
        args.seq_len,
        base_key,
    )

    save_csv(
        OUTPUT_DIR / "factual_predictions.csv",
        facts,
    )

    factual_summary = summarize_facts(facts)

    save_csv(
        OUTPUT_DIR / "factual_summary.csv",
        factual_summary,
    )


def summarize_corpus(records):
    groups = defaultdict(list)

    for record in records:
        key = (
            record["method"],
            record["keep"],
        )

        groups[key].append(record)

    baseline_rows = [
        r for r in records
        if r["method"] == "baseline"
    ]

    baseline_loss = sum(
        r["loss"] for r in baseline_rows
    ) / len(baseline_rows)

    summaries = []

    print("\nCORPUS RESULTS\n")

    print(
        f"{'Method':<10}"
        f"{'Keep':>8}"
        f"{'Sparsity':>12}"
        f"{'Loss':>12}"
        f"{'Delta loss':>14}"
        f"{'PPL':>12}"
        f"{'KL':>12}"
        f"{'Agreement':>12}"
    )

    for (method, keep), rows in groups.items():
        def mean(field):
            return sum(
                r[field] for r in rows
            ) / len(rows)

        loss = mean("loss")
        kl = mean("kl")
        agreement = mean("agreement")
        accuracy = mean("accuracy")

        sparsity = rows[0]["sparsity"]

        delta_loss = loss - baseline_loss

        relative_loss_increase = (
            delta_loss / baseline_loss
        )

        perplexity = math.exp(loss)

        summaries.append({
            "method": method,
            "keep": keep,
            "sparsity": sparsity,
            "loss": loss,
            "delta_loss": delta_loss,
            "relative_loss_increase": (
                relative_loss_increase
            ),
            "perplexity": perplexity,
            "kl": kl,
            "agreement": agreement,
            "accuracy": accuracy,
            "num_blocks": len({
                r["block"] for r in rows
            }),
            "num_trials": len({
                r["trial"] for r in rows
            }),
        })

        print(
            f"{method:<10}"
            f"{keep:>8d}"
            f"{sparsity:>12.2%}"
            f"{loss:>12.4f}"
            f"{relative_loss_increase:>14.2%}"
            f"{perplexity:>12.2f}"
            f"{kl:>12.4f}"
            f"{agreement:>12.2%}"
        )

    return summaries


def evaluate_facts(
    tokenizer,
    evaluate,
    conditions,
    num_neurons,
    seq_len,
    base_key,
):
    records = []

    for fact_id, (prompt, answer) in enumerate(FACTS):
        ids = tokenizer.encode(
            prompt,
            add_special_tokens=False,
        )

        completed = tokenizer.encode(
            prompt + answer,
            add_special_tokens=False,
        )

        if (
            completed[:len(ids)] != ids
            or len(completed) != len(ids) + 1
        ):
            print(
                f"Skipping non-single-token answer: "
                f"{prompt!r} -> {answer!r}"
            )

            continue

        if len(ids) > seq_len:
            raise ValueError("Factual prompt too long")

        answer_id = completed[-1]
        position = len(ids) - 1

        padded = ids + [
            tokenizer.eos_token_id
        ] * (seq_len - len(ids))

        tokens = jnp.asarray(
            padded, dtype=jnp.int32
        )

        fact_key = jax.random.fold_in(
            base_key, 1_000_000 + fact_id
        )

        original = evaluate(
            tokens,
            num_neurons,
            TOP,
            fact_key,
        )[position]

        original_id = int(jnp.argmax(original))

        print(f"\n{prompt}")

        for method, keep, mode, trial in conditions:
            key = jax.random.fold_in(
                fact_key, trial
            )

            log_probs = (
                original
                if method == "baseline"
                else evaluate(
                    tokens, keep, mode, key
                )[position]
            )

            prediction_id = int(
                jnp.argmax(log_probs)
            )

            probability = float(
                jnp.exp(log_probs[answer_id])
            )

            records.append({
                "prompt": prompt,
                "answer": answer,
                "method": method,
                "keep": keep,
                "trial": trial,
                "sparsity": (
                    1 - keep / num_neurons
                ),
                "answer_probability": probability,
                "answer_log_probability": float(
                    log_probs[answer_id]
                ),
                "prediction": tokenizer.decode([
                    prediction_id
                ]),
                "correct": int(
                    prediction_id == answer_id
                ),
                "baseline_correct": int(
                    original_id == answer_id
                ),
                "agreement": int(
                    prediction_id == original_id
                ),
            })

            if method in ("top", "baseline"):
                print(
                    f"keep={keep:4d} "
                    f"p(answer)={probability:8.3%} "
                    f"pred={records[-1]['prediction']!r}"
                )

    return records


def summarize_facts(records):
    groups = defaultdict(list)

    for record in records:
        key = (
            record["method"],
            record["keep"],
        )

        groups[key].append(record)

    summaries = []

    for (method, keep), rows in groups.items():
        baseline_correct = [
            r for r in rows
            if r["baseline_correct"]
        ]

        n = len(rows)

        summaries.append({
            "method": method,
            "keep": keep,
            "sparsity": rows[0]["sparsity"],
            "num_prompts": len({
                r["prompt"] for r in rows
            }),
            "mean_answer_probability": sum(
                r["answer_probability"]
                for r in rows
            ) / n,
            "mean_answer_log_probability": sum(
                r["answer_log_probability"]
                for r in rows
            ) / n,
            "accuracy": sum(
                r["correct"] for r in rows
            ) / n,
            "agreement": sum(
                r["agreement"] for r in rows
            ) / n,
            "baseline_correct_preservation": (
                sum(
                    r["correct"]
                    for r in baseline_correct
                ) / len(baseline_correct)
                if baseline_correct else None
            ),
        })

    return summaries


def plot_corpus(summary):
    metrics = [
        ("relative_loss_increase", "Loss increase (%)", 100),
        ("perplexity", "Perplexity", 1),
        ("kl", "KL divergence (nats)", 1),
        ("agreement", "Top-1 agreement (%)", 100),
    ]

    fig, axes = plt.subplots(
        2, 2, figsize=(12, 9)
    )

    for ax, (metric, label, factor) in zip(
        axes.ravel(), metrics
    ):
        for method in ["top", "random", "bottom"]:
            rows = [
                r for r in summary
                if r["method"] == method
            ]

            if method == "top":
                rows += [
                    r for r in summary
                    if r["method"] == "baseline"
                ]

            rows.sort(
                key=lambda r: r["sparsity"]
            )

            if not rows:
                continue

            ax.plot(
                [100 * r["sparsity"] for r in rows],
                [factor * r[metric] for r in rows],
                marker="o",
                label=method.title(),
            )

        ax.axvline(
            92.1875,
            linestyle="--",
            alpha=0.4,
            label="Goodfire sparsity level",
        )

        ax.set_xlabel("Sparsity (%)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
        ax.legend()

    plt.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "sparsification.png",
        dpi=200,
    )

    plt.close(fig)


if __name__ == "__main__":
    main()
