import csv
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from penzai import pz
from jax.scipy.special import logsumexp

from mechinterp.models import load_smollm2, single_token_id
from mechinterp.interventions import (
    instrument_mlp_neurons,
    with_scales,
    with_activation_patch,
)
from mechinterp.activations import (
    capture_final_mlp_activations,
)


TARGET_LAYER = 26
TARGET_NEURONS = (829, 920)

TARGET_CITIES = {
    829: "Paris",
    920: "Rome",
}

ANSWERS = {
    "Paris": {
        "capital": " France",
        "language": " French",
    },
    "Rome": {
        "capital": " Italy",
        "language": " Italian",
    },
}

TASKS = ("capital", "language", "population")

GROUP_SIZE = 32
NUM_RANDOM_CONTROLS = 3
MATCH_POOL_SIZE = 64
SEED = 42

INPUT_DIR = Path("results/experiment_16")
OUTPUT_DIR = Path("results/experiment_16b")


def save_csv(path, records):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(records[0])
        )
        writer.writeheader()
        writer.writerows(records)


def load_fingerprints():
    with (INPUT_DIR / "prompts.csv").open() as f:
        prompts = list(csv.DictReader(f))

    for record in prompts:
        record["index"] = int(record["index"])
        record["template"] = int(record["template"])

    prompts.sort(key=lambda r: r["index"])

    with np.load(
        INPUT_DIR / "interaction_fingerprints.npz"
    ) as data:
        contributions = data["contributions"]

        if tuple(data["target_neurons"]) != TARGET_NEURONS:
            raise ValueError("Unexpected target neurons")

        if tuple(data["upstream_layers"]) != tuple(
            range(TARGET_LAYER)
        ):
            raise ValueError("Unexpected upstream layers")

    n = len(prompts)

    if contributions.shape[0] != n:
        raise ValueError("Prompt/fingerprint mismatch")

    if [r["index"] for r in prompts] != list(range(n)):
        raise ValueError("Unexpected prompt indices")

    x = np.abs(
        contributions.reshape(n, 2, 2, -1)
    )

    x /= np.maximum(
        x.sum(axis=-1, keepdims=True),
        1e-12,
    )

    # Average normalized gate and value contributions.
    return prompts, x.mean(axis=2)


def select_drivers(
    prompts,
    scores,
    target_idx,
    task,
    held_out_template,
):
    def task_mean(t):
        indices = [
            i for i, r in enumerate(prompts)
            if r["task"] == t
            and r["template"] != held_out_template
        ]

        if not indices:
            raise ValueError("Empty training set")

        return scores[indices, target_idx].mean(axis=0)

    positive = task_mean(task)

    negative = np.mean([
        task_mean(t)
        for t in TASKS
        if t != task
    ], axis=0)

    contrast = positive - negative

    ranked = np.argsort(
        -contrast, kind="stable"
    )[:GROUP_SIZE]

    if np.any(contrast[ranked] <= 0):
        raise ValueError("Insufficient positive contrasts")

    return ranked, contrast[ranked]


def matched_random_group(
    selected,
    upstream,
    rng,
):
    num_neurons = upstream.shape[1]

    magnitudes = np.log1p(np.abs(upstream))
    excluded = set(map(int, selected))
    sampled = []

    for idx in selected:
        layer, neuron = divmod(
            int(idx), num_neurons
        )

        candidates = np.array([
            j for j in range(num_neurons)
            if layer * num_neurons + j not in excluded
        ])

        if not len(candidates):
            raise ValueError("No matched candidates")

        distances = np.abs(
            magnitudes[layer, candidates]
            - magnitudes[layer, neuron]
        )

        nearest = candidates[
            np.argsort(distances)[:MATCH_POOL_SIZE]
        ]

        chosen = int(rng.choice(nearest))

        flat_idx = layer * num_neurons + chosen

        excluded.add(flat_idx)
        sampled.append(flat_idx)

    return np.asarray(sampled)


def make_scales(gated, group):
    indices = jnp.asarray(group)

    layers = indices // gated.num_neurons
    neurons = indices % gated.num_neurons

    return gated.ones().at[
        layers, neurons
    ].set(0.)


def log_odds(logits, mask):
    positive = logsumexp(
        jnp.where(mask, logits, -jnp.inf)
    )

    negative = logsumexp(
        jnp.where(~mask, logits, -jnp.inf)
    )

    return float(positive - negative)


def probability(log_odds_value):
    return float(
        jax.nn.sigmoid(log_odds_value)
    )


def numeric_token_mask(tokenizer, vocab_size):
    mask = np.zeros(vocab_size, dtype=bool)

    special = set(tokenizer.all_special_ids)

    for i in range(min(vocab_size, len(tokenizer))):
        if i in special:
            continue

        text = tokenizer.decode([i]).lstrip()

        mask[i] = bool(
            text and text[0] in "0123456789"
        )

    return jnp.asarray(mask)


def main():
    OUTPUT_DIR.mkdir(
        parents=True, exist_ok=True
    )

    prompts, scores = load_fingerprints()

    loaded = load_smollm2(
        "HuggingFaceTB/SmolLM2-360M"
    )

    tokenizer = loaded.tokenizer

    gated = instrument_mlp_neurons(
        loaded.model,
        expected_layers=loaded.num_layers,
    )

    if scores.shape[-1] != (
        TARGET_LAYER * gated.num_neurons
    ):
        raise ValueError("Fingerprint/model mismatch")

    target_indices = jnp.asarray(TARGET_NEURONS)
    layers = tuple(range(TARGET_LAYER + 1))

    rng = np.random.default_rng(SEED)

    @jax.jit
    def measure(tokens, scales):
        model = with_scales(gated, scales)

        h, logits = capture_final_mlp_activations(
            model,
            tokens,
            layers,
            return_logits=True,
        )

        return (
            h.astype(jnp.float32),
            logits.astype(jnp.float32),
        )

    @jax.jit
    def restored_logits(tokens, scales, values):
        scaled = replace(
            gated,
            model=with_scales(gated, scales),
        )

        model = with_activation_patch(
            scaled,
            TARGET_LAYER,
            TARGET_NEURONS,
            values,
        )

        named_tokens = pz.nx.wrap(tokens, "seq")

        return model(named_tokens).unwrap(
            "seq", "vocabulary"
        )[-1].astype(jnp.float32)

    groups = {}
    selection_records = []

    templates = sorted({
        r["template"] for r in prompts
    })

    for target_idx, target in enumerate(TARGET_NEURONS):
        for fold in templates:
            for mode in TASKS:
                group, contrast = select_drivers(
                    prompts,
                    scores,
                    target_idx,
                    mode,
                    fold,
                )

                groups[target, fold, mode] = group

                for rank, (idx, score) in enumerate(
                    zip(group, contrast), 1
                ):
                    layer, neuron = divmod(
                        int(idx), gated.num_neurons
                    )

                    selection_records.append({
                        "target_neuron": target,
                        "held_out_template": fold,
                        "mode": mode,
                        "rank": rank,
                        "driver_layer": layer,
                        "driver_neuron": neuron,
                        "contrast": float(score),
                    })

    save_csv(
        OUTPUT_DIR / "selected_drivers.csv",
        selection_records,
    )

    jobs = []

    for target_idx, target in enumerate(TARGET_NEURONS):
        for record in prompts:
            if record["city"] != TARGET_CITIES[target]:
                continue

            ids = tokenizer.encode(
                record["prompt"],
                add_special_tokens=False,
            )

            jobs.append((
                target_idx,
                target,
                record,
                ids,
            ))

    jobs.sort(key=lambda job: len(job[3]))

    records = []
    masks = {}
    last_length = None

    for target_idx, target, record, ids in jobs:
        if (
            last_length is not None
            and len(ids) != last_length
        ):
            jax.clear_caches()

        last_length = len(ids)

        tokens = jnp.asarray(
            ids, dtype=jnp.int32
        )

        city = record["city"]
        task = record["task"]
        fold = record["template"]

        baseline_h, baseline_logits = measure(
            tokens, gated.ones()
        )

        upstream = np.asarray(baseline_h[:-1])

        baseline_values = baseline_h[
            -1, target_indices
        ]

        original_activation = float(
            baseline_values[target_idx]
        )

        mask_key = (city, task)

        if mask_key not in masks:
            vocab_size = baseline_logits.shape[0]

            if task == "population":
                if "numeric" not in masks:
                    masks["numeric"] = numeric_token_mask(
                        tokenizer, vocab_size
                    )

                masks[mask_key] = masks["numeric"]

            else:
                answer = ANSWERS[city][task]

                answer_id = single_token_id(
                    tokenizer, answer
                )

                masks[mask_key] = (
                    jnp.arange(vocab_size) == answer_id
                )

        mask = masks[mask_key]

        baseline_score = log_odds(
            baseline_logits, mask
        )

        baseline_prediction = tokenizer.decode([
            int(jnp.argmax(baseline_logits))
        ])

        self_patch = restored_logits(
            tokens,
            gated.ones(),
            baseline_values,
        )

        if not jnp.allclose(
            self_patch,
            baseline_logits,
            atol=1e-4,
            rtol=1e-4,
        ):
            raise ValueError("Self-patching changed logits")

        print(
            f"\nL26:{target} | {record['prompt']}"
        )

        print(
            f"Baseline activation: "
            f"{original_activation:+.4f}"
        )

        print(
            f"Baseline probability: "
            f"{probability(baseline_score):.2%}"
        )

        print(
            f"Baseline prediction: "
            f"{baseline_prediction!r}",
            flush=True,
        )

        for mode in TASKS:
            selected = groups[target, fold, mode]

            conditions = [
                ("targeted", 0, selected)
            ]

            for trial in range(NUM_RANDOM_CONTROLS):
                random_group = matched_random_group(
                    selected,
                    upstream,
                    rng,
                )

                conditions.append((
                    "random",
                    trial,
                    random_group,
                ))

            for kind, trial, group in conditions:
                scales = make_scales(gated, group)

                ablated_h, ablated_logits = measure(
                    tokens, scales
                )

                ablated_values = ablated_h[
                    -1, target_indices
                ]

                replacement_values = ablated_values.at[
                    target_idx
                ].set(original_activation)

                recovered_logits = restored_logits(
                    tokens,
                    scales,
                    replacement_values,
                )

                ablated_score = log_odds(
                    ablated_logits, mask
                )

                restored_score = log_odds(
                    recovered_logits, mask
                )

                damage = (
                    baseline_score - ablated_score
                )

                recovery = (
                    restored_score - ablated_score
                )

                group_activations = upstream.ravel()[
                    group
                ]

                records.append({
                    "target_neuron": target,
                    "city": city,
                    "task": task,
                    "template": fold,
                    "prompt": record["prompt"],
                    "driver_mode": mode,
                    "group_type": kind,
                    "trial": trial,
                    "group_size": len(group),
                    "group_indices": " ".join(
                        map(str, group)
                    ),
                    "mean_abs_group_activation": float(
                        np.mean(np.abs(group_activations))
                    ),
                    "baseline_activation": (
                        original_activation
                    ),
                    "ablated_activation": float(
                        ablated_values[target_idx]
                    ),
                    "baseline_probability": probability(
                        baseline_score
                    ),
                    "ablated_probability": probability(
                        ablated_score
                    ),
                    "restored_probability": probability(
                        restored_score
                    ),
                    "baseline_log_odds": baseline_score,
                    "ablated_log_odds": ablated_score,
                    "restored_log_odds": restored_score,
                    "damage": damage,
                    "recovery": recovery,
                    "baseline_prediction": baseline_prediction,
                    "ablated_prediction": tokenizer.decode([
                        int(jnp.argmax(ablated_logits))
                    ]),
                    "restored_prediction": tokenizer.decode([
                        int(jnp.argmax(recovered_logits))
                    ]),
                })

                if kind == "targeted":
                    print(
                        f"{mode:<12}"
                        f"h={float(ablated_values[target_idx]):+8.3f} "
                        f"damage={damage:+8.3f} "
                        f"recovery={recovery:+8.3f}",
                        flush=True,
                    )

    save_csv(
        OUTPUT_DIR / "interventions.csv",
        records,
    )

    summarize(records)


def summarize(records):
    groups = defaultdict(list)

    for r in records:
        key = (
            r["target_neuron"],
            r["task"],
            r["driver_mode"],
            r["group_type"],
        )

        groups[key].append(r)

    summaries = []

    for target in TARGET_NEURONS:
        print(f"\nL26:{target} SUMMARY\n")

        print(
            f"{'Task':<12}"
            f"{'Drivers':<12}"
            f"{'Damage':>10}"
            f"{'Random':>10}"
            f"{'Excess':>10}"
            f"{'Recovery':>12}"
            f"{'Rnd recovery':>14}"
        )

        for task in TASKS:
            for mode in TASKS:
                targeted = groups[
                    target, task, mode, "targeted"
                ]

                random = groups[
                    target, task, mode, "random"
                ]

                damage = np.mean([
                    r["damage"] for r in targeted
                ])

                random_damage = np.mean([
                    r["damage"] for r in random
                ])

                recovery = np.mean([
                    r["recovery"] for r in targeted
                ])

                random_recovery = np.mean([
                    r["recovery"] for r in random
                ])

                excess = damage - random_damage

                summaries.append({
                    "target_neuron": target,
                    "task": task,
                    "driver_mode": mode,
                    "targeted_damage": float(damage),
                    "random_damage": float(random_damage),
                    "excess_damage": float(excess),
                    "targeted_recovery": float(recovery),
                    "random_recovery": float(random_recovery),
                    "recovery_fraction": (
                        float(recovery / damage)
                        if damage > 0.1
                        else None
                    ),
                })

                print(
                    f"{task:<12}"
                    f"{mode:<12}"
                    f"{damage:+10.3f}"
                    f"{random_damage:+10.3f}"
                    f"{excess:+10.3f}"
                    f"{recovery:+12.3f}"
                    f"{random_recovery:+14.3f}"
                )

    save_csv(
        OUTPUT_DIR / "summary.csv",
        summaries,
    )


if __name__ == "__main__":
    main()
