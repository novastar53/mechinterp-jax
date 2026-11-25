import argparse
import heapq
import json
from pathlib import Path

import jax
import jax.numpy as jnp
from datasets import load_dataset

from mechinterp.models import load_smollm2
from mechinterp.interventions import instrument_mlp_neurons
from mechinterp.activations import capture_swiglu_activations


def iter_sequences(dataset, tokenizer, seq_len, overlap):
    stride = seq_len - overlap

    for row_idx, row in enumerate(dataset):
        text = row["text"]

        if not text.strip():
            continue

        ids = tokenizer.encode(
            text, add_special_tokens=False
        )

        if len(ids) < 16:
            continue

        for start in range(0, len(ids), stride):
            chunk = ids[start:start + seq_len]
            offset = overlap if start > 0 else 0

            if len(chunk) > offset:
                yield row_idx, start, chunk, offset

            if start + seq_len >= len(ids):
                break


def main(args):
    loaded = load_smollm2("HuggingFaceTB/SmolLM2-360M")
    tokenizer = loaded.tokenizer

    gated = instrument_mlp_neurons(
        loaded.model,
        expected_layers=loaded.num_layers,
    )

    layer = 26
    neurons = [829, 920]
    indices = jnp.asarray(neurons)

    def capture(tokens):
        activations = capture_swiglu_activations(
            gated, tokens, layer
        )

        return jnp.stack(
            [
                activations.gate[:, indices],
                activations.value[:, indices],
                activations.activation[:, indices],
            ],
            axis=-1,
        )

    if not args.no_jit:
        capture = jax.jit(capture)

    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
        split="train",
        streaming=True,
    ).shuffle(
        seed=args.seed,
        buffer_size=10_000,
    )

    heaps = {
        (neuron, pole): []
        for neuron in neurons
        for pole in ["positive", "negative"]
    }

    pad_id = tokenizer.eos_token_id
    scanned = 0
    progress = 0

    print(f"Model: {loaded.model_id}")
    print(f"Layer: {layer}")
    print(f"Neurons: {neurons}")
    print(f"Token budget: {args.num_tokens:,}")
    print("Scanning corpus...", flush=True)

    for row_idx, start, ids, offset in iter_sequences(
        dataset, tokenizer, args.seq_len, overlap=32
    ):
        padded = ids + [pad_id] * (args.seq_len - len(ids))
        tokens = jnp.asarray(padded, dtype=jnp.int32)

        values = jax.device_get(capture(tokens)).tolist()

        end = min(
            len(ids),
            offset + args.num_tokens - scanned,
        )

        for pos in range(offset, end):
            for j, neuron in enumerate(neurons):
                g, v, h = values[pos][j]

                for pole, score in [
                    ("positive", h),
                    ("negative", -h),
                ]:
                    heap = heaps[(neuron, pole)]

                    if len(heap) >= args.top_k:
                        if score <= heap[0][0]:
                            continue

                    record = {
                        "layer": layer,
                        "neuron": neuron,
                        "pole": pole,
                        "activation": h,
                        "gate": g,
                        "value": v,
                        "row": row_idx,
                        "position": start + pos,
                        "before": tokenizer.decode(
                            ids[max(0, pos - 48):pos]
                        ),
                        "token": tokenizer.decode(
                            [ids[pos]]
                        ),
                        "after": tokenizer.decode(
                            ids[pos + 1:pos + 17]
                        ),
                        "prefix_ids": ids[:pos + 1],
                    }

                    entry = (score, scanned, record)

                    if len(heap) < args.top_k:
                        heapq.heappush(heap, entry)
                    else:
                        heapq.heapreplace(heap, entry)

            scanned += 1

        if scanned // 5000 > progress:
            progress = scanned // 5000
            print(f"Scanned {scanned:,} tokens", flush=True)

        if scanned >= args.num_tokens:
            break

    output_dir = Path("results/experiment_10")
    output_dir.mkdir(parents=True, exist_ok=True)

    for (neuron, pole), heap in heaps.items():
        ranked = sorted(heap, reverse=True)

        path = output_dir / f"L{layer}_{neuron}_{pole}.jsonl"

        with path.open("w", encoding="utf-8") as f:
            for rank, (_, _, record) in enumerate(ranked, 1):
                record["rank"] = rank
                f.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

        print(f"\nNEURON {neuron}: {pole.upper()} EXTREMES")

        for _, _, record in ranked[:10]:
            context = (
                record["before"]
                + " <<<"
                + record["token"]
                + ">>> "
                + record["after"]
            )

            print(
                f"\nh={record['activation']:+.4f} "
                f"g={record['gate']:+.4f} "
                f"v={record['value']:+.4f}"
            )
            print(repr(context))

        print(f"\nSaved: {path}")

    print(f"\nTotal tokens scanned: {scanned:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num-tokens", type=int, default=100_000
    )
    parser.add_argument(
        "--seq-len", type=int, default=256
    )
    parser.add_argument(
        "--top-k", type=int, default=50
    )
    parser.add_argument(
        "--seed", type=int, default=42
    )
    parser.add_argument(
        "--no-jit", action="store_true"
    )

    args = parser.parse_args()

    if args.seq_len <= 32:
        parser.error("--seq-len must exceed 32")

    if args.num_tokens <= 0 or args.top_k <= 0:
        parser.error("Token budget and top-k must be positive")

    main(args)
