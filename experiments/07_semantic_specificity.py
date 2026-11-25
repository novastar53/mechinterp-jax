import jax
import jax.numpy as jnp

from mechinterp.models import load_smollm2, single_token_id
from mechinterp.interventions import instrument_mlp_neurons
from mechinterp.attribution import (
    next_token_logits_function,
    ablated_scales,
)
from mechinterp.metrics import top_positive_neurons


def main():
    loaded = load_smollm2("HuggingFaceTB/SmolLM2-360M")
    tokenizer = loaded.tokenizer

    gated = instrument_mlp_neurons(
        loaded.model,
        expected_layers=loaded.num_layers,
    )

    scales = gated.ones()
    k = 16
    num_controls = 8

    def tokenize(text):
        return jnp.asarray(
            tokenizer.encode(text, add_special_tokens=False),
            dtype=jnp.int32,
        )

    @jax.jit
    def logits_fn(scales, tokens):
        return next_token_logits_function(
            gated, tokens
        )(scales)

    reference_tokens = tokenize("Paris is the capital of")

    france_id = single_token_id(tokenizer, " France")
    italy_id = single_token_id(tokenizer, " Italy")

    def margin(scales):
        logits = logits_fn(scales, reference_tokens)
        return logits[france_id] - logits[italy_id]

    attribution = jax.grad(margin)(scales)

    ranked = top_positive_neurons(attribution, k)

    selected = [
        (layer, neuron)
        for layer, neuron, _ in ranked
    ]

    if len(selected) != k:
        raise ValueError("Not enough positive attribution scores")

    targeted_scales = ablated_scales(scales, selected)

    key = jax.random.key(42)
    random_scales = []

    for _ in range(num_controls):
        key, subkey = jax.random.split(key)

        layer_keys = jax.random.split(
            subkey, gated.num_layers
        )

        permutations = jax.vmap(
            lambda key: jax.random.permutation(
                key, gated.num_neurons
            )
        )(layer_keys).tolist()

        offsets = [0] * gated.num_layers
        control = []

        for layer, _ in selected:
            neuron = permutations[layer][offsets[layer]]
            control.append((layer, neuron))
            offsets[layer] += 1

        random_scales.append(
            ablated_scales(scales, control)
        )

    examples = [
        ("Paris is the capital of", " France"),
        ("The city of Paris is the capital of", " France"),
        ("The Eiffel Tower is located in", " Paris"),
        ("The official language of France is", " French"),
        ("Berlin is the capital of", " Germany"),
        ("Rome is the capital of", " Italy"),
        ("Madrid is the capital of", " Spain"),
        ("The largest planet is", " Jupiter"),
        ("The chemical symbol for gold is", " Au"),
        ("The opposite of cold is", " hot"),
    ]

    def evaluate(scales, tokens, target_id):
        logits = logits_fn(scales, tokens)
        probs = jax.nn.softmax(logits.astype(jnp.float32))

        probability = float(probs[target_id])
        rank = 1 + int(
            jnp.sum(logits > logits[target_id])
        )

        prediction = tokenizer.decode(
            [int(jnp.argmax(logits))]
        )

        return probability, rank, prediction

    print(f"Model: {loaded.model_id}")
    print(f"Ablated neurons: {k}\n")

    for prompt, target in examples:
        target_ids = tokenize(target)

        if len(target_ids) != 1:
            print(f"Skipping {target!r}: not a single token")
            continue

        target_id = int(target_ids[0])
        tokens = tokenize(prompt)

        p_before, rank_before, pred_before = evaluate(
            scales, tokens, target_id
        )

        p_after, rank_after, pred_after = evaluate(
            targeted_scales, tokens, target_id
        )

        random_probs = jnp.asarray([
            evaluate(s, tokens, target_id)[0]
            for s in random_scales
        ])

        print(f"Prompt: {prompt!r}")
        print(f"Target: {target!r}")
        print(
            f"Probability: {p_before:.4%} -> {p_after:.4%}"
        )
        print(
            f"Delta: {100 * (p_after - p_before):+.4f} pp"
        )
        print(
            f"Rank: {rank_before} -> {rank_after}"
        )
        print(
            f"Prediction: {pred_before!r} -> {pred_after!r}"
        )
        print(
            f"Random control: "
            f"{float(random_probs.mean()):.4%} "
            f"+/- {100 * float(random_probs.std()):.4f} pp"
        )
        print()


if __name__ == "__main__":
    main()
