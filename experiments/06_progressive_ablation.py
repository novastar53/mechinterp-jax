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

    prompt = "Paris is the capital of"
    target_id = single_token_id(tokenizer, " France")
    foil_id = single_token_id(tokenizer, " Italy")

    tokens = jnp.asarray(
        tokenizer.encode(prompt, add_special_tokens=False),
        dtype=jnp.int32,
    )

    gated = instrument_mlp_neurons(
        loaded.model,
        expected_layers=loaded.num_layers,
    )

    scales = gated.ones()
    logits_fn = jax.jit(
        next_token_logits_function(gated, tokens)
    )

    def margin(scales):
        logits = logits_fn(scales)
        return logits[target_id] - logits[foil_id]

    baseline, attribution = jax.value_and_grad(margin)(scales)

    sizes = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]
    num_controls = 8

    ranked = top_positive_neurons(attribution, max(sizes))
    selected = [(layer, neuron) for layer, neuron, _ in ranked]

    if len(selected) < max(sizes):
        raise ValueError("Not enough positively attributed neurons")

    key = jax.random.key(42)
    controls = []

    for _ in range(num_controls):
        key, subkey = jax.random.split(key)

        layer_keys = jax.random.split(
            subkey, gated.num_layers
        )

        permutations = jax.vmap(
            lambda k: jax.random.permutation(
                k, gated.num_neurons
            )
        )(layer_keys).tolist()

        offsets = [0] * gated.num_layers
        control = []

        for layer, _ in selected:
            neuron = permutations[layer][offsets[layer]]
            control.append((layer, neuron))
            offsets[layer] += 1

        controls.append(control)

    def evaluate(neurons):
        logits = logits_fn(
            ablated_scales(scales, neurons)
        )

        probs = jax.nn.softmax(
            logits.astype(jnp.float32)
        )

        prediction = int(jnp.argmax(logits))

        return (
            float(probs[target_id]),
            float(logits[target_id] - logits[foil_id]),
            prediction,
        )

    print(f"Model: {loaded.model_id}")
    print(f"Prompt: {prompt!r}")
    print(f"Baseline margin: {float(baseline):.4f}\n")

    print(
        f"{'K':>5} "
        f"{'France %':>10} "
        f"{'Margin':>10} "
        f"{'Random %':>10} "
        f"{'Rand std':>10} "
        f"{'Rand margin':>12} "
        f"{'Prediction':>18}"
    )

    for k in sizes:
        prob, m, prediction = evaluate(selected[:k])

        if k == 0:
            random_results = [(prob, m, prediction)]
        else:
            random_results = [
                evaluate(control[:k])
                for control in controls
            ]

        random_probs = jnp.asarray(
            [r[0] for r in random_results]
        )
        random_margins = jnp.asarray(
            [r[1] for r in random_results]
        )

        predicted_token = tokenizer.decode([prediction])

        print(
            f"{k:5d} "
            f"{100 * prob:10.4f} "
            f"{m:+10.4f} "
            f"{100 * float(random_probs.mean()):10.4f} "
            f"{100 * float(random_probs.std()):10.4f} "
            f"{float(random_margins.mean()):+12.4f} "
            f"{predicted_token!r:>18}",
            flush=True,
        )


if __name__ == "__main__":
    main()
