import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from mechinterp.models import load_smollm2
from mechinterp.interventions import instrument_mlp_neurons
from mechinterp.activations import capture_swiglu_activations
from mechinterp.attribution import (
    next_token_logits_function,
    ablated_scales,
)


def main():
    loaded = load_smollm2("HuggingFaceTB/SmolLM2-360M")
    tokenizer = loaded.tokenizer

    gated = instrument_mlp_neurons(
        loaded.model,
        expected_layers=loaded.num_layers,
    )

    scales = gated.ones()

    layer = 26
    neurons = [829, 920]

    examples = [
        ("Paris is the capital of", " France"),
        ("The city of Paris is the capital of", " France"),
        ("The Eiffel Tower is located in", " Paris"),
        ("Lyon is a city in", " France"),
        ("The official language of France is", " French"),
        ("Rome is the capital of", " Italy"),
        ("The city of Rome is the capital of", " Italy"),
        ("The Colosseum is located in", " Rome"),
        ("Milan is a city in", " Italy"),
        ("The official language of Italy is", " Italian"),
        ("Berlin is the capital of", " Germany"),
        ("Madrid is the capital of", " Spain"),
        ("Munich is a city in", " Germany"),
        ("The largest planet is", " Jupiter"),
        ("The chemical symbol for gold is", " Au"),
        ("The opposite of cold is", " hot"),
    ]

    def tokenize(text):
        return jnp.asarray(
            tokenizer.encode(text, add_special_tokens=False),
            dtype=jnp.int32,
        )

    @jax.jit
    def logits_fn(scales, tokens):
        return next_token_logits_function(
            gated, tokens
        )(scales).astype(jnp.float32)

    def log_odds(logits, target_id):
        others = jnp.where(
            jnp.arange(logits.size) == target_id,
            -jnp.inf,
            logits,
        )

        return logits[target_id] - logsumexp(others)

    for prompt, target in examples:
        tokens = tokenize(prompt)
        target_ids = tokenize(target)

        captured = capture_swiglu_activations(
            gated, tokens, layer
        )

        g = captured.gate[-1]
        v = captured.value[-1]
        h = captured.activation[-1]

        assert jnp.allclose(
            h,
            jax.nn.silu(g) * v,
            rtol=1e-4,
            atol=1e-5,
        )

        baseline_logits = logits_fn(scales, tokens)

        target_id = (
            int(target_ids[0])
            if len(target_ids) == 1
            else None
        )

        print(f"\nPrompt: {prompt!r}")
        print(f"Target: {target!r}")

        if target_id is not None:
            probs = jax.nn.softmax(baseline_logits)
            print(f"P(target): {float(probs[target_id]):.4%}")

        print(
            f"{'Neuron':>8} "
            f"{'Gate':>10} "
            f"{'SiLU':>10} "
            f"{'Value':>10} "
            f"{'Activation':>12} "
            f"{'Effect':>10}"
        )

        for neuron in neurons:
            effect = float("nan")

            if target_id is not None:
                ablated = ablated_scales(
                    scales, [(layer, neuron)]
                )

                ablated_logits = logits_fn(ablated, tokens)

                effect = float(
                    log_odds(baseline_logits, target_id)
                    - log_odds(ablated_logits, target_id)
                )

            print(
                f"{neuron:8d} "
                f"{float(g[neuron]):+10.4f} "
                f"{float(jax.nn.silu(g[neuron])):+10.4f} "
                f"{float(v[neuron]):+10.4f} "
                f"{float(h[neuron]):+12.4f} "
                f"{effect:+10.4f}"
            )


if __name__ == "__main__":
    main()
