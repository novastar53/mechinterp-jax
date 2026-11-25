import jax
import jax.numpy as jnp

from mechinterp.models import load_smollm2, single_token_id
from mechinterp.interventions import instrument_mlp_neurons
from mechinterp.attribution import margin_function, ablated_scales


def main():
    loaded = load_smollm2()
    tokenizer = loaded.tokenizer

    gated = instrument_mlp_neurons(
        loaded.model,
        expected_layers=loaded.num_layers,
    )

    target_id = single_token_id(tokenizer, " Paris")
    foil_id = single_token_id(tokenizer, " Rome")

    neuron = (28, 1095)

    scales = gated.ones()
    ablated = ablated_scales(scales, [neuron])

    prompts = [
        "The capital of France is",
        "France's capital city is",
        "The Eiffel Tower is in",
        "The Louvre Museum is in",
        "The capital of Italy is",
        "The Colosseum is in",
        "The capital of Germany is",
        "The capital of Spain is",
        "The largest planet is",
        "The opposite of cold is",
    ]

    print(f"Neuron: {neuron}")
    print(f"{'Baseline':>10} {'Ablated':>10} {'Delta':>10}  Prompt")

    for prompt in prompts:
        tokens = jnp.asarray(
            tokenizer.encode(prompt, add_special_tokens=False),
            dtype=jnp.int32,
        )

        margin = jax.jit(
            margin_function(gated, tokens, target_id, foil_id)
        )

        baseline = float(margin(scales))
        after = float(margin(ablated))

        print(
            f"{baseline:+10.4f} "
            f"{after:+10.4f} "
            f"{after - baseline:+10.4f}  "
            f"{prompt}"
        )


if __name__ == "__main__":
    main()
