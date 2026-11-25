import jax.numpy as jnp

from mechinterp.models import load_smollm2
from mechinterp.interventions import instrument_mlp_neurons
from mechinterp.attribution import (
    next_token_logits_function,
    ablated_scales,
)


def generate(
    gated,
    tokenizer,
    prompt,
    neurons,
    mode="baseline",
    max_new_tokens=20,
):
    ids = tokenizer.encode(
        prompt, add_special_tokens=False
    )

    initial_length = len(ids)

    scales = gated.ones()
    ablated = ablated_scales(scales, neurons)

    for step in range(max_new_tokens):
        tokens = jnp.asarray(ids, dtype=jnp.int32)

        logits_fn = next_token_logits_function(
            gated, tokens
        )

        if mode == "every_token":
            active_scales = ablated
        elif mode == "first_token" and step == 0:
            active_scales = ablated
        else:
            active_scales = scales

        logits = logits_fn(active_scales)
        next_id = int(jnp.argmax(logits))

        ids.append(next_id)

        if next_id == tokenizer.eos_token_id:
            break

    return tokenizer.decode(
        ids[initial_length:],
        skip_special_tokens=True,
    )


def main():
    loaded = load_smollm2()

    gated = instrument_mlp_neurons(
        loaded.model,
        expected_layers=loaded.num_layers,
    )

    neurons = [
        (25, 1120),
        (29, 511),
        (22, 754),
        (29, 385),
        (23, 1043),
        (28, 51),
        (29, 1244),
        (27, 1392),
    ]

    prompt = "Paris is the capital of"

    for mode in [
        "baseline",
        "first_token",
        "every_token",
    ]:
        continuation = generate(
            gated,
            loaded.tokenizer,
            prompt,
            neurons,
            mode=mode,
        )

        print(f"\n{mode.upper()}")
        print(prompt + continuation)


if __name__ == "__main__":
    main()
