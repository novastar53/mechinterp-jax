import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import matplotlib.pyplot as plt

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

    reference_tokens = tokenize("Paris is the capital of")

    france_id = single_token_id(tokenizer, " France")
    italy_id = single_token_id(tokenizer, " Italy")

    def margin(scales):
        logits = logits_fn(scales, reference_tokens)
        return logits[france_id] - logits[italy_id]

    attribution = jax.grad(margin)(scales)

    ranked = top_positive_neurons(attribution, k)

    neurons = [
        (layer, neuron)
        for layer, neuron, _ in ranked
    ]

    if len(neurons) != k:
        raise ValueError("Not enough positive attribution scores")

    examples = [
        ("France", "Paris is the capital of", " France"),
        ("Paraphrase", "The city of Paris is the capital of", " France"),
        ("Eiffel", "The Eiffel Tower is located in", " Paris"),
        ("French", "The official language of France is", " French"),
        ("Germany", "Berlin is the capital of", " Germany"),
        ("Italy", "Rome is the capital of", " Italy"),
        ("Spain", "Madrid is the capital of", " Spain"),
        ("Jupiter", "The largest planet is", " Jupiter"),
        ("Gold", "The chemical symbol for gold is", " Au"),
        ("Antonym", "The opposite of cold is", " hot"),
    ]

    tasks = [
        (label, tokenize(prompt), single_token_id(tokenizer, target))
        for label, prompt, target in examples
    ]

    @jax.jit
    def evaluate(scales, tokens, target_id):
        logits = logits_fn(scales, tokens)

        other_logits = jnp.where(
            jnp.arange(logits.size) == target_id,
            -jnp.inf,
            logits,
        )

        return logits[target_id] - logsumexp(other_logits)

    baselines = jnp.asarray([
        evaluate(scales, tokens, target_id)
        for _, tokens, target_id in tasks
    ])

    effects = jnp.zeros((k, len(tasks)))

    for i, neuron in enumerate(neurons):
        ablated = ablated_scales(scales, [neuron])

        after = jnp.asarray([
            evaluate(ablated, tokens, target_id)
            for _, tokens, target_id in tasks
        ])

        effects = effects.at[i].set(baselines - after)

        print(f"Evaluated neuron {neuron}", flush=True)

    labels = [label for label, _, _ in tasks]
    neuron_labels = [
        f"L{layer:02d}:{neuron:04d}"
        for layer, neuron in neurons
    ]

    print("\nIndividual causal effects (log-odds reduction)\n")

    print(f"{'Neuron':>12}", end="")

    for label in labels:
        print(f"{label:>12}", end="")

    print()

    for label, row in zip(neuron_labels, effects):
        print(f"{label:>12}", end="")

        for value in row:
            print(f"{float(value):+12.3f}", end="")

        print()

    vmax = float(jnp.max(jnp.abs(effects)))
    vmax = max(vmax, 1e-8)

    fig, ax = plt.subplots(figsize=(14, 10))

    im = ax.imshow(
        effects,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
    )

    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(k), neuron_labels)

    ax.set_xlabel("Task")
    ax.set_ylabel("Ablated neuron")

    for i in range(k):
        for j in range(len(tasks)):
            value = float(effects[i, j])

            ax.text(
                j,
                i,
                f"{value:+.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if abs(value) > 0.6 * vmax else "black",
            )

    fig.colorbar(im, ax=ax, label="Log-odds reduction")

    plt.tight_layout()
    plt.savefig("individual_neuron_specificity.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
