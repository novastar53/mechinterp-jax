# mechinterp

A small, experiment-driven mechanistic-interpretability library using **JAX**,
**Penzai**, and **uv**.

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/) are recommended:

```bash
uv sync
uv run pytest -q
uv run python experiments/01_neuron_ablation.py
```

`uv sync` creates `.venv` and a `uv.lock` file. Commit the lockfile after the
first successful resolution so subsequent installations are reproducible.

The first experiment downloads `EleutherAI/pythia-70m-deduped` (~70M parameter
GPT-NeoX model) and its tokenizer. PyTorch and Hugging Face Transformers are
installed **only for checkpoint loading**; forward passes, gradients,
interventions, and statistics are all computed in JAX.

The unit tests do not download pretrained weights. An optional integration
smoke test does:

```bash
MECHINTERP_RUN_DOWNLOAD_TESTS=1 uv run pytest -q tests/test_pythia_integration.py
```

## Experiment 1: do native neurons causally influence a prediction?

Use the prompt `The capital of France is`, comparing the next-token logits for
`" Paris"` and `" Rome"`. Both must be *single tokens* under the model's
tokenizer; the experiment checks this rather than silently assuming it.

For a given prompt, the model has a margin

\[
M = \ell_{\text{Paris}} - \ell_{\text{Rome}}.
\]

We place a multiplicative **gate** immediately after the GELU nonlinearity in
each GPT-NeoX MLP. For the final prompt token, `h_i` becomes `scale_i * h_i`.

- `scale_i = 1`: original neuron activity.
- `scale_i = 0`: ablated neuron.
- Differentiating the margin w.r.t. `scale_i` **at 1** yields
  `h_i * dM/dh_i` (gradient x activation) directly.

We rank native neurons by positive attribution, ablate each candidate and the
whole selected set, and compare measured changes against randomly selected
sets. Causal impact does **not** demonstrate that a neuron encodes Paris or
France; understanding semantic selectivity is a later experiment.

Optional arguments:

```bash
uv run python experiments/01_neuron_ablation.py --top-k 8 --num-controls 8 --seed 0
```

## Reusable code

```text
src/mechinterp/
  models.py          # pretrained checkpoint conversion and single-token checks
  interventions.py   # explicit, immutable model surgery and MLP neuron gates
  attribution.py     # margin function, gradients, group ablation
  metrics.py         # output metric and candidate ranking
experiments/
  01_neuron_ablation.py
```

**Supported scope:** GPT-NeoX/Pythia-style, non-stacked MLPs with one GELU and
an output projection. The gate currently assumes an *unbatched* prompt and
changes only the final token's hidden-neuron values. This is intentional: later
experiments will broaden the library only when we need more functionality.

## Limitations of this starter

- Pythia-70M is small and may not strongly prefer Paris over Rome for this
  prompt. Report the baseline margin and do not treat a weak or negative
  baseline as a successful factual-answer experiment.
- Gradients are local approximations. Real ablations can differ significantly.
- Random ablations are elementary controls, not matched controls.
- This repository does not yet capture activation datasets or trace circuits.
- The model-loading path relies on a Hugging Face GPT-NeoX checkpoint and a
  compatible `transformers` 4.x version; it is not a generic model adapter.
