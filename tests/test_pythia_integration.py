"""Optional pretrained-model smoke test, requiring a Hugging Face download."""

import os
import pytest

pytestmark = pytest.mark.integration


def test_pythia_end_to_end():
    if os.environ.get("MECHINTERP_RUN_DOWNLOAD_TESTS") != "1":
        pytest.skip("Set MECHINTERP_RUN_DOWNLOAD_TESTS=1 to download Pythia")

    import jax
    import jax.numpy as jnp

    from mechinterp.attribution import margin_function
    from mechinterp.interventions import instrument_mlp_neurons
    from mechinterp.models import load_pythia, single_token_id

    loaded = load_pythia()
    gated = instrument_mlp_neurons(loaded.model, expected_layers=loaded.num_layers)
    ids = jnp.asarray(loaded.tokenizer.encode("The capital of France is"))
    target = single_token_id(loaded.tokenizer, " Paris")
    foil = single_token_id(loaded.tokenizer, " Rome")
    objective = margin_function(gated, ids, target, foil)
    margin, grad = jax.value_and_grad(objective)(gated.ones())
    assert bool(jnp.isfinite(margin))
    assert grad.shape == gated.ones().shape
    assert bool(jnp.all(jnp.isfinite(grad)))
