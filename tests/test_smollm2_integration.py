"""Check Llama conversion parity and optionally the actual SmolLM2 checkpoint."""
import os

import pytest


def check_conversion(hf_model, ids):
    import jax.numpy as jnp
    import torch
    from penzai import pz
    from mechinterp.models import convert_llama

    hf_model.eval()
    with torch.no_grad():
        expected = hf_model(torch.tensor([ids]), use_cache=False).logits[0]
    model = convert_llama(hf_model)
    actual = model(pz.nx.wrap(jnp.asarray(ids, dtype=jnp.int32), "seq"))
    actual = actual.unwrap("seq", "vocabulary")
    assert bool(jnp.allclose(actual, jnp.asarray(expected.numpy()), atol=2e-3, rtol=2e-3))
    return model


def test_tied_llama_conversion():
    pytest.importorskip("penzai")
    transformers = pytest.importorskip("transformers")
    config = transformers.LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=24,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        tie_word_embeddings=True, max_position_embeddings=32,
    )
    config._name_or_path = "local-test"
    config.is_llama_config = True
    config.rope_interleaved = False
    config._attn_implementation = "eager"
    model = transformers.LlamaForCausalLM(config)
    check_conversion(model, [1, 2, 3])
    assert model.config.tie_word_embeddings
    assert model.config._name_or_path == "local-test"


@pytest.mark.integration
def test_smollm2_end_to_end():
    if os.environ.get("MECHINTERP_RUN_DOWNLOAD_TESTS") != "1":
        pytest.skip("Set MECHINTERP_RUN_DOWNLOAD_TESTS=1 to download SmolLM2")

    import jax
    import jax.numpy as jnp
    import torch
    from transformers import AutoTokenizer, LlamaForCausalLM
    from mechinterp.attribution import margin_function, ablated_scales
    from mechinterp.interventions import instrument_mlp_neurons
    from mechinterp.models import DEFAULT_MODEL_ID, single_token_id

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_ID)
    hf_model = LlamaForCausalLM.from_pretrained(
        DEFAULT_MODEL_ID, torch_dtype=torch.float32, attn_implementation="eager"
    )
    ids = tokenizer.encode("The capital of France is", add_special_tokens=False)
    model = check_conversion(hf_model, ids)
    del hf_model
    gated = instrument_mlp_neurons(model, expected_layers=30)
    assert gated.ones().shape == (30, 1536)
    target = single_token_id(tokenizer, " Paris")
    foil = single_token_id(tokenizer, " Rome")
    objective = margin_function(gated, jnp.asarray(ids), target, foil)
    margin, grad = jax.value_and_grad(objective)(gated.ones())
    assert bool(jnp.isfinite(margin))
    assert grad.shape == (30, 1536)
    assert bool(jnp.all(jnp.isfinite(grad)))
    edited = objective(ablated_scales(gated.ones(), [(0, 0)]))
    assert bool(jnp.isfinite(edited))
