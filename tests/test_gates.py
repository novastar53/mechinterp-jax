import jax
import jax.numpy as jnp
import pytest

pytest.importorskip("penzai")

from mechinterp.attribution import ablated_scales, margin_function
from mechinterp.interventions import instrument_mlp_neurons, scale_final_position
from penzai import pz
from penzai.models.transformer import variants


def test_only_last_position_is_scaled():
    values = jnp.array([[1., 2., 3.], [4., 5., 6.]])
    edited = scale_final_position(values, jnp.array([0., 2., 1.]))
    assert jnp.array_equal(edited, jnp.array([[1., 2., 3.], [0., 10., 6.]]))
    assert jnp.array_equal(values, jnp.array([[1., 2., 3.], [4., 5., 6.]]))


def tiny_model():
    config = variants.gpt_neox.GPTNeoXTransformerConfig(
        num_attention_heads=2,
        embedding_dim=8,
        projection_dim=4,
        mlp_hidden_dim=16,
        num_decoder_blocks=2,
        vocab_size=32,
        activation_fn="gelu_exact",
        rope_subset_size=2,
        rope_wavelength=10000.0,
        layernorm_epsilon=1e-5,
        parameter_dtype=jnp.float32,
        activation_dtype=jnp.float32,
    )
    model = variants.gpt_neox.build_gpt_neox_transformer(
        config, init_base_rng=jax.random.key(0)
    )
    return pz.freeze_params(model)


def tiny_llama_model():
    from penzai.models.transformer.variants import llamalike_common

    config = llamalike_common.LlamalikeTransformerConfig(
        num_kv_heads=1,
        query_head_multiplier=2,
        embedding_dim=8,
        projection_dim=4,
        mlp_hidden_dim=16,
        num_decoder_blocks=2,
        vocab_size=32,
        mlp_variant="swiglu",
        tie_embedder_and_logits=False,
    )
    return pz.freeze_params(llamalike_common.build_llamalike_transformer(
        config, init_base_rng=jax.random.key(0)
    ))


@pytest.mark.parametrize("model_factory", [tiny_model, tiny_llama_model])
def test_gates_are_identity_and_are_differentiable(model_factory):
    model = model_factory()
    ids = jnp.array([1, 2, 3], dtype=jnp.int32)
    original_logits = model(pz.nx.wrap(ids, "seq")).unwrap("seq", "vocabulary")
    gated = instrument_mlp_neurons(model, expected_layers=2)
    assert gated.ones().shape == (2, 16)

    objective = margin_function(gated, ids, target_id=7, foil_id=9)
    baseline = objective(gated.ones())
    assert baseline == pytest.approx(float(original_logits[-1, 7] - original_logits[-1, 9]), abs=1e-5)

    grad = jax.grad(objective)(gated.ones())
    assert grad.shape == (2, 16)
    assert bool(jnp.all(jnp.isfinite(grad)))

    # Check the local derivative against a finite difference.
    epsilon = 0.01
    scales = gated.ones()
    plus = scales.at[0, 0].add(epsilon)
    minus = scales.at[0, 0].add(-epsilon)
    numerical = (objective(plus) - objective(minus)) / (2 * epsilon)
    assert float(grad[0, 0]) == pytest.approx(float(numerical), abs=2e-2)

    # Ablation must be a non-mutating model transformation.
    ablated = ablated_scales(scales, [(0, 0), (1, 2)])
    assert float(ablated[0, 0]) == 0.0
    assert float(ablated[1, 2]) == 0.0
    assert float(scales[0, 0]) == 1.0



from mechinterp.interventions import (
    patch_final_position,
    with_activation_patch,
)
from mechinterp.activations import capture_swiglu_activations


def test_patch_final_position():
    activations = jnp.array([
        [1., 2., 3.],
        [4., 5., 6.],
    ])

    edited = patch_final_position(
        activations,
        jnp.array([0, 2]),
        jnp.array([-10., 20.]),
    )

    expected = jnp.array([
        [1., 2., 3.],
        [-10., 5., 20.],
    ])

    assert jnp.array_equal(edited, expected)
    assert float(activations[-1, 0]) == 4.


def test_activation_patch():
    gated = instrument_mlp_neurons(
        tiny_llama_model(),
        expected_layers=2,
    )

    ids = jnp.array([1, 2, 3], dtype=jnp.int32)
    tokens = pz.nx.wrap(ids, "seq")

    baseline = gated.model(tokens).unwrap(
        "seq", "vocabulary"
    )

    captured = capture_swiglu_activations(
        gated, ids, layer_index=1
    )

    original = captured.activation[-1, 3:4]

    patched = with_activation_patch(
        gated, 1, (3,), original
    )

    restored = patched(tokens).unwrap(
        "seq", "vocabulary"
    )

    assert jnp.allclose(
        baseline, restored,
        atol=1e-5, rtol=1e-5,
    )

    patched = with_activation_patch(
        gated, 1, (3,), jnp.array([0.])
    )

    patched_logits = patched(tokens).unwrap(
        "seq", "vocabulary"
    )

    ablated = ablated_scales(
        gated.ones(), [(1, 3)]
    )

    from mechinterp.interventions import with_scales

    ablated_logits = with_scales(
        gated, ablated
    )(tokens).unwrap("seq", "vocabulary")

    assert jnp.allclose(
        patched_logits, ablated_logits,
        atol=1e-5, rtol=1e-5,
    )

    assert jnp.allclose(
        patched_logits[:-1], baseline[:-1]
    )


from mechinterp.interventions import (
    with_activation_offsets,
    with_activation_patch,
)

from mechinterp.activations import (
    capture_final_mlp_activations,
    capture_swiglu_activations,

)


from mechinterp.interventions import (
    with_activation_offsets,
    with_activation_patch,
)

from mechinterp.activations import (
    capture_final_mlp_activations,
    capture_swiglu_activations,
)



def test_capture_activations_and_logits():
    gated = instrument_mlp_neurons(
        tiny_llama_model(),
        expected_layers=2,
    )

    ids = jnp.array([1, 2, 3], dtype=jnp.int32)

    activations, logits = capture_final_mlp_activations(
        gated.model,
        ids,
        (0, 1),
        return_logits=True,
    )

    expected_activations = capture_final_mlp_activations(
        gated.model,
        ids,
        (0, 1),
    )

    expected_logits = gated.model(
        pz.nx.wrap(ids, "seq")
    ).unwrap("seq", "vocabulary")[-1]

    assert jnp.allclose(
        activations, expected_activations
    )

    assert jnp.allclose(
        logits, expected_logits
    )



def test_full_mlp_retention():
    import jax

    from mechinterp.sparsity import (
        TOP,
        with_mlp_sparsity,
    )

    gated = instrument_mlp_neurons(
        tiny_llama_model(),
        expected_layers=2,
    )

    ids = pz.nx.wrap(
        jnp.array([1, 2, 3]), "seq"
    )

    baseline = gated.model(ids).unwrap(
        "seq", "vocabulary"
    )

    sparse = with_mlp_sparsity(
        gated,
        jnp.int32(gated.num_neurons),
        jnp.int32(TOP),
        jax.random.key(0),
    )

    actual = sparse(ids).unwrap(
        "seq", "vocabulary"
    )

    assert jnp.allclose(
        actual, baseline,
        atol=1e-5, rtol=1e-5,
    
    )



def test_quadrant_intervention_matches_sparsification():
    import jax

    from mechinterp.quadrants import (
        BASELINE,
        TOP_K,
        NN,
        with_quadrant_intervention,
    )

    from mechinterp.sparsity import (
        TOP,
        with_mlp_sparsity,
    )

    gated = instrument_mlp_neurons(
        tiny_llama_model(),
        expected_layers=2,
    )

    tokens = pz.nx.wrap(
        jnp.array([1, 2, 3]), "seq"
    )

    key = jax.random.key(0)

    baseline = gated.model(tokens).unwrap(
        "seq", "vocabulary"
    )

    model, saved = with_quadrant_intervention(
        gated,
        jnp.int32(BASELINE),
        jnp.int32(NN),
        jnp.int32(4),
        key,
    )

    actual = model(tokens).unwrap(
        "seq", "vocabulary"
    )

    assert jnp.allclose(
        actual, baseline,
        atol=1e-5, rtol=1e-5,
    )

    for state in saved:
        assert state.value.shape == (4, 5)

    model, _ = with_quadrant_intervention(
        gated,
        jnp.int32(TOP_K),
        jnp.int32(NN),
        jnp.int32(4),
        key,
    )

    reference = with_mlp_sparsity(
        gated,
        jnp.int32(4),
        jnp.int32(TOP),
        key,
    )

    actual = model(tokens).unwrap(
        "seq", "vocabulary"
    )

    expected = reference(tokens).unwrap(
        "seq", "vocabulary"
    )

    assert jnp.allclose(
        actual, expected,
        atol=1e-5, rtol=1e-5,
    )

