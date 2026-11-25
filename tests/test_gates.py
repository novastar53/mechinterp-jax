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


def test_gates_are_identity_and_are_differentiable():
    model = tiny_model()
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
