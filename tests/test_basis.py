import jax
import jax.numpy as jnp

from mechinterp.basis import (
    make_rotation,
    rotate,
    inverse_rotate,
    topk_mask,
    basis_ablation,
)


def test_rotation_invertibility():
    key = jax.random.key(42)
    key1, key2 = jax.random.split(key)

    x = jax.random.normal(key1, (16, 80))
    rotation = make_rotation(key2, 80)

    reconstructed = inverse_rotate(
        rotate(x, rotation),
        rotation,
    )

    assert jnp.allclose(
        x, reconstructed, atol=1e-5
    )


def test_rotation_preserves_norm():
    key = jax.random.key(42)
    key1, key2 = jax.random.split(key)

    x = jax.random.normal(key1, (16, 80))
    rotation = make_rotation(key2, 80)

    z = rotate(x, rotation)

    assert jnp.allclose(
        jnp.linalg.norm(x, axis=-1),
        jnp.linalg.norm(z, axis=-1),
        atol=1e-5,
    )


def test_rotation_preserves_inner_products():
    keys = jax.random.split(jax.random.key(42), 3)

    x = jax.random.normal(keys[0], (16, 80))
    y = jax.random.normal(keys[1], (16, 80))

    rotation = make_rotation(keys[2], 80)

    assert jnp.allclose(
        jnp.sum(x * y, axis=-1),
        jnp.sum(
            rotate(x, rotation) * rotate(y, rotation),
            axis=-1,
        ),
        atol=1e-4,
    )


def test_topk_mask():
    x = jnp.array([[1.0, -5.0, 3.0, 2.0]])

    mask = topk_mask(x, 2)

    expected = jnp.array([
        [False, True, True, False]
    ])

    assert jnp.array_equal(mask, expected)


def test_native_retention():
    x = jnp.array([[1.0, -5.0, 3.0, 2.0]])
    rotation = make_rotation(jax.random.key(42), 4)

    result = basis_ablation(
        x, rotation, False, 2, True
    )

    assert jnp.array_equal(
        result,
        jnp.array([[0.0, -5.0, 3.0, 0.0]]),
    )


def test_native_removal():
    x = jnp.array([[1.0, -5.0, 3.0, 2.0]])
    rotation = make_rotation(jax.random.key(42), 4)

    result = basis_ablation(
        x, rotation, False, 2, False
    )

    assert jnp.array_equal(
        result,
        jnp.array([[1.0, 0.0, 0.0, 2.0]]),
    )


def test_full_rotated_retention():
    keys = jax.random.split(jax.random.key(42))

    x = jax.random.normal(keys[0], (16, 80))
    rotation = make_rotation(keys[1], 80)

    result = basis_ablation(
        x, rotation, True, 80, True
    )

    assert jnp.allclose(x, result, atol=1e-5)


def test_full_rotated_removal():
    keys = jax.random.split(jax.random.key(42))

    x = jax.random.normal(keys[0], (16, 80))
    rotation = make_rotation(keys[1], 80)

    result = basis_ablation(
        x, rotation, True, 80, False
    )

    assert jnp.allclose(result, 0, atol=1e-6)


def test_complementary_interventions():
    keys = jax.random.split(jax.random.key(42))

    x = jax.random.normal(keys[0], (16, 80))
    rotation = make_rotation(keys[1], 80)

    for rotated in (False, True):
        retained = basis_ablation(
            x, rotation, rotated, 20, True
        )

        removed = basis_ablation(
            x, rotation, rotated, 20, False
        )

        assert jnp.allclose(
            retained + removed, x, atol=1e-5
        )