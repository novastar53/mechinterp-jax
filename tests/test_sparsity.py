import jax
import jax.numpy as jnp

from mechinterp.sparsity import (
    TOP,
    RANDOM,
    BOTTOM,
    sparsify_activations,
)


ACTIVATIONS = jnp.array([
    [1., -8., 3., -2.],
    [-4., 2., 9., 1.],
])


def test_top_k():
    result = sparsify_activations(
        ACTIVATIONS,
        2,
        TOP,
        jax.random.key(0),
    )

    expected = jnp.array([
        [0., -8., 3., 0.],
        [-4., 0., 9., 0.],
    ])

    assert jnp.array_equal(result, expected)


def test_bottom_k():
    result = sparsify_activations(
        ACTIVATIONS,
        2,
        BOTTOM,
        jax.random.key(0),
    )

    expected = jnp.array([
        [1., 0., 0., -2.],
        [0., 2., 0., 1.],
    ])

    assert jnp.array_equal(result, expected)


def test_random_k():
    key = jax.random.key(42)

    result = sparsify_activations(
        ACTIVATIONS, 2, RANDOM, key
    )

    assert jnp.all(
        jnp.count_nonzero(result, axis=-1) == 2
    )

    assert jnp.all(
        (result == 0) | (result == ACTIVATIONS)
    )

    repeated = sparsify_activations(
        ACTIVATIONS, 2, RANDOM, key
    )

    assert jnp.array_equal(result, repeated)


def test_full_retention():
    result = sparsify_activations(
        ACTIVATIONS,
        ACTIVATIONS.shape[-1],
        TOP,
        jax.random.key(0),
    )

    assert jnp.array_equal(result, ACTIVATIONS)


def test_zero_retention():
    result = sparsify_activations(
        ACTIVATIONS,
        0,
        TOP,
        jax.random.key(0),
    )

    assert jnp.array_equal(
        result, jnp.zeros_like(ACTIVATIONS)
    )


def test_tied_activations():
    activations = jnp.ones((3, 10))

    result = sparsify_activations(
        activations,
        3,
        TOP,
        jax.random.key(0),
    )

    assert jnp.all(
        jnp.count_nonzero(result, axis=-1) == 3
    )


def test_dynamic_keep_under_jit():
    compiled = jax.jit(sparsify_activations)
    key = jax.random.key(0)

    for keep in range(5):
        result = compiled(
            ACTIVATIONS,
            jnp.int32(keep),
            jnp.int32(TOP),
            key,
        )

        assert jnp.all(
            jnp.count_nonzero(result, axis=-1) == keep
        )
