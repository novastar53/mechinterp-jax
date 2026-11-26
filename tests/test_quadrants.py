import jax
import jax.numpy as jnp

from mechinterp.quadrants import (
    PP,
    PN,
    NP,
    NN,
    BASELINE,
    DROP,
    ONLY,
    TOP_K,
    RESTORE_QUADRANT,
    RESTORE_RANDOM,
    RESTORE_LARGEST,
    quadrant_ids,
    quadrant_mask,
    quadrant_statistics,
)


GATE = jnp.array([
    [2., 2., -1., -1., 4., -2.],
    [3., -2., 1., -1., 2., -3.],
])

VALUE = jnp.array([
    [3., -2., 3., -2., 1., -4.],
    [-2., 3., 2., -4., 1., -2.],
])

ACTIVATION = jax.nn.silu(GATE) * VALUE

KEY = jax.random.key(42)


def get_mask(mode, quadrant=NN, keep=2):
    return quadrant_mask(
        GATE,
        VALUE,
        ACTIVATION,
        mode,
        quadrant,
        keep,
        KEY,
    )


def test_quadrant_ids():
    expected = jnp.array([
        [PP, PN, NP, NN, PP, NN],
        [PN, NP, PP, NN, PP, NN],
    ])

    assert jnp.array_equal(
        quadrant_ids(GATE, VALUE),
        expected,
    )


def test_baseline():
    assert jnp.all(get_mask(BASELINE))


def test_drop_quadrants():
    ids = quadrant_ids(GATE, VALUE)

    for quadrant in range(4):
        mask = get_mask(DROP, quadrant)

        assert jnp.array_equal(
            mask, ids != quadrant
        )


def test_keep_only_quadrant():
    ids = quadrant_ids(GATE, VALUE)

    for quadrant in range(4):
        mask = get_mask(ONLY, quadrant)

        assert jnp.array_equal(
            mask, ids == quadrant
        )


def test_top_k():
    mask = get_mask(TOP_K)

    assert jnp.all(mask.sum(axis=-1) == 2)

    retained = jnp.where(
        mask, jnp.abs(ACTIVATION), jnp.inf
    )

    discarded = jnp.where(
        ~mask, jnp.abs(ACTIVATION), -jnp.inf
    )

    assert jnp.all(
        retained.min(axis=-1)
        >= discarded.max(axis=-1)
    )


def test_quadrant_restoration():
    top = get_mask(TOP_K)

    ids = quadrant_ids(GATE, VALUE)

    for quadrant in range(4):
        restored = get_mask(
            RESTORE_QUADRANT, quadrant
        )

        assert jnp.array_equal(
            restored,
            top | (ids == quadrant),
        )


def test_matched_restoration_counts():
    top = get_mask(TOP_K)

    for quadrant in range(4):
        expected = get_mask(
            RESTORE_QUADRANT, quadrant
        )

        for mode in [
            RESTORE_RANDOM,
            RESTORE_LARGEST,
        ]:
            actual = get_mask(mode, quadrant)

            assert jnp.all(actual | ~top)

            assert jnp.array_equal(
                actual.sum(axis=-1),
                expected.sum(axis=-1),
            )


def test_largest_restoration():
    for quadrant in range(4):
        largest = get_mask(
            RESTORE_LARGEST, quadrant
        )

        restored = get_mask(
            RESTORE_QUADRANT, quadrant
        )

        largest_mass = jnp.sum(
            jnp.abs(ACTIVATION) * largest,
            axis=-1,
        )

        restored_mass = jnp.sum(
            jnp.abs(ACTIVATION) * restored,
            axis=-1,
        )

        assert jnp.all(
            largest_mass >= restored_mass - 1e-6
        )


def test_quadrant_statistics():
    mask = get_mask(TOP_K)

    stats = quadrant_statistics(
        GATE,
        VALUE,
        ACTIVATION,
        mask,
    )

    assert stats.shape == (4, 5)

    assert jnp.allclose(
        stats[:, 0].sum(),
        ACTIVATION.size,
    )

    assert jnp.allclose(
        stats[:, 1].sum(),
        jnp.abs(ACTIVATION).sum(),
    )

    assert jnp.allclose(
        stats[:, 2].sum(),
        jnp.square(ACTIVATION).sum(),
    )

    assert jnp.allclose(
        stats[:, 3].sum(),
        mask.sum(),
    )


def test_jit():
    compiled = jax.jit(quadrant_mask)

    for mode in range(7):
        actual = compiled(
            GATE,
            VALUE,
            ACTIVATION,
            jnp.int32(mode),
            jnp.int32(NN),
            jnp.int32(2),
            KEY,
        )

        assert jnp.array_equal(
            actual, get_mask(mode)
        )
