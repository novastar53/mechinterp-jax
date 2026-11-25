import jax.numpy as jnp
import pytest

from mechinterp.metrics import logit_margin, top_positive_neurons


def test_margin_at_final_position():
    logits = jnp.array([[10., 0., 0.], [0., 5., 2.]])
    assert float(logit_margin(logits, 1, 2)) == pytest.approx(3.0)


def test_rank_positive_neurons():
    scores = jnp.array([[1.0, -2.0], [0.0, 3.0]])
    assert top_positive_neurons(scores, 4) == [
        (1, 1, 3.0), (0, 0, 1.0)
    ]
