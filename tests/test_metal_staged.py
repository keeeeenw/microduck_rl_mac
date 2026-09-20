import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp
from mjlab_microduck.native_gpu.staged import StagedJit


def test_nested_loops_preserve_original_graph():
    @jax.jit
    def function(x):
        def body(state):
            i, value = state
            value = jax.lax.scan(
                lambda v, _: (v + jnp.sin(v) * 0.01, None), value, None, length=3
            )[0]
            return i + 1, value

        return jax.lax.while_loop(lambda s: s[0] < 4, body, (0, x))[1]

    x = jnp.arange(8, dtype=jnp.float32)
    staged = StagedJit(function)
    np.testing.assert_allclose(
        np.asarray(staged(x)), np.asarray(function(x)), atol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(staged(x + 1)), np.asarray(function(x + 1)), atol=1e-5
    )


def test_branch_and_scan_outputs():
    def function(x):
        result, ys = jax.lax.scan(
            lambda v, y: (v + y, v * y), x, jnp.arange(4.0), reverse=True
        )
        return jax.lax.cond(result > 3, lambda a: a * 2, lambda a: a - 1, result), ys

    a = StagedJit(function)(jnp.array(1.0))
    b = jax.jit(function)(jnp.array(1.0))
    for x, y in zip(a, b):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), atol=1e-5)
