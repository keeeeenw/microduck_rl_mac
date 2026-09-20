"""The Metal backend must not route the physics solver through host LAPACK."""

import itertools

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from mjlab_microduck.native_gpu.linalg import cholesky, triangular_solve


def test_batched_cholesky_matches_reference():
    rng = np.random.default_rng(42)
    a = rng.normal(size=(3, 20, 20)).astype(np.float32)
    a = a @ a.swapaxes(-1, -2) + np.eye(20, dtype=np.float32)
    result = np.asarray(jax.jit(cholesky)(jnp.asarray(a)))
    np.testing.assert_allclose(result, np.linalg.cholesky(a), atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(
        result @ result.swapaxes(-1, -2), a, atol=2e-5, rtol=2e-5
    )


@pytest.mark.parametrize(
    "left,lower,transpose,unit", itertools.product([False, True], repeat=4)
)
def test_triangular_solve_all_real_variants(left, lower, transpose, unit):
    rng = np.random.default_rng(12)
    a = rng.normal(scale=0.1, size=(2, 7, 7)).astype(np.float32)
    a = (np.tril(a) if lower else np.triu(a)) + 2 * np.eye(7, dtype=np.float32)
    b = rng.normal(size=(2, 7, 3) if left else (2, 3, 7)).astype(np.float32)
    if unit:
        a[:, np.arange(7), np.arange(7)] = 1
    effective = a.swapaxes(-1, -2) if transpose else a
    expected = (
        np.linalg.solve(effective, b)
        if left
        else np.linalg.solve(effective.swapaxes(-1, -2), b.swapaxes(-1, -2)).swapaxes(
            -1, -2
        )
    )
    solve = jax.jit(
        lambda x, y: triangular_solve(
            x,
            y,
            left_side=left,
            lower=lower,
            transpose_a=transpose,
            conjugate_a=False,
            unit_diagonal=unit,
        )
    )
    np.testing.assert_allclose(
        np.asarray(solve(jnp.asarray(a), jnp.asarray(b))),
        expected,
        atol=2e-5,
        rtol=2e-5,
    )


def test_lowered_solver_contains_no_host_linalg_primitives():
    a, b = jnp.eye(20, dtype=jnp.float32), jnp.ones((20, 2), dtype=jnp.float32)
    lowered = str(
        jax.jit(
            lambda a, b: triangular_solve(
                cholesky(a),
                b,
                left_side=True,
                lower=True,
                transpose_a=False,
                conjugate_a=False,
                unit_diagonal=False,
            )
        )
        .lower(a, b)
        .compiler_ir()
    )
    assert "stablehlo.cholesky" not in lowered
    assert "stablehlo.triangular_solve" not in lowered
    assert "stablehlo.custom_call" not in lowered


def test_invalid_cholesky_does_not_become_a_valid_factor():
    result = np.asarray(jax.jit(cholesky)(jnp.diag(jnp.array([1.0, -1.0]))))
    assert not np.isfinite(result).all()
