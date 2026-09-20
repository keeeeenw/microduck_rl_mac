"""Small dense solver operations expressed entirely in accelerator array primitives.

jax-mps 0.11.0 routes StableHLO Cholesky/triangular_solve through CPU LAPACK.
These explicit opt-in lowerings preserve those operations' mathematics without
changing MuJoCo's integrator or contact solver. CPU/CUDA lowerings are untouched.
This is a correctness implementation for qualification, not a speedup claim.
"""

from functools import partial

import jax
from jax import lax
import jax.numpy as jnp


def _check_matrix(a):
    if a.dtype != jnp.float32:
        raise TypeError("Native GPU physics currently requires real float32 matrices")
    if a.ndim < 2 or a.shape[-1] != a.shape[-2] or a.shape[-1] == 0:
        raise ValueError("Expected a nonempty square matrix or batch of matrices")


def cholesky(a):
    """Lower Cholesky factor of a real SPD matrix, including leading batch axes."""
    _check_matrix(a)
    n = a.shape[-1]
    indices = jnp.arange(n)

    def column(k, factor):
        row = lax.dynamic_index_in_dim(factor, k, axis=-2, keepdims=False)
        correction = jnp.sum(factor * row[..., None, :], axis=-1)
        residual = lax.dynamic_index_in_dim(a, k, axis=-1, keepdims=False) - correction
        diagonal = jnp.sqrt(
            lax.dynamic_index_in_dim(residual, k, axis=-1, keepdims=False)
        )
        values = jnp.where(indices >= k, residual / diagonal[..., None], 0.0)
        return jnp.where(indices == k, values[..., :, None], factor)

    return lax.fori_loop(0, n, column, jnp.zeros_like(a))


def triangular_solve(
    a, b, *, left_side, lower, transpose_a, conjugate_a, unit_diagonal
):
    """Solve a real triangular system without calling a backend linalg routine."""
    _check_matrix(a)
    if b.dtype != jnp.float32:
        raise TypeError(
            "Native GPU physics currently requires real float32 right-hand sides"
        )
    # Conjugation is the identity on real float32. Right-side systems become
    # left-side systems after transposition; batch dimensions remain in place.
    if transpose_a:
        a = jnp.swapaxes(a, -1, -2)
        lower = not lower
    if not left_side:
        a, b = jnp.swapaxes(a, -1, -2), jnp.swapaxes(b, -1, -2)
        lower = not lower
    n = a.shape[-1]
    if b.shape[-2] != n or a.shape[:-2] != b.shape[:-2]:
        raise ValueError(
            "Triangular solve requires matching batch and matrix dimensions"
        )
    indices = jnp.arange(n)

    def row(i, solution):
        k = i if lower else n - i - 1
        coefficients = lax.dynamic_index_in_dim(a, k, axis=-2, keepdims=False)
        # Ignore the unused triangle explicitly, including any NaNs in it.
        active = indices < k if lower else indices > k
        coefficients_used = jnp.where(active, coefficients, 0.0)
        residual = lax.dynamic_index_in_dim(b, k, axis=-2, keepdims=False) - jnp.sum(
            coefficients_used[..., :, None] * solution, axis=-2
        )
        if not unit_diagonal:
            diagonal = lax.dynamic_index_in_dim(
                coefficients, k, axis=-1, keepdims=False
            )
            residual = residual / diagonal[..., None]
        return jnp.where((indices == k)[:, None], residual[..., None, :], solution)

    result = lax.fori_loop(0, n, row, jnp.zeros_like(b))
    return result if left_side else jnp.swapaxes(result, -1, -2)


def register_mps_solver_lowerings():
    """Opt in before compiling MJX; only the pinned MPS backend is modified.

    JAX does not expose stable public lowering registration for these primitives.
    Keep this narrow use of its extension internals pinned and independently tested.
    """
    import importlib.metadata
    from jax._src.interpreters import mlir
    from jax._src.lax import linalg

    versions = {
        name: importlib.metadata.version(name) for name in ("jax", "jaxlib", "jax-mps")
    }
    expected = {"jax": "0.11.1", "jaxlib": "0.11.1", "jax-mps": "0.11.0"}
    if versions != expected:
        raise RuntimeError(
            f"Unqualified GPU lowering versions: {versions}; expected {expected}"
        )
    # Initialize plugins first: plugin initialization itself installs lowerings.
    devices = jax.devices("mps")
    if not devices or any(device.platform != "mps" for device in devices):
        raise RuntimeError("A native MPS device is required; CPU fallback is disabled")
    mlir.register_lowering(
        linalg.cholesky_p,
        mlir.lower_fun(cholesky, multiple_results=False),
        platform="mps",
    )

    def lower_solve(ctx, a, b, **kwargs):
        return mlir.lower_fun(
            partial(triangular_solve, **kwargs), multiple_results=False
        )(ctx, a, b)

    mlir.register_lowering(linalg.triangular_solve_p, lower_solve, platform="mps")
    return versions
