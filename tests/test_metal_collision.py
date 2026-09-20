import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

pytest.importorskip("mujoco.mjx")
from mujoco.mjx._src import mesh, collision_convex
from mujoco.mjx._src.collision_types import GeomInfo
from mjlab_microduck.native_gpu.collision import _narrow, bounded_convex


def box(size):
    c = mesh.box(GeomInfo(jnp.zeros((1, 3)), jnp.eye(3)[None], size[None]))
    return c.replace(
        pos=c.pos[0], mat=c.mat[0], size=c.size[0], vert=c.vert[0], face=c.face[0]
    )


@pytest.mark.parametrize(
    "position", [(0.2, 0.1, 0.05), (2.0, 0.0, 0.0), (0.9, 0.1, 0.1)]
)
def test_bounded_sat_matches_original(position):
    a = box(jnp.array([0.5, 0.4, 0.3]))
    b = a.replace(pos=jnp.array(position))
    result = jax.jit(_narrow)(a, b)
    ref = jax.jit(collision_convex._convex_convex)(a, b)
    for x, y in zip(result, ref):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), atol=1e-5)


def test_nested_batches_keep_broadphase_conditional():
    a = box(jnp.ones(3))
    b = a.replace(pos=jnp.array([3.0, 0.0, 0.0]))
    aa, bb = jax.tree.map(lambda x: jnp.broadcast_to(x, (2, 3) + x.shape), (a, b))
    function = jax.jit(jax.vmap(jax.vmap(bounded_convex)))
    result = function(aa, bb)
    assert result[0].shape == (2, 3, 4)
    np.testing.assert_array_equal(np.asarray(result[0]), 1.0)
    hlo = str(function.lower(aa, bb).compiler_ir())
    assert "stablehlo.case" in hlo


def test_mixed_batch_culls_only_separated_worlds():
    a = box(jnp.array([0.5, 0.4, 0.3]))
    b = a.replace(pos=jnp.array([0.2, 0.1, 0.05]))
    aa, bb = jax.tree.map(lambda x: jnp.broadcast_to(x, (3,) + x.shape), (a, b))
    bb = bb.replace(pos=bb.pos.at[1:].set(jnp.array([3.0, 0.0, 0.0])))
    result = jax.jit(jax.vmap(bounded_convex))(aa, bb)
    reference = jax.jit(collision_convex._convex_convex)(a, b)
    for actual, expected in zip(result, reference):
        np.testing.assert_allclose(
            np.asarray(actual[0]), np.asarray(expected), atol=1e-5
        )
    np.testing.assert_array_equal(np.asarray(result[0][1:]), 1.0)


def test_edge_reduction_across_partial_tiles():
    from mjlab_microduck.native_gpu.collision_sat import _best_edges

    a = box(jnp.array([0.5, 0.4, 0.3]))
    angle = 0.31
    rotation = jnp.array(
        [
            [jnp.cos(angle), -jnp.sin(angle), 0.0],
            [jnp.sin(angle), jnp.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    ea = a.vert[a.edge]
    eb = ea @ rotation.T + jnp.array([0.2, 0.1, 0.1])
    nb = a.edge_face_normal @ rotation.T
    results = []
    for tile in (7, 144):
        fn = jax.jit(
            lambda ea, eb, na, nb: _best_edges(
                jnp.zeros(3), ea, eb, na, nb, tile_size=tile
            )
        )
        results.append(fn(ea, eb, a.edge_face_normal, nb))
    for x, y in zip(*results):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), atol=1e-6)
