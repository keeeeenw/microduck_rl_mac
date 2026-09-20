"""Opt-in bounded SAT collision implementation with conservative AABB culling."""

import importlib.metadata
import math as host_math
from types import FunctionType

import jax
from jax import numpy as jnp
from jax.custom_batching import custom_vmap
from mujoco.mjx._src import collision_convex as convex, collision_driver, math
from mujoco.mjx._src.collision_types import ConvexInfo
from mujoco.mjx._src.types import GeomType

from .collision_sat import _sat_gaussmap

# Keep the original implementation's transforms and manifold construction.
_globals = dict(convex._convex_convex.__globals__, _sat_gaussmap=_sat_gaussmap)
_narrow = FunctionType(convex._convex_convex.__code__, _globals)


@custom_vmap
def bounded_convex(c1, c2):
    batch = c1.pos.shape[:-1]
    n = host_math.prod(batch)
    rank = len(batch)
    a, b = jax.tree.map(lambda x: x.reshape((n,) + x.shape[rank:]), (c1, c2))
    verts_a = a.pos[:, None, :] + jnp.einsum("bvi,bji->bvj", a.vert, a.mat)
    verts_b = b.pos[:, None, :] + jnp.einsum("bvi,bji->bvj", b.vert, b.mat)
    # Roundoff slack only makes the broad phase more conservative. The original
    # collider's SAT reports no contact for separated faces irrespective of margin.
    axes_a, axes_b = a.mat.swapaxes(-1, -2), b.mat.swapaxes(-1, -2)
    crosses = jnp.cross(axes_a[:, :, None, :], axes_b[:, None, :, :]).reshape(n, 9, 3)
    # Any separating direction is a valid conservative rejection. Add sampled
    # original hull normals to reject slender CAD hulls whose boxes overlap.
    ai = jnp.linspace(
        0, a.face_normal.shape[1] - 1, min(32, a.face_normal.shape[1])
    ).astype(jnp.int32)
    bi = jnp.linspace(
        0, b.face_normal.shape[1] - 1, min(32, b.face_normal.shape[1])
    ).astype(jnp.int32)
    normals_a = jnp.einsum("bvi,bji->bvj", a.face_normal[:, ai], a.mat)
    normals_b = jnp.einsum("bvi,bji->bvj", b.face_normal[:, bi], b.mat)
    axes = jnp.concatenate(
        (
            jnp.broadcast_to(jnp.eye(3), (n, 3, 3)),
            axes_a,
            axes_b,
            crosses,
            (a.pos - b.pos)[:, None],
            normals_a,
            normals_b,
        ),
        axis=1,
    )
    projection_a = jnp.einsum("bki,bvi->bkv", axes, verts_a)
    projection_b = jnp.einsum("bki,bvi->bkv", axes, verts_b)
    separated = jnp.any(
        (projection_a.max(-1) + 1e-6 < projection_b.min(-1))
        | (projection_b.max(-1) + 1e-6 < projection_a.min(-1)),
        axis=-1,
    )
    empty = (
        jnp.ones((n, 4)),
        jnp.zeros((n, 4, 3)),
        jnp.broadcast_to(jnp.array([1.0, 0.0, 0.0]), (n, 4, 3)),
    )
    single_empty = jax.tree.map(lambda x: x[0], empty)

    def one(pair):
        first, second, skip = pair
        return jax.lax.cond(skip, lambda: single_empty, lambda: _narrow(first, second))

    # Conditional per-world execution is intentional. vmap(cond) would evaluate
    # both branches and repeat a huge mesh test for every separated world.
    result = jax.lax.cond(
        jnp.all(separated), lambda: empty, lambda: jax.lax.map(one, (a, b, separated))
    )
    return jax.tree.map(lambda x: x.reshape(batch + x.shape[1:]), result)


@bounded_convex.def_vmap
def _batch(axis_size, in_batched, c1, c2):
    def expand(x, batched):
        return x if batched else jnp.broadcast_to(x, (axis_size,) + x.shape)

    c1 = jax.tree.map(expand, c1, in_batched[0])
    c2 = jax.tree.map(expand, c2, in_batched[1])
    result = bounded_convex(c1, c2)
    return result, jax.tree.map(lambda _: True, result)


@convex.collider(ncon=4)
def collider(c1: ConvexInfo, c2: ConvexInfo):
    distance, position, normal = bounded_convex(c1, c2)
    return distance, position, jax.vmap(math.make_frame)(normal)


def register_bounded_collisions():
    if importlib.metadata.version("mujoco-mjx") != "3.10.0":
        raise RuntimeError("Bounded collision adapter requires MJX 3.10.0")
    collision_driver._COLLISION_FUNC[(GeomType.BOX, GeomType.MESH)] = collider
    collision_driver._COLLISION_FUNC[(GeomType.MESH, GeomType.MESH)] = collider
