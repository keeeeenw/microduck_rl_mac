# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Bounded-workspace version of MJX 3.10 SAT; unchanged axes/contact equations.

Derived from mujoco.mjx._src.collision_convex._sat_gaussmap (Apache-2.0).
Only support evaluation tiling and the edge-pair maximum reduction differ.
"""

import functools
from typing import Tuple
import jax
from jax import numpy as jp
from mujoco.mjx._src import math
from mujoco.mjx._src.collision_convex import _create_contact_manifold, _arcs_intersect


def _best_edges(centroid, a, b, na, nb, tile_size=262144):
    count = a.shape[0] * b.shape[0]
    tile = min(count, tile_size)
    direction_a = jax.vmap(math.normalize)(a[:, 0] - a[:, 1])
    direction_b = jax.vmap(math.normalize)(b[:, 0] - b[:, 1])

    def candidates(indices):
        ai, bi = indices % a.shape[0], indices // a.shape[0]
        bi = jp.minimum(bi, b.shape[0] - 1)
        axis = jp.cross(direction_a[ai], direction_b[bi])
        degenerate = jp.sum(axis**2, axis=-1) < 1e-6
        axis = jax.vmap(math.normalize)(axis)
        sign = jp.where(jp.sum(axis * (a[ai, 0] - centroid), axis=-1) > 0, 1.0, -1.0)
        axis = axis * sign[:, None]
        valid = jax.vmap(_arcs_intersect)(na[ai, 0], na[ai, 1], -nb[bi, 0], -nb[bi, 1])
        distance = jp.sum(axis * (b[bi, 0] - a[ai, 0]), axis=-1)
        distance = jp.where(valid & ~degenerate & (indices < count), distance, -jp.inf)
        return distance, axis, ai, bi

    def reduce_tile(i, best):
        indices = i * tile + jp.arange(tile)
        distances, _, _, _ = candidates(indices)
        local = jp.argmax(distances)
        better = distances[local] > best[0]
        return jp.where(better, distances[local], best[0]), jp.where(
            better, indices[local], best[1]
        )

    distance, index = jax.lax.fori_loop(
        0, (count + tile - 1) // tile, reduce_tile, (jp.array(-jp.inf), jp.array(0))
    )
    _, axes, ai, bi = candidates(index[None])
    return distance, axes, a[ai, 0], a[ai, 1], b[bi, 0], b[bi, 1]


def _sat_gaussmap(
    centroid_a: jax.Array,
    faces_a: jax.Array,
    faces_b: jax.Array,
    vertices_a: jax.Array,
    vertices_b: jax.Array,
    normals_a: jax.Array,
    normals_b: jax.Array,
    edges_a: jax.Array,
    edges_b: jax.Array,
    edge_face_normals_a: jax.Array,
    edge_face_normals_b: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Runs the Separating Axis Test for a pair of hulls.

    Runs the separating axis test for all faces. Tests edge separating axes via
    edge intersections on gauss maps for all edge pairs. h/t to Dirk Gregorius
    for the implementation details and gauss map trick.

    Args:
      centroid_a: Centroid of hull A.
      faces_a: Faces for hull A.
      faces_b: Faces for hull B.
      vertices_a: Vertices for hull A.
      vertices_b: Vertices for hull B.
      normals_a: Normal vectors for hull A faces.
      normals_b: Normal vectors for hull B faces.
      edges_a: Edges for hull A.
      edges_b: Edges for hull B.
      edge_face_normals_a: Face normals for edges in hull A.
      edge_face_normals_b: Face normals for edges in hull B.

    Returns:
      tuple of dist, pos, and normal
    """
    # Handle face separating axes.
    axes = jp.concatenate([normals_a, -normals_b])

    def get_support(axis):
        # the matmul here is more performant with vmap(dot)
        dot = functools.partial(jp.dot, precision=jax.lax.Precision.HIGH)
        support_a = jax.vmap(dot, in_axes=[None, 0])(axis, vertices_a)
        support_b = jax.vmap(dot, in_axes=[None, 0])(axis, vertices_b)
        dist = support_a.max() - support_b.min()
        separating = dist < 0
        dist = jp.where(dist < 0, 1e6, dist)
        return dist, separating

    support, separating = jax.lax.map(get_support, axes, batch_size=256)
    is_face_separating = separating.any()

    # choose the best separating axis
    best_idx = jp.argmin(support)
    best_axis = axes[best_idx]

    # get the (reference) face most aligned with the separating axis
    dist_a = normals_a @ best_axis
    dist_b = normals_b @ -best_axis
    face_a_idx = dist_a.argmax()
    face_b_idx = dist_b.argmax()

    cond = best_idx < normals_a.shape[0]
    ref_face = jp.where(cond, faces_a[face_a_idx], faces_b[face_b_idx])
    incident_face = jp.where(cond, faces_b[face_b_idx], faces_a[face_a_idx])
    ref_face_norm = jp.where(cond, normals_a[face_a_idx], normals_b[face_b_idx])
    incident_face_norm = jp.where(cond, normals_b[face_b_idx], normals_a[face_a_idx])

    dist, pos, normal = _create_contact_manifold(
        ref_face,
        incident_face,
        ref_face_norm,
        incident_face_norm,
        -best_axis,
    )
    dist = jp.where(is_face_separating, 1.0, dist)

    best_edge_dist, edge_axes, edge_a_pt, edge_a_pt_2, edge_b_pt, edge_b_pt_2 = (
        _best_edges(
            centroid_a, edges_a, edges_b, edge_face_normals_a, edge_face_normals_b
        )
    )
    best_edge_idx = 0
    is_edge_contact = jp.where(
        dist.max() < 0.0,
        best_edge_dist > dist.max() - 1e-6,
        (best_edge_dist < 0) & ~jp.isinf(best_edge_dist),
    )
    is_edge_contact = is_edge_contact & ~is_face_separating
    normal = jp.where(is_edge_contact, edge_axes[best_edge_idx], normal)
    dist = jp.where(
        is_edge_contact,
        jp.array([best_edge_dist, 1, 1, 1]),
        dist,
    )
    a_closest, b_closest = math.closest_segment_to_segment_points(
        edge_a_pt[best_edge_idx],
        edge_a_pt_2[best_edge_idx],
        edge_b_pt[best_edge_idx],
        edge_b_pt_2[best_edge_idx],
    )
    pos = jp.where(is_edge_contact, jp.tile(0.5 * (a_closest + b_closest), (4, 1)), pos)

    return dist, pos, normal
