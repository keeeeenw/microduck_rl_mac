"""Unified Metal Simulation Adapter for mjlab ManagerBasedRlEnv.

Executes Microduck forward kinematics, CAD plane-convex contact manifold selection,
joint limits, bilateral BAM friction loss, Delassus PGS solver, and ImplicitFast integration
on Apple Silicon Metal. Self-contact narrowphase and reset-time constant
recomputation use CPU MuJoCo and explicit host staging.
"""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Optional, Sequence, Union
import copy
import numpy as np
import torch
import mujoco

from .representative_physics_slice import RepresentativePhysicsSlice
from mjlab_microduck.native_gpu.simulation import ModelView, Transfer
from mjlab.utils.lab_api.math import quat_from_matrix


def changed_environment_ids(inputs, previous, num_envs: int) -> np.ndarray:
    """Return rows whose randomized model inputs differ from the cached snapshot."""
    if previous is None:
        return np.arange(num_envs, dtype=np.int64)
    changed = np.zeros(num_envs, dtype=bool)
    for key, value in inputs.items():
        old = previous.get(key)
        if (value is None) != (old is None):
            changed[:] = True
            continue
        if value is None:
            continue
        if old is None or old.shape != value.shape:
            changed[:] = True
        else:
            axes = tuple(range(1, value.ndim))
            changed |= np.any(value != old, axis=axes) if axes else value != old
    return np.flatnonzero(changed)


def select_geom_friction(friction: torch.Tensor, geom_ids: Sequence[int]) -> torch.Tensor:
    """Gather friction only for geoms that can enter the self-contact manifold."""
    indices = torch.as_tensor(geom_ids, dtype=torch.long, device=friction.device)
    return friction.index_select(1, indices)


class UnifiedMetalSimulation:
    physics_backend = "metal"

    _output_fields = (
        "time",
        "qpos",
        "qvel",
        "qacc",
        "qacc_warmstart",
        "ctrl",
        "act",
        "qfrc_applied",
        "xfrc_applied",
        "qfrc_actuator",
        "qfrc_bias",
        "qfrc_constraint",
        "xpos",
        "xquat",
        "xmat",
        "xipos",
        "ximat",
        "geom_xpos",
        "geom_xmat",
        "site_xpos",
        "site_xmat",
        "cvel",
        "cacc",
        "subtree_com",
        "subtree_linvel",
        "subtree_angmom",
        "sensordata",
    )

    _input_fields = (
        "qpos",
        "qvel",
        "qacc_warmstart",
        "ctrl",
        "act",
        "qfrc_applied",
        "xfrc_applied",
    )

    def __init__(self, num_envs: int, cfg, model: mujoco.MjModel, device: str = "mps"):
        if str(device) != "mps" or not torch.backends.mps.is_available():
            raise RuntimeError("UnifiedMetalSimulation requires Torch MPS; no CPU fallback")

        self.cfg = cfg
        self.device = "mps"
        self.num_envs = num_envs
        self.mj_model = model
        cfg.mujoco.apply(model)
        self.mj_data = mujoco.MjData(model)
        self.model = ModelView(model, num_envs)
        self.transfer = Transfer()
        # Separate counter for CPU host transfers during reset-time recompute_constants.
        # self.transfer counts only device-resident stepping copies (MJX→MPS, MPS→numpy).
        # self.reset_transfer counts MPS→CPU→MPS round-trips in recompute_constants,
        # so training status.json can report both accurately.
        self.reset_transfer = Transfer()
        # Dirty-row cache for mj_setConst equivalents. The empty caches make the
        # first call a full recomputation; later calls compare all randomized
        # model rows and recompute only rows whose actual values changed.
        self._constant_inputs_cache = None
        self._constant_outputs_cache = None
        self.constants_recomputed_envs = 0
        self.constants_cache_skips = 0
        # FK results are shared only while qpos is the same in-place tensor
        # generation. Call _invalidate_fk_cache() whenever model transforms or
        # the qpos tensor object itself are replaced.
        self._fk_cache_signature = None
        # Explicit counters for runtime CPU narrowphase collision detection fallback work.
        # Tracks staging bytes, full transfer bytes (in+out), evaluation counts, and isolated timers.
        self.collision_transfer = Transfer()
        self.collision_qpos_staging_bytes = 0
        self.collision_fric_staging_bytes = 0
        self.collision_narrowphase_seconds = 0.0
        self.collision_fallback_branch_seconds = 0.0
        self.collision_evaluations_count = 0

        self.expanded_fields = set()
        self.default_model_fields = {}
        self._sensors = []
        self.use_cuda_graph = False

        # Allocate native Metal physics slice
        self.slice = RepresentativePhysicsSlice(
            device="mps",
            autonomous_capacity=64,
            batch_size=num_envs,
        )

        self.data = SimpleNamespace(nworld=num_envs)

        # Basic simulation buffers on MPS
        self.data.time = torch.zeros(num_envs, dtype=torch.float32, device="mps")
        self.data.qpos = torch.as_tensor(
            np.tile(model.qpos0.copy(), (num_envs, 1)),
            dtype=torch.float32,
            device="mps",
        )
        self.data.qvel = torch.zeros((num_envs, 20), dtype=torch.float32, device="mps")
        self.data.qacc = torch.zeros((num_envs, 20), dtype=torch.float32, device="mps")
        self.data.qacc_warmstart = torch.zeros((num_envs, 20), dtype=torch.float32, device="mps")
        self.data.ctrl = torch.zeros((num_envs, 14), dtype=torch.float32, device="mps")
        self.data.act = torch.zeros((num_envs, 0), dtype=torch.float32, device="mps")
        self.data.qfrc_applied = torch.zeros((num_envs, 20), dtype=torch.float32, device="mps")
        self.data.xfrc_applied = torch.zeros((num_envs, 17, 6), dtype=torch.float32, device="mps")
        self.data.qfrc_actuator = torch.zeros((num_envs, 20), dtype=torch.float32, device="mps")
        self.data.qfrc_bias = torch.zeros((num_envs, 20), dtype=torch.float32, device="mps")
        self.data.qfrc_constraint = torch.zeros((num_envs, 20), dtype=torch.float32, device="mps")

        # Kinematics buffers on MPS
        self.data.xpos = torch.zeros((num_envs, 17, 3), dtype=torch.float32, device="mps")
        self.data.xquat = torch.zeros((num_envs, 17, 4), dtype=torch.float32, device="mps")
        self.data.xmat = torch.zeros((num_envs, 17, 9), dtype=torch.float32, device="mps")
        self.data.xipos = torch.zeros((num_envs, 17, 3), dtype=torch.float32, device="mps")
        self.data.ximat = torch.zeros((num_envs, 17, 9), dtype=torch.float32, device="mps")

        # Geoms and sites
        ngeom = model.ngeom
        self.data.geom_xpos = torch.zeros((num_envs, ngeom, 3), dtype=torch.float32, device="mps")
        self.data.geom_xmat = torch.zeros((num_envs, ngeom, 9), dtype=torch.float32, device="mps")

        nsite = model.nsite
        self.data.site_xpos = torch.zeros((num_envs, nsite, 3), dtype=torch.float32, device="mps")
        self.data.site_xmat = torch.zeros((num_envs, nsite, 9), dtype=torch.float32, device="mps")

        # Constant geometric tables for vectorized site and geom forward kinematics on MPS
        self.site_bodyid = torch.as_tensor(model.site_bodyid, dtype=torch.long, device="mps")
        self.site_pos = torch.as_tensor(model.site_pos, dtype=torch.float32, device="mps")
        site_mat = np.zeros((nsite, 3, 3), dtype=np.float32)
        for s in range(nsite):
            res = np.zeros(9, dtype=np.float64)
            mujoco.mju_quat2Mat(res, model.site_quat[s])
            site_mat[s] = res.reshape(3, 3)
        self.site_mat = torch.as_tensor(site_mat, dtype=torch.float32, device="mps")

        self.geom_bodyid = torch.as_tensor(model.geom_bodyid, dtype=torch.long, device="mps")
        self.geom_pos = torch.as_tensor(model.geom_pos, dtype=torch.float32, device="mps")
        geom_mat = np.zeros((ngeom, 3, 3), dtype=np.float32)
        for g in range(ngeom):
            res = np.zeros(9, dtype=np.float64)
            mujoco.mju_quat2Mat(res, model.geom_quat[g])
            geom_mat[g] = res.reshape(3, 3)
        self.geom_mat = torch.as_tensor(geom_mat, dtype=torch.float32, device="mps")

        # Kinematic tree hierarchy constants for recursive spatial velocity
        self.body_parentid = torch.as_tensor(model.body_parentid[:17], dtype=torch.long, device="mps")
        jnt_axis = np.zeros((17, 3), dtype=np.float32)
        dof_adr = np.zeros(17, dtype=np.int64)
        for b in range(3, 17):
            j_id = [j for j in range(model.njnt) if model.jnt_bodyid[j] == b][0]
            jnt_axis[b] = model.jnt_axis[j_id]
            dof_adr[b] = model.jnt_dofadr[j_id]
        self.body_jnt_axis = torch.as_tensor(jnt_axis, dtype=torch.float32, device="mps")
        self.body_dof_adr = torch.as_tensor(dof_adr, dtype=torch.long, device="mps")

        # Velocities and accelerations
        self.data.cvel = torch.zeros((num_envs, 17, 6), dtype=torch.float32, device="mps")
        self.data.cacc = torch.zeros((num_envs, 17, 6), dtype=torch.float32, device="mps")
        self.data.subtree_com = torch.zeros((num_envs, 17, 3), dtype=torch.float32, device="mps")
        self.data.subtree_linvel = torch.zeros((num_envs, 17, 3), dtype=torch.float32, device="mps")
        self.data.subtree_angmom = torch.zeros((num_envs, 17, 3), dtype=torch.float32, device="mps")
        self.data.sensordata = torch.zeros((num_envs, model.nsensordata), dtype=torch.float32, device="mps")

        # Constraint diagnostics buffers for BAM
        capacity = self.slice.autonomous_capacity
        self.data.nefc = torch.zeros((num_envs,), dtype=torch.int32, device="mps")
        self.data.efc = SimpleNamespace(
            type=torch.zeros((num_envs, capacity), dtype=torch.int32, device="mps"),
            id=torch.zeros((num_envs, capacity), dtype=torch.int32, device="mps"),
            force=torch.zeros((num_envs, capacity), dtype=torch.float32, device="mps"),
        )

        # Dynamic entity resolution for task parity
        self.left_foot_body_id = self._find_body_id(["robot/ankle_left", "ankle_left"], 7)
        self.right_foot_body_id = self._find_body_id(["robot/ankle_right", "ankle_right"], 16)
        self.trunk_body_id = self._find_body_id(["robot/trunk_base", "trunk_base"], 2)

        self.terrain_geom_id = self._find_geom_id(["terrain", "floor"], 0)
        self.left_foot_geom_id = self._find_geom_id(["robot/left_foot_collision", "left_foot_collision"], 27)
        self.right_foot_geom_id = self._find_geom_id(["robot/right_foot_collision", "right_foot_collision"], 73)

        contype2_geoms = [
            g for g in range(model.ngeom)
            if model.geom_contype[g] == 2 and model.geom_conaffinity[g] == 2
        ]
        self.trunk_geom_id = next((g for g in contype2_geoms if model.geom_bodyid[g] == self.trunk_body_id), 7)
        self.left_leg_geom_id = next((g for g in contype2_geoms if model.geom_bodyid[g] == 6), 24)
        self.right_leg_geom_id = next((g for g in contype2_geoms if model.geom_bodyid[g] == 15), 70)

        # Dynamic sensor address resolution
        self.left_found_adr = self._find_sensor_adr("left_foot_collision_found", 19)
        self.left_force_adr = self._find_sensor_adr("left_foot_collision_force", 20)
        self.right_found_adr = self._find_sensor_adr("right_foot_collision_found", 23)
        self.right_force_adr = self._find_sensor_adr("right_foot_collision_force", 24)
        self.self_col_adr = self._find_sensor_adr("self_collision", 27)
        self.imu_gyro_adr = self._find_sensor_adr("robot/imu_ang_vel", 7)
        self.ang_vel_adr = self._find_sensor_adr("robot/angular-velocity", 4)
        self.imu_orient_adr = self._find_sensor_adr("robot/orientation", 0)

        # Precompute OBB half-extents and mesh-local center offsets for candidate self-collision geoms.
        # For mesh geoms the geom body origin != mesh centroid; the center offset (in geom/local frame)
        # must be rotated to world frame before calling SAT so OBBs are correctly centered.
        self.e_trunk = self._compute_geom_obb_half_extents(self.trunk_geom_id)
        self.e_left_leg = self._compute_geom_obb_half_extents(self.left_leg_geom_id)
        self.e_right_leg = self._compute_geom_obb_half_extents(self.right_leg_geom_id)
        self.e_left_foot = self._compute_geom_obb_half_extents(self.left_foot_geom_id)
        self.e_right_foot = self._compute_geom_obb_half_extents(self.right_foot_geom_id)

        self.center_trunk = self._compute_geom_mesh_center(self.trunk_geom_id)
        self.center_left_leg = self._compute_geom_mesh_center(self.left_leg_geom_id)
        self.center_right_leg = self._compute_geom_mesh_center(self.right_leg_geom_id)
        self.center_left_foot = self._compute_geom_mesh_center(self.left_foot_geom_id)
        self.center_right_foot = self._compute_geom_mesh_center(self.right_foot_geom_id)

        # Keep a CPU scratch model/data for narrowphase self-collision validation.
        # State (qpos/qvel) is overwritten per narrowphase call; no MPS transfers tracked here
        # because narrowphase is only invoked when the OBB broadphase fires (rare in valid walking).
        self._sc_cpu_model = self.mj_model
        self._sc_cpu_data = mujoco.MjData(self._sc_cpu_model)
        # Geom pair sets for narrowphase counting (contype=2 self-collision pairs)
        self._sc_geom_pairs = frozenset([
            (min(self.trunk_geom_id, self.left_leg_geom_id),
             max(self.trunk_geom_id, self.left_leg_geom_id)),
            (min(self.trunk_geom_id, self.right_leg_geom_id),
             max(self.trunk_geom_id, self.right_leg_geom_id)),
            (min(self.left_leg_geom_id, self.right_leg_geom_id),
             max(self.left_leg_geom_id, self.right_leg_geom_id)),
            (min(self.left_foot_geom_id, self.right_foot_geom_id),
             max(self.left_foot_geom_id, self.right_foot_geom_id)),
        ])
        self._sc_friction_geom_ids = np.asarray(
            sorted({geom_id for pair in self._sc_geom_pairs for geom_id in pair}),
            dtype=np.int64,
        )
        self._sc_friction_geom_slots = {
            int(geom_id): slot
            for slot, geom_id in enumerate(self._sc_friction_geom_ids)
        }

        # Persistent buffers for self-contact constraint assembly and solve
        self.max_extra_contacts = 8
        self.extra_contact_overflow_count = 0
        self._extra_c_pos = torch.zeros((self.num_envs, self.max_extra_contacts, 3), dtype=torch.float32, device=self.device)
        self._extra_c_dist = torch.zeros((self.num_envs, self.max_extra_contacts), dtype=torch.float32, device=self.device)
        self._extra_c_b1 = torch.zeros((self.num_envs, self.max_extra_contacts), dtype=torch.int32, device=self.device)
        self._extra_c_b2 = torch.zeros((self.num_envs, self.max_extra_contacts), dtype=torch.int32, device=self.device)
        self._extra_c_frame = torch.zeros((self.num_envs, self.max_extra_contacts, 9), dtype=torch.float32, device=self.device)
        self._extra_c_fric = torch.zeros((self.num_envs, self.max_extra_contacts, 2), dtype=torch.float32, device=self.device)
        self._extra_ncon = torch.zeros((self.num_envs,), dtype=torch.int32, device=self.device)
        self._current_self_col_count = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        from mjlab.utils.nan_guard import NanGuard
        self.nan_guard = NanGuard(cfg.nan_guard, num_envs, model)

        # Initial forward pass to populate kinematics and bias forces
        self.forward()

    def _find_body_id(self, candidates: Sequence[str], default_id: int) -> int:
        for name in candidates:
            bid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                return bid
        return default_id

    def _find_geom_id(self, candidates: Sequence[str], default_id: int) -> int:
        for name in candidates:
            gid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                return gid
        return default_id

    def _find_sensor_adr(self, substr: str, default_adr: int) -> int:
        for i in range(self.mj_model.nsensor):
            sname = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            if sname and substr in sname:
                return int(self.mj_model.sensor_adr[i])
        return default_adr

    def _compute_geom_obb_half_extents(self, geom_id: int) -> torch.Tensor:
        t = self.mj_model.geom_type[geom_id]
        if t == mujoco.mjtGeom.mjGEOM_BOX:
            return torch.as_tensor(self.mj_model.geom_size[geom_id][:3].copy(), dtype=torch.float32, device="mps")
        elif t == mujoco.mjtGeom.mjGEOM_MESH:
            dataid = self.mj_model.geom_dataid[geom_id]
            vertadr = self.mj_model.mesh_vertadr[dataid]
            vertnum = self.mj_model.mesh_vertnum[dataid]
            verts = self.mj_model.mesh_vert[vertadr : vertadr + vertnum]
            mins = verts.min(axis=0)
            maxs = verts.max(axis=0)
            return torch.as_tensor((maxs - mins) / 2.0, dtype=torch.float32, device="mps")
        else:
            return torch.as_tensor(self.mj_model.geom_size[geom_id][:3].copy(), dtype=torch.float32, device="mps")

    def _compute_geom_mesh_center(self, geom_id: int) -> torch.Tensor:
        """Return the mesh bounding-box center in the local mesh frame (shape (3,), on MPS).

        For mesh geoms, geom_xpos gives the body frame origin of the geom, not the mesh
        centroid.  The center is (max_verts + min_verts) / 2 in local mesh coordinates.
        For non-mesh geoms we assume the geom origin coincides with the geom center,
        so return zeros.
        """
        t = self.mj_model.geom_type[geom_id]
        if t == mujoco.mjtGeom.mjGEOM_MESH:
            dataid = self.mj_model.geom_dataid[geom_id]
            vertadr = self.mj_model.mesh_vertadr[dataid]
            vertnum = self.mj_model.mesh_vertnum[dataid]
            verts = self.mj_model.mesh_vert[vertadr : vertadr + vertnum]
            mins = verts.min(axis=0)
            maxs = verts.max(axis=0)
            center = (maxs + mins) / 2.0
            return torch.as_tensor(center.copy(), dtype=torch.float32, device="mps")
        else:
            return torch.zeros(3, dtype=torch.float32, device="mps")

    @staticmethod
    def _sat_obb_intersection(pA: torch.Tensor, RA: torch.Tensor, eA: torch.Tensor,
                              pB: torch.Tensor, RB: torch.Tensor, eB: torch.Tensor) -> torch.Tensor:
        """Vectorized Separating Axis Theorem (SAT) 3D OBB intersection on GPU."""
        R = torch.bmm(RA.transpose(1, 2), RB)
        Q = torch.abs(R) + 1e-6
        t = torch.bmm(RA.transpose(1, 2), (pB - pA).unsqueeze(-1)).squeeze(-1)
        a0, a1, a2 = eA[0], eA[1], eA[2]
        b0, b1, b2 = eB[0], eB[1], eB[2]
        sep = (torch.abs(t[:, 0]) > a0 + b0 * Q[:, 0, 0] + b1 * Q[:, 0, 1] + b2 * Q[:, 0, 2])
        sep = sep | (torch.abs(t[:, 1]) > a1 + b0 * Q[:, 1, 0] + b1 * Q[:, 1, 1] + b2 * Q[:, 1, 2])
        sep = sep | (torch.abs(t[:, 2]) > a2 + b0 * Q[:, 2, 0] + b1 * Q[:, 2, 1] + b2 * Q[:, 2, 2])
        sep = sep | (torch.abs(t[:, 0]*R[:, 0, 0] + t[:, 1]*R[:, 1, 0] + t[:, 2]*R[:, 2, 0]) > b0 + a0*Q[:, 0, 0] + a1*Q[:, 1, 0] + a2*Q[:, 2, 0])
        sep = sep | (torch.abs(t[:, 0]*R[:, 0, 1] + t[:, 1]*R[:, 1, 1] + t[:, 2]*R[:, 2, 1]) > b1 + a0*Q[:, 0, 1] + a1*Q[:, 1, 1] + a2*Q[:, 2, 1])
        sep = sep | (torch.abs(t[:, 0]*R[:, 0, 2] + t[:, 1]*R[:, 1, 2] + t[:, 2]*R[:, 2, 2]) > b2 + a0*Q[:, 0, 2] + a1*Q[:, 1, 2] + a2*Q[:, 2, 2])
        sep = sep | (torch.abs(t[:, 2]*R[:, 1, 0] - t[:, 1]*R[:, 2, 0]) > a1*Q[:, 2, 0] + a2*Q[:, 1, 0] + b1*Q[:, 0, 2] + b2*Q[:, 0, 1])
        sep = sep | (torch.abs(t[:, 2]*R[:, 1, 1] - t[:, 1]*R[:, 2, 1]) > a1*Q[:, 2, 1] + a2*Q[:, 1, 1] + b0*Q[:, 0, 2] + b2*Q[:, 0, 0])
        sep = sep | (torch.abs(t[:, 2]*R[:, 1, 2] - t[:, 1]*R[:, 2, 2]) > a1*Q[:, 2, 2] + a2*Q[:, 1, 2] + b0*Q[:, 0, 1] + b1*Q[:, 0, 0])
        sep = sep | (torch.abs(t[:, 0]*R[:, 2, 0] - t[:, 2]*R[:, 0, 0]) > a0*Q[:, 2, 0] + a2*Q[:, 0, 0] + b1*Q[:, 1, 2] + b2*Q[:, 1, 1])
        sep = sep | (torch.abs(t[:, 0]*R[:, 2, 1] - t[:, 2]*R[:, 0, 1]) > a0*Q[:, 2, 1] + a2*Q[:, 0, 1] + b0*Q[:, 1, 2] + b2*Q[:, 1, 0])
        sep = sep | (torch.abs(t[:, 0]*R[:, 2, 2] - t[:, 2]*R[:, 0, 2]) > a0*Q[:, 2, 2] + a2*Q[:, 0, 2] + b0*Q[:, 1, 1] + b1*Q[:, 1, 0])
        sep = sep | (torch.abs(t[:, 1]*R[:, 0, 0] - t[:, 0]*R[:, 1, 0]) > a0*Q[:, 1, 0] + a1*Q[:, 0, 0] + b1*Q[:, 2, 2] + b2*Q[:, 2, 1])
        sep = sep | (torch.abs(t[:, 1]*R[:, 0, 1] - t[:, 0]*R[:, 1, 1]) > a0*Q[:, 1, 1] + a1*Q[:, 0, 1] + b0*Q[:, 2, 2] + b2*Q[:, 2, 0])
        sep = sep | (torch.abs(t[:, 1]*R[:, 0, 2] - t[:, 0]*R[:, 1, 2]) > a0*Q[:, 1, 2] + a1*Q[:, 0, 2] + b0*Q[:, 2, 1] + b1*Q[:, 2, 0])
        return ~sep

    def expand_model_fields(self, fields: Sequence[str]):
        for name in fields:
            getattr(self.model, name)
            self.expanded_fields.add(name)

    def get_default_field(self, field: str) -> torch.Tensor:
        if field not in self.default_model_fields:
            self.default_model_fields[field] = torch.as_tensor(
                getattr(self.mj_model, field).copy(),
                device="mps",
                dtype=getattr(self.model, field).dtype,
            )
        return self.default_model_fields[field]

    def _get_model_inputs(self):
        mass = getattr(self.model, "body_mass", None)
        ipos = getattr(self.model, "body_ipos", None)
        inertia = getattr(self.model, "body_inertia", None)
        iquat = getattr(self.model, "body_iquat", None)
        armature = getattr(self.model, "dof_armature", None)
        damping = getattr(self.model, "dof_damping", None)
        dof_fl = getattr(self.model, "dof_frictionloss", None)
        body_inv = getattr(self.model, "body_invweight0", None)
        dof_inv = getattr(self.model, "dof_invweight0", None)
        qfrc_app = getattr(self.data, "qfrc_applied", None)
        xfrc_app = getattr(self.data, "xfrc_applied", None)

        if xfrc_app is not None and (xfrc_app != 0.0).any():
            # Backward spatial wrench propagation (Featherstone §4.2 convention).
            # Computes FRESH body kinematics directly on MPS from current self.data.qpos
            # and current (possibly domain-randomized) body_ipos so projection is never stale.
            self._compute_fk_for_current_state()
            R_bodies = self.slice.body_xmat[:self.num_envs].view(self.num_envs, self.mj_model.nbody, 3, 3)
            xpos_cur = self.slice.body_xpos[:self.num_envs]

            # Current body CoMs in world frame: xipos = xpos + R @ ipos
            if ipos is not None:
                ipos_cur = ipos if ipos.ndim == 3 else ipos.unsqueeze(0).expand(self.num_envs, self.mj_model.nbody, 3)
            else:
                ipos_cur = self.get_default_field("body_ipos").unsqueeze(0).expand(self.num_envs, self.mj_model.nbody, 3)
            xipos_cur = xpos_cur + torch.bmm(R_bodies.view(-1, 3, 3), ipos_cur.view(-1, 3, 1)).view(self.num_envs, self.mj_model.nbody, 3)

            # Update cached simulation data so any immediate caller also sees fresh kinematics
            self.data.xpos.copy_(xpos_cur)
            self.data.xmat.copy_(self.slice.body_xmat[:self.num_envs])
            self.data.xipos.copy_(xipos_cur)

            total_qfrc = qfrc_app.clone() if qfrc_app is not None else torch.zeros(
                self.num_envs, self.mj_model.nv, dtype=torch.float32, device=self.device
            )

            # Carry wrenches (force, torque) for each body in world frame
            # Initialized from xfrc_applied
            f_carry = xfrc_app[:, :, :3].clone()    # (B, nbody, 3)
            tau_carry = xfrc_app[:, :, 3:].clone()  # (B, nbody, 3)

            # Backward pass: b = nbody-1 down to 3 (hinge DOFs)
            for b in range(self.mj_model.nbody - 1, 2, -1):
                p_id = int(self.body_parentid[b])
                dof_adr = int(self.body_dof_adr[b])

                # World-frame joint axis = R_body @ local_axis
                u_b = torch.matmul(
                    R_bodies[:, b],
                    self.body_jnt_axis[b].view(1, 3, 1).expand(self.num_envs, 3, 1)
                ).squeeze(-1)  # (B, 3)

                r_arm = xipos_cur[:, b] - xpos_cur[:, b]  # (B, 3)
                tau_joint = tau_carry[:, b] + torch.cross(r_arm, f_carry[:, b], dim=-1)
                total_qfrc[:, dof_adr] += (u_b * tau_joint).sum(dim=-1)

                # Propagate wrench to parent at parent's CoM (xipos[p_id])
                r_to_parent = xipos_cur[:, b] - xipos_cur[:, p_id]
                f_carry[:, p_id] += f_carry[:, b]
                tau_carry[:, p_id] += tau_carry[:, b] + torch.cross(r_to_parent, f_carry[:, b], dim=-1)

            # Root body 2 (free joint): qfrc[0:3] = world force; qfrc[3:6] = body-frame torque
            R_root = R_bodies[:, self.trunk_body_id]
            r_arm_root = xipos_cur[:, self.trunk_body_id] - xpos_cur[:, self.trunk_body_id]
            tau_root_w = tau_carry[:, self.trunk_body_id] + torch.cross(r_arm_root, f_carry[:, self.trunk_body_id], dim=-1)
            total_qfrc[:, :3] += f_carry[:, self.trunk_body_id]
            total_qfrc[:, 3:6] += torch.bmm(R_root.transpose(1, 2), tau_root_w.unsqueeze(-1)).squeeze(-1)

            qfrc_app = total_qfrc

        per_foot_friction = None
        geom_fric = getattr(self.model, "geom_friction", None)
        if isinstance(geom_fric, torch.Tensor):
            t_mu = geom_fric[:, self.terrain_geom_id, 0]
            fl_mu = geom_fric[:, self.left_foot_geom_id, 0]
            fr_mu = geom_fric[:, self.right_foot_geom_id, 0]
            per_foot_friction = torch.stack([torch.maximum(t_mu, fl_mu), torch.maximum(t_mu, fr_mu)], dim=1)

        return {
            "per_world_mass": mass,
            "per_world_ipos": ipos,
            "per_world_inertia": inertia,
            "per_world_armature": armature,
            "dof_damping": damping,
            "dof_frictionloss": dof_fl,
            "per_world_iquat": iquat,
            "body_invweight0": body_inv,
            "dof_invweight0": dof_inv,
            "per_foot_friction": per_foot_friction,
            "qfrc_applied": qfrc_app,
        }

    def _evaluate_contact_sensors(self, p):
        """Vectorized evaluation of foot ground contact sensors, forces, and self-collision into sensordata."""
        B = self.num_envs
        nconmax = p.contact_dist.shape[1]
        c_idx = torch.arange(nconmax, device=self.device).unsqueeze(0)
        valid_c = c_idx < p.ncon.unsqueeze(1)
        active_c = valid_c & (p.contact_dist <= 0.0)

        is_left = active_c & (p.contact_body == self.left_foot_body_id)
        is_right = active_c & (p.contact_body == self.right_foot_body_id)

        left_found = is_left.sum(dim=1, dtype=torch.float32)
        right_found = is_right.sum(dim=1, dtype=torch.float32)

        # Pyramidal contact forces: 4 facets per active contact c
        cap = p.lambda_force.shape[1]
        max_c = min(nconmax, cap // 4)
        if max_c > 0:
            lam = p.lambda_force[:, :max_c * 4].view(B, max_c, 4)
            lam0 = lam[:, :, 0]
            lam1 = lam[:, :, 1]
            lam2 = lam[:, :, 2]
            lam3 = lam[:, :, 3]

            # Scale tangential forces by assembled contact friction coefficients
            fric = self.slice.assembled_friction[:, :max_c, :]
            mu1 = fric[:, :, 0]
            mu2 = fric[:, :, 1]

            fz_c = lam0 + lam1 + lam2 + lam3
            fy_c = mu1 * (lam0 - lam1)
            fx_c = mu2 * (lam3 - lam2)

            is_left_sub = is_left[:, :max_c]
            is_right_sub = is_right[:, :max_c]

            # In MuJoCo convention, contact sensor records force on ground by foot (-F_foot)
            left_fx = -torch.where(is_left_sub, fx_c, 0.0).sum(dim=1)
            left_fy = -torch.where(is_left_sub, fy_c, 0.0).sum(dim=1)
            left_fz = -torch.where(is_left_sub, fz_c, 0.0).sum(dim=1)

            right_fx = -torch.where(is_right_sub, fx_c, 0.0).sum(dim=1)
            right_fy = -torch.where(is_right_sub, fy_c, 0.0).sum(dim=1)
            right_fz = -torch.where(is_right_sub, fz_c, 0.0).sum(dim=1)
        else:
            left_fx = left_fy = left_fz = torch.zeros(B, dtype=torch.float32, device=self.device)
            right_fx = right_fy = right_fz = torch.zeros(B, dtype=torch.float32, device=self.device)

        # Populate ground contact sensordata slots dynamically
        self.data.sensordata[:, self.left_found_adr] = left_found
        self.data.sensordata[:, self.left_force_adr] = left_fx
        self.data.sensordata[:, self.left_force_adr + 1] = left_fy
        self.data.sensordata[:, self.left_force_adr + 2] = left_fz
        self.data.sensordata[:, self.right_found_adr] = right_found
        self.data.sensordata[:, self.right_force_adr] = right_fx
        self.data.sensordata[:, self.right_force_adr + 1] = right_fy
        self.data.sensordata[:, self.right_force_adr + 2] = right_fz

        # Populate IMU gyro and orientation sensors
        if self.imu_gyro_adr >= 0:
            self.data.sensordata[:, self.imu_gyro_adr:self.imu_gyro_adr + 3] = self.data.qvel[:, 3:6]
        if self.ang_vel_adr >= 0:
            self.data.sensordata[:, self.ang_vel_adr:self.ang_vel_adr + 3] = self.data.qvel[:, 3:6]
        if self.imu_orient_adr >= 0:
            self.data.sensordata[:, self.imu_orient_adr:self.imu_orient_adr + 4] = self.data.qpos[:, 3:7]

        # Populate self-collision sensor
        if self.self_col_adr >= 0:
            self.data.sensordata[:, self.self_col_adr] = self._current_self_col_count

    def _detect_self_contacts(self) -> Optional[Dict[str, torch.Tensor]]:
        """Detects robot-robot self-contacts and prepares geometry tensors for constraint assembly.

        Phase 1 (MPS): Compute corrected OBB centers by rotating the local mesh-center offset
        into world frame and adding it to geom_xpos. Run SAT on each candidate pair.
        This is O(B) on MPS and costs ~0 μs for typical walking configurations.

        Phase 2 (CPU, conditional): For any environment where ANY broadphase pair fires,
        run mj_kinematics + mj_collision on a shared scratch MjData to get the true contact
        set, extract contact geometry (points, distances, frames, body IDs, friction), and
        pack them into device-resident tensors for the constraint solver.
        """
        B = self.num_envs
        self._compute_fk_for_current_state()
        R_bodies = self.slice.body_xmat[:B].view(B, self.mj_model.nbody, 3, 3)
        xpos_bodies = self.slice.body_xpos[:B]

        R_geoms = R_bodies[:, self.geom_bodyid]
        p_geoms = xpos_bodies[:, self.geom_bodyid]
        p_g = p_geoms + torch.matmul(R_geoms, self.geom_pos.unsqueeze(-1)).squeeze(-1)
        R_g = torch.matmul(R_geoms, self.geom_mat)

        self.data.geom_xpos.copy_(p_g)
        self.data.geom_xmat.copy_(R_g.reshape(B, self.mj_model.ngeom, 9))

        def _centered(gid: int, center: torch.Tensor) -> torch.Tensor:
            return p_g[:, gid] + torch.bmm(R_g[:, gid], center.view(1, 3, 1).expand(B, 3, 1)).squeeze(-1)

        hit_ll_rl = self._sat_obb_intersection(
            _centered(self.left_leg_geom_id, self.center_left_leg),
            R_g[:, self.left_leg_geom_id], self.e_left_leg,
            _centered(self.right_leg_geom_id, self.center_right_leg),
            R_g[:, self.right_leg_geom_id], self.e_right_leg,
        )
        hit_lf_rf = self._sat_obb_intersection(
            _centered(self.left_foot_geom_id, self.center_left_foot),
            R_g[:, self.left_foot_geom_id], self.e_left_foot,
            _centered(self.right_foot_geom_id, self.center_right_foot),
            R_g[:, self.right_foot_geom_id], self.e_right_foot,
        )
        hit_t_ll = self._sat_obb_intersection(
            _centered(self.trunk_geom_id, self.center_trunk),
            R_g[:, self.trunk_geom_id], self.e_trunk,
            _centered(self.left_leg_geom_id, self.center_left_leg),
            R_g[:, self.left_leg_geom_id], self.e_left_leg,
        )
        hit_t_rl = self._sat_obb_intersection(
            _centered(self.trunk_geom_id, self.center_trunk),
            R_g[:, self.trunk_geom_id], self.e_trunk,
            _centered(self.right_leg_geom_id, self.center_right_leg),
            R_g[:, self.right_leg_geom_id], self.e_right_leg,
        )

        broadphase_hit = hit_ll_rl | hit_lf_rf | hit_t_ll | hit_t_rl

        if not broadphase_hit.any():
            self._current_self_col_count.zero_()
            self._extra_ncon.zero_()
            return None

        import time as _time
        _t_branch_start = _time.perf_counter()

        _t_copy_in_start = _time.perf_counter()
        hit_env_ids_t = broadphase_hit.nonzero(as_tuple=True)[0]
        hit_env_ids = hit_env_ids_t.cpu().tolist()
        qpos_cpu = self.data.qpos[broadphase_hit].detach().cpu().numpy()

        geom_fric = getattr(self.model, "geom_friction", None)
        if isinstance(geom_fric, torch.Tensor):
            geom_fric_cpu = select_geom_friction(
                geom_fric.index_select(0, hit_env_ids_t), self._sc_friction_geom_ids
            ).detach().cpu().numpy()
            fric_bytes = geom_fric_cpu.nbytes
        else:
            geom_fric_cpu = None
            fric_bytes = 0
        _t_copy_in = _time.perf_counter() - _t_copy_in_start

        n_hit = len(hit_env_ids)
        qpos_bytes = qpos_cpu.nbytes
        id_bytes = n_hit * 8
        self.collision_qpos_staging_bytes += qpos_bytes
        self.collision_fric_staging_bytes += fric_bytes
        self.collision_evaluations_count += n_hit

        _t_geom_start = _time.perf_counter()
        counts = []
        c_pos_host = np.zeros((n_hit, self.max_extra_contacts, 3), dtype=np.float32)
        c_dist_host = np.zeros((n_hit, self.max_extra_contacts), dtype=np.float32)
        c_b1_host = np.zeros((n_hit, self.max_extra_contacts), dtype=np.int32)
        c_b2_host = np.zeros((n_hit, self.max_extra_contacts), dtype=np.int32)
        c_frame_host = np.zeros((n_hit, self.max_extra_contacts, 9), dtype=np.float32)
        c_fric_host = np.zeros((n_hit, self.max_extra_contacts, 2), dtype=np.float32)
        ncon_host = np.zeros((n_hit,), dtype=np.int32)

        for local_i in range(n_hit):
            d = self._sc_cpu_data
            d.qpos[:] = qpos_cpu[local_i]
            d.qvel[:] = 0.0
            mujoco.mj_kinematics(self._sc_cpu_model, d)
            mujoco.mj_collision(self._sc_cpu_model, d)

            ncon = int(d.ncon)
            count = 0
            for ci in range(ncon):
                g1 = int(d.contact[ci].geom1)
                g2 = int(d.contact[ci].geom2)
                pair = (min(g1, g2), max(g1, g2))
                if pair in self._sc_geom_pairs and d.contact[ci].dist <= 0.0:
                    if count < self.max_extra_contacts:
                        c_pos_host[local_i, count] = d.contact[ci].pos
                        c_dist_host[local_i, count] = d.contact[ci].dist
                        c_b1_host[local_i, count] = self.mj_model.geom_bodyid[g1]
                        c_b2_host[local_i, count] = self.mj_model.geom_bodyid[g2]
                        c_frame_host[local_i, count] = d.contact[ci].frame
                        if geom_fric_cpu is not None:
                            slot1 = self._sc_friction_geom_slots[g1]
                            slot2 = self._sc_friction_geom_slots[g2]
                            mu = max(float(geom_fric_cpu[local_i, slot1, 0]), float(geom_fric_cpu[local_i, slot2, 0]))
                            c_fric_host[local_i, count] = [mu, mu]
                        else:
                            c_fric_host[local_i, count] = d.contact[ci].friction[:2]
                    count += 1
            if count > self.max_extra_contacts:
                self.extra_contact_overflow_count += (count - self.max_extra_contacts)
                raise RuntimeError(
                    f"Self-contact capacity overflow in environment {hit_env_ids[local_i]}: "
                    f"detected {count} contacts exceeding max_extra_contacts={self.max_extra_contacts}. "
                    f"Incomplete contact dynamics cannot safely enter rollout."
                )
            counts.append(float(count))
            ncon_host[local_i] = min(count, self.max_extra_contacts)
        _t_geom = _time.perf_counter() - _t_geom_start

        _t_copy_out_start = _time.perf_counter()
        counts_tensor = torch.as_tensor(counts, dtype=torch.float32, device=self.device)
        self._current_self_col_count.zero_()
        self._current_self_col_count.scatter_(0, hit_env_ids_t, counts_tensor)

        self._extra_ncon.zero_()
        self._extra_ncon.scatter_(0, hit_env_ids_t, torch.as_tensor(ncon_host, dtype=torch.int32, device=self.device))
        self._extra_c_pos.zero_()
        self._extra_c_pos[hit_env_ids_t] = torch.as_tensor(c_pos_host, dtype=torch.float32, device=self.device)
        self._extra_c_dist.zero_()
        self._extra_c_dist[hit_env_ids_t] = torch.as_tensor(c_dist_host, dtype=torch.float32, device=self.device)
        self._extra_c_b1.zero_()
        self._extra_c_b1[hit_env_ids_t] = torch.as_tensor(c_b1_host, dtype=torch.int32, device=self.device)
        self._extra_c_b2.zero_()
        self._extra_c_b2[hit_env_ids_t] = torch.as_tensor(c_b2_host, dtype=torch.int32, device=self.device)
        self._extra_c_frame.zero_()
        self._extra_c_frame[hit_env_ids_t] = torch.as_tensor(c_frame_host, dtype=torch.float32, device=self.device)
        self._extra_c_fric.zero_()
        self._extra_c_fric[hit_env_ids_t] = torch.as_tensor(c_fric_host, dtype=torch.float32, device=self.device)
        _t_copy_out = _time.perf_counter() - _t_copy_out_start

        contact_payload_bytes = n_hit * (
            4 + 4 + self.max_extra_contacts * (3 * 4 + 4 + 4 + 4 + 9 * 4 + 2 * 4)
        )
        self.collision_transfer.bytes += (qpos_bytes + fric_bytes + id_bytes + contact_payload_bytes)
        self.collision_transfer.seconds += (_t_copy_in + _t_copy_out)
        self.collision_narrowphase_seconds += _t_geom
        self.collision_fallback_branch_seconds += (_time.perf_counter() - _t_branch_start)

        if self._extra_ncon.max() == 0:
            return None

        return {
            "pos": self._extra_c_pos,
            "dist": self._extra_c_dist,
            "body1": self._extra_c_b1,
            "body2": self._extra_c_b2,
            "frame": self._extra_c_frame,
            "friction": self._extra_c_fric,
            "ncon": self._extra_ncon,
        }


    def _sync_out(self, physics_out):
        """Propagates native GPU physics outputs to persistent simulation data views."""
        p = physics_out
        B = self.num_envs

        self.data.qacc.copy_(p.qacc)
        self.data.qfrc_actuator.copy_(p.qfrc_actuator)
        self.data.qfrc_bias.copy_(p.qfrc_bias)
        self.data.qfrc_constraint.copy_(p.qfrc_constraint)

        self.data.xpos.copy_(p.body_xpos)
        self.data.xmat.copy_(p.body_xmat)
        self.data.xipos.copy_(p.body_xipos)
        self.data.ximat.copy_(p.body_ximat)

        # Quaternions from orientation matrices
        R_bodies = self.data.xmat.view(B, 17, 3, 3)
        self.data.xquat.copy_(quat_from_matrix(R_bodies))
        self.data.xquat[:, 2].copy_(self.data.qpos[:, 3:7])

        # Vectorized site and geom poses on MPS
        R_sites = R_bodies[:, self.site_bodyid]
        p_sites = self.data.xpos[:, self.site_bodyid]
        self.data.site_xpos.copy_(p_sites + torch.matmul(R_sites, self.site_pos.unsqueeze(-1)).squeeze(-1))
        self.data.site_xmat.copy_(torch.matmul(R_sites, self.site_mat).reshape(B, len(self.site_bodyid), 9))

        R_geoms = R_bodies[:, self.geom_bodyid]
        p_geoms = self.data.xpos[:, self.geom_bodyid]
        self.data.geom_xpos.copy_(p_geoms + torch.matmul(R_geoms, self.geom_pos.unsqueeze(-1)).squeeze(-1))
        self.data.geom_xmat.copy_(torch.matmul(R_geoms, self.geom_mat).reshape(B, len(self.geom_bodyid), 9))

        # Subtree CoM
        com_root = p.subtree_com.view(B, 3)
        self.data.subtree_com[:, 2].copy_(com_root)

        # Recursive Spatial Velocity Propagation (cvel) with Root Coordinate Frame Fix
        # Root body 2 (trunk):
        # qvel[:, 3:6] is in LOCAL body frame -> rotate to world frame!
        R_root = R_bodies[:, 2]
        ang_vel_b = self.data.qvel[:, 3:6]
        ang_vel_w = torch.matmul(R_root, ang_vel_b.unsqueeze(-1)).squeeze(-1)
        lin_vel_w = self.data.qvel[:, :3]

        pos_root = self.data.xpos[:, 2]
        offset = com_root - pos_root
        lin_vel_c = lin_vel_w + torch.cross(ang_vel_w, offset, dim=-1)

        self.data.cvel[:, 2, :3].copy_(ang_vel_w)
        self.data.cvel[:, 2, 3:].copy_(lin_vel_c)

        # Children bodies 3..16:
        omega_w = {2: ang_vel_w}
        v_w = {2: lin_vel_w}
        for b in range(3, 17):
            p_id = int(self.body_parentid[b])
            u_b = torch.matmul(R_bodies[:, b], self.body_jnt_axis[b].unsqueeze(-1)).squeeze(-1)
            qdot_b = self.data.qvel[:, self.body_dof_adr[b]].unsqueeze(-1)
            w_b = omega_w[p_id] + qdot_b * u_b
            v_b = v_w[p_id] + torch.cross(omega_w[p_id], self.data.xpos[:, b] - self.data.xpos[:, p_id], dim=-1)
            omega_w[b] = w_b
            v_w[b] = v_b
            self.data.cvel[:, b, :3].copy_(w_b)
            self.data.cvel[:, b, 3:].copy_(v_b + torch.cross(w_b, com_root - self.data.xpos[:, b], dim=-1))

        # Evaluate Contact Sensors
        self._evaluate_contact_sensors(p)

        # Constraint diagnostics for BAM
        self.data.nefc.copy_(p.nefc)
        self.data.efc.type.copy_(p.efc_type)
        if p.efc_id is not None:
            self.data.efc.id.copy_(p.efc_id)
        self.data.efc.force.copy_(p.lambda_force)

    def step(self):
        """Advances simulation by one 5 ms physics substep directly on Metal."""
        self._invalidate_fk_cache()
        with self.nan_guard.watch(self.data):
            dt = float(self.mj_model.opt.timestep)
            params = self._get_model_inputs()
            extra = self._detect_self_contacts()
            extra_kwargs = {}
            if extra is not None:
                extra_kwargs = {
                    "extra_contact_pos": extra["pos"],
                    "extra_contact_dist": extra["dist"],
                    "extra_contact_body1": extra["body1"],
                    "extra_contact_body2": extra["body2"],
                    "extra_contact_frame": extra["frame"],
                    "extra_contact_friction": extra["friction"],
                    "extra_ncon": extra["ncon"],
                }

            step_res = self.slice.step_autonomous(
                self.data.qpos,
                self.data.qvel,
                ctrl=self.data.ctrl,
                dt=dt,
                **extra_kwargs,
                **params,
            )

            # step_autonomous writes the slice's shared FK buffers.
            self._invalidate_fk_cache()
            p = step_res.physics_outputs
            if (p.contact_overflow != 0).any():
                bad_env_ids = torch.where(p.contact_overflow != 0)[0].tolist()
                val = p.contact_overflow[bad_env_ids[0]].item()
                if val > 0:
                    raise RuntimeError(
                        f"Contact capacity overflow ({val} extra contacts beyond capacity) "
                        f"in environments {bad_env_ids}. Incomplete contact constraints cannot enter rollout."
                    )
                else:
                    raise RuntimeError(
                        f"Non-finite contact inputs (status {val}) detected in environments {bad_env_ids}."
                    )
            if (p.solver_status < 0).any():
                bad_env_ids = torch.where(p.solver_status < 0)[0].tolist()
                stat = p.solver_status[bad_env_ids[0]].item()
                raise RuntimeError(
                    f"Solver upstream status failure ({stat}) in environments {bad_env_ids}."
                )

            self._sync_out(step_res.physics_outputs)
            self.data.qpos.copy_(step_res.qpos)
            self.data.qvel.copy_(step_res.qvel)
            self.data.time += dt

    def forward(self):
        """Executes forward kinematics and dynamics without advancing state."""
        self._invalidate_fk_cache()
        params = self._get_model_inputs()
        extra = self._detect_self_contacts()
        extra_kwargs = {}
        if extra is not None:
            extra_kwargs = {
                "extra_contact_pos": extra["pos"],
                "extra_contact_dist": extra["dist"],
                "extra_contact_body1": extra["body1"],
                "extra_contact_body2": extra["body2"],
                "extra_contact_frame": extra["frame"],
                "extra_contact_friction": extra["friction"],
                "extra_ncon": extra["ncon"],
            }
        out = self.slice.forward_autonomous(
            self.data.qpos,
            self.data.qvel,
            ctrl=self.data.ctrl,
            **extra_kwargs,
            **params,
        )
        # forward_autonomous writes the slice's shared FK buffers.
        self._invalidate_fk_cache()
        if (out.contact_overflow != 0).any():
            bad_env_ids = torch.where(out.contact_overflow != 0)[0].tolist()
            val = out.contact_overflow[bad_env_ids[0]].item()
            if val > 0:
                raise RuntimeError(
                    f"Contact capacity overflow ({val} extra contacts beyond capacity) "
                    f"in environments {bad_env_ids}. Incomplete contact constraints cannot enter rollout."
                )
            else:
                raise RuntimeError(
                    f"Non-finite contact inputs (status {val}) detected in environments {bad_env_ids}."
                )
        if (out.solver_status < 0).any():
            bad_env_ids = torch.where(out.solver_status < 0)[0].tolist()
            stat = out.solver_status[bad_env_ids[0]].item()
            raise RuntimeError(
                f"Solver upstream status failure ({stat}) in environments {bad_env_ids}."
            )
        self._sync_out(out)

    def reset(self, env_ids: Optional[torch.Tensor] = None):
        """Resets specified environments to canonical default coordinates."""
        if env_ids is None:
            ids = slice(None)
        else:
            ids = env_ids

        q0 = torch.as_tensor(self.mj_model.qpos0.copy(), dtype=torch.float32, device="mps")
        self.data.qpos[ids] = q0
        self.data.qvel[ids] = 0.0
        self.data.ctrl[ids] = 0.0
        self.data.qfrc_applied[ids] = 0.0
        self.data.xfrc_applied[ids] = 0.0
        self.data.time[ids] = 0.0

        self._invalidate_fk_cache()
        self.forward()

    def _invalidate_fk_cache(self):
        """Invalidate derived poses after reset, restore, or model-transform writes."""
        self._fk_cache_signature = None

    def _compute_fk_for_current_state(self):
        """Compute FK once per in-place qpos version when the version is observable.

        PyTorch increments Tensor._version for in-place updates, including indexed
        resets and copy_ integration. If a tensor implementation does not expose a
        version, disable reuse rather than risk stale contact or force geometry.
        """
        qpos = self.data.qpos
        try:
            version = getattr(qpos, "_version", None)
        except RuntimeError:
            # Inference tensors intentionally do not expose a mutation counter.
            version = None
        if version is None:
            self._invalidate_fk_cache()
            self.slice.compute_forward_kinematics(qpos)
            return
        signature = (id(qpos), int(version))
        if signature != self._fk_cache_signature:
            self.slice.compute_forward_kinematics(qpos)
            self._fk_cache_signature = signature

    def sense(self):
        """Evaluates sensors (e.g. terrain height sensor)."""
        for s in self._sensors:
            s.sense()

    def recompute_constants(self, level, env_ids=None):
        """Domain randomization constants hook matching MuJoCo mj_setConst.

        Performs CPU recomputation of body_subtreemass, dof_invweight0, and body_invweight0
        for environments whose per-world model parameters changed. If env_ids is
        supplied it is treated as the caller's complete dirty-row set. Otherwise,
        changed rows are found by exact comparison with the last synchronized inputs.
        The first call remains a full-batch recomputation.

        Transfer accounting:
          self.reset_transfer — accumulates bytes and time for MPS→CPU copies (inputs) and
                                CPU→MPS copies (outputs) performed here.  This is separate from
                                self.transfer which counts only device-resident stepping copies.
          Net bytes per call  = (in: num_fields × B × field_size) + (out: 3 × B × field_size)
        """
        import time as _time

        names = ("body_subtreemass", "dof_invweight0", "body_invweight0")
        for name in names:
            getattr(self.model, name)
            self.expanded_fields.add(name)

        if not hasattr(self, "_constant_scratch_model"):
            self._constant_scratch_model = copy.copy(self.mj_model)
            self._constant_scratch_data = mujoco.MjData(self._constant_scratch_model)
        scratch_m = self._constant_scratch_model
        scratch_d = self._constant_scratch_data

        mass_t = getattr(self.model, "body_mass", None)
        ipos_t = getattr(self.model, "body_ipos", None)
        inertia_t = getattr(self.model, "body_inertia", None)
        iquat_t = getattr(self.model, "body_iquat", None)
        armature_t = getattr(self.model, "dof_armature", None)

        # Convert supplied IDs once. With no explicit IDs, compare the transferred
        # inputs to find dirtied environments without relying on tensor-wide version
        # counters, which cannot identify changed rows.
        explicit_ids = None
        if env_ids is not None:
            explicit_ids = torch.as_tensor(env_ids, dtype=torch.long, device="cpu").reshape(-1).numpy()
            if explicit_ids.size and (explicit_ids.min() < 0 or explicit_ids.max() >= self.num_envs):
                raise IndexError("env_ids contains an environment outside this simulation")
            explicit_ids = np.unique(explicit_ids)
        if self._constant_inputs_cache is None:
            # The initial synchronization must seed constants for every world.
            explicit_ids = None

        selected_tensor_ids = (
            torch.as_tensor(explicit_ids, dtype=torch.long, device=self.device)
            if explicit_ids is not None
            else None
        )

        def copy_input(tensor):
            if tensor is None:
                return None
            if selected_tensor_ids is not None:
                tensor = tensor.index_select(0, selected_tensor_ids)
            return tensor.detach().cpu().numpy()

        # --- MPS → CPU: instrument each tensor transfer ---
        _t0 = _time.perf_counter()
        mass_cpu     = copy_input(mass_t)
        ipos_cpu     = copy_input(ipos_t)
        inertia_cpu  = copy_input(inertia_t)
        iquat_cpu    = copy_input(iquat_t)
        armature_cpu = copy_input(armature_t)
        _t1 = _time.perf_counter()
        self.reset_transfer.seconds += _t1 - _t0
        for arr in (mass_cpu, ipos_cpu, inertia_cpu, iquat_cpu, armature_cpu):
            if arr is not None:
                self.reset_transfer.bytes += arr.nbytes

        inputs = {
            "mass": mass_cpu,
            "ipos": ipos_cpu,
            "inertia": inertia_cpu,
            "iquat": iquat_cpu,
            "armature": armature_cpu,
        }
        first_recompute = self._constant_inputs_cache is None
        if first_recompute:
            dirty_ids = np.arange(self.num_envs, dtype=np.int64)
            outputs = {
                name: np.empty((self.num_envs, *getattr(self.mj_model, name).shape), dtype=np.float32)
                for name in names
            }
        elif explicit_ids is not None:
            dirty_ids = explicit_ids
            outputs = {key: value.copy() for key, value in self._constant_outputs_cache.items()}
        else:
            dirty_ids = changed_environment_ids(inputs, self._constant_inputs_cache, self.num_envs)
            outputs = {key: value.copy() for key, value in self._constant_outputs_cache.items()}

        if dirty_ids.size == 0:
            self.constants_cache_skips += 1
            self.reset_transfer.seconds += _time.perf_counter() - _t1
            return

        try:
            for local_i, env_i in enumerate(dirty_ids.tolist()):
                source_i = local_i if explicit_ids is not None else env_i
                if mass_cpu is not None:
                    scratch_m.body_mass[:] = mass_cpu[source_i]
                if ipos_cpu is not None:
                    scratch_m.body_ipos[:] = ipos_cpu[source_i]
                if inertia_cpu is not None:
                    scratch_m.body_inertia[:] = inertia_cpu[source_i]
                if iquat_cpu is not None:
                    scratch_m.body_iquat[:] = iquat_cpu[source_i]
                if armature_cpu is not None:
                    scratch_m.dof_armature[:] = armature_cpu[source_i]

                mujoco.mj_setConst(scratch_m, scratch_d)
                for name in names:
                    outputs[name][env_i] = getattr(scratch_m, name)

            # Commit the input snapshot only after all CPU work and output uploads
            # succeeded. A failed partial upload forces a full retry.
            _t2 = _time.perf_counter()
            index = torch.as_tensor(dirty_ids, dtype=torch.long, device=self.device)
            for name in names:
                changed_values = outputs[name][dirty_ids]
                self.reset_transfer.bytes += changed_values.nbytes
                getattr(self.model, name).index_copy_(
                    0, index, torch.as_tensor(changed_values, dtype=torch.float32, device=self.device)
                )
            self.reset_transfer.seconds += _time.perf_counter() - _t2

            input_cache = (
                {key: None if value is None else value.copy() for key, value in inputs.items()}
                if first_recompute
                else {key: None if value is None else value.copy() for key, value in self._constant_inputs_cache.items()}
            )
            if not first_recompute:
                source_ids = np.arange(self.num_envs, dtype=np.int64) if explicit_ids is None else explicit_ids
                for key, value in inputs.items():
                    if value is None:
                        input_cache[key] = None
                    else:
                        if input_cache[key] is None or input_cache[key].shape != (self.num_envs, *value.shape[1:]):
                            input_cache[key] = np.empty((self.num_envs, *value.shape[1:]), dtype=value.dtype)
                        input_cache[key][source_ids] = value
            self._constant_outputs_cache = outputs
            self._constant_inputs_cache = input_cache
            self.constants_recomputed_envs += int(dirty_ids.size)
            self._invalidate_fk_cache()
        except Exception:
            self._constant_inputs_cache = None
            self._constant_outputs_cache = None
            self._invalidate_fk_cache()
            raise

    def ensure_constraint_capacity(self, capacity: int, exact: bool = False):
        """Capacity management hook conforming to simulation checkpoint interface."""
        pass

    def capture_physics(self):
        """Serializes current native GPU simulation and domain randomized parameters."""
        saved = {
            "backend": "metal",
            "time": self.data.time.detach().cpu().clone(),
            "qpos": self.data.qpos.detach().cpu().clone(),
            "qvel": self.data.qvel.detach().cpu().clone(),
            "qacc_warmstart": self.data.qacc_warmstart.detach().cpu().clone(),
            "ctrl": self.data.ctrl.detach().cpu().clone(),
            "qfrc_applied": self.data.qfrc_applied.detach().cpu().clone(),
            "xfrc_applied": self.data.xfrc_applied.detach().cpu().clone(),
            "qfrc_bias": self.data.qfrc_bias.detach().cpu().clone(),
            "transfer_bytes": self.transfer.bytes,
            "reset_transfer_bytes": self.reset_transfer.bytes,
            "collision_transfer_bytes": self.collision_transfer.bytes,
            "collision_qpos_staging_bytes": self.collision_qpos_staging_bytes,
            "collision_fric_staging_bytes": self.collision_fric_staging_bytes,
            "collision_evaluations_count": self.collision_evaluations_count,
            "collision_narrowphase_seconds": self.collision_narrowphase_seconds,
            "collision_transfer_seconds": self.collision_transfer.seconds,
            "collision_fallback_branch_seconds": self.collision_fallback_branch_seconds,
        }
        for name in (
            "body_mass",
            "body_ipos",
            "body_inertia",
            "body_iquat",
            "body_subtreemass",
            "body_invweight0",
            "dof_armature",
            "dof_damping",
            "dof_frictionloss",
            "dof_invweight0",
            "geom_friction",
        ):
            val = getattr(self.model, name, None)
            if isinstance(val, torch.Tensor):
                saved[name] = val.detach().cpu().clone()
        return saved

    def restore_physics(self, saved):
        """Restores native GPU simulation state from checkpoint dictionary."""
        if saved.get("backend") != "metal":
            raise ValueError(f"Incompatible physics checkpoint backend: {saved.get('backend')}")
        # A later tensor copy may fail after earlier fields changed. Invalidate
        # before the first mutation so a caught error cannot reuse stale caches.
        self._constant_inputs_cache = None
        self._constant_outputs_cache = None
        self._invalidate_fk_cache()
        self.data.time.copy_(saved["time"].to(self.device))
        self.data.qpos.copy_(saved["qpos"].to(self.device))
        self.data.qvel.copy_(saved["qvel"].to(self.device))
        self.data.qacc_warmstart.copy_(saved["qacc_warmstart"].to(self.device))
        self.data.ctrl.copy_(saved["ctrl"].to(self.device))
        if "qfrc_applied" in saved:
            self.data.qfrc_applied.copy_(saved["qfrc_applied"].to(self.device))
        if "xfrc_applied" in saved:
            self.data.xfrc_applied.copy_(saved["xfrc_applied"].to(self.device))
        self.data.qfrc_bias.copy_(saved["qfrc_bias"].to(self.device))
        for key in (
            "body_mass",
            "body_ipos",
            "body_inertia",
            "body_iquat",
            "body_subtreemass",
            "body_invweight0",
            "dof_armature",
            "dof_damping",
            "dof_frictionloss",
            "dof_invweight0",
            "geom_friction",
        ):
            if key in saved and hasattr(self.model, key):
                val = getattr(self.model, key)
                if isinstance(val, torch.Tensor):
                    val.copy_(saved[key].to(self.device))
        self.forward()

    def close(self):
        """Releases any simulation resources."""
        pass
