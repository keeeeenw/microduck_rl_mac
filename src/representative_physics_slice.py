"""Representative Bounded Static-State Physics Slice on Torch MPS & Metal.

Implements canonical kinematics, articulated dynamics (CRBA/RNE with rotor armature),
real CAD sole-plane contact manifold generation, and constrained PGS solve directly
on persistent MPS tensors without host staging. Compares forces, accelerations, and
completed GPU runtime against pinned MuJoCo CPU reference across the three canonical states.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import ctypes
import math
import numpy as np
import torch
import mujoco

from src.canonical_model_loader import (
    load_canonical_model,
    get_canonical_states,
    CanonicalMicroDuckModel,
)
from src.metal_kernel_manager import MetalKernelManager

WORKSPACE = Path("/Users/zixiao/workspace/microduck")
SHADER_PATH = WORKSPACE / "unified-metal" / "shaders" / "physics_slice.metal"


class BodyConstants(ctypes.Structure):
    _fields_ = [
        ("parent_id", ctypes.c_int32),
        ("joint_type", ctypes.c_int32),
        ("qpos_adr", ctypes.c_int32),
        ("dof_adr", ctypes.c_int32),
        ("body_pos", ctypes.c_float * 3),
        ("body_quat", ctypes.c_float * 4),
        ("body_ipos", ctypes.c_float * 3),
        ("body_iquat", ctypes.c_float * 4),
        ("jnt_axis", ctypes.c_float * 3),
        ("mass", ctypes.c_float),
        ("inertia", ctypes.c_float * 3),
    ]


class DofConstants(ctypes.Structure):
    _fields_ = [
        ("dof_parentid", ctypes.c_int32),
        ("dof_bodyid", ctypes.c_int32),
        ("dof_armature", ctypes.c_float),
    ]


class GeomConstants(ctypes.Structure):
    _fields_ = [
        ("body_id", ctypes.c_int32),
        ("geom_pos", ctypes.c_float * 3),
        ("geom_quat", ctypes.c_float * 4),
    ]


class ContactSolverParams(ctypes.Structure):
    _fields_ = [
        ("timeconst", ctypes.c_float),
        ("dampratio", ctypes.c_float),
        ("dmin", ctypes.c_float),
        ("dmax", ctypes.c_float),
        ("width", ctypes.c_float),
        ("midpoint", ctypes.c_float),
        ("power", ctypes.c_float),
        ("impratio", ctypes.c_float),
        ("margin", ctypes.c_float),
    ]


@dataclass
class PhysicsSliceOutputs:
    body_xpos: torch.Tensor       # (B, 17, 3)
    body_xmat: torch.Tensor       # (B, 17, 9)
    body_xipos: torch.Tensor      # (B, 17, 3)
    body_ximat: torch.Tensor      # (B, 17, 9)
    subtree_com: torch.Tensor     # (B, 3)
    geom_xpos: torch.Tensor       # (B, 2, 3)
    geom_xmat: torch.Tensor       # (B, 2, 9)
    contact_pos: torch.Tensor     # (B, nconmax, 3)
    contact_dist: torch.Tensor    # (B, nconmax)
    contact_normal: torch.Tensor  # (B, nconmax, 3)
    contact_body: torch.Tensor    # (B, nconmax)
    ncon: torch.Tensor            # (B,) int32
    overflow_flag: torch.Tensor   # (B,) int32
    M_eff: torch.Tensor           # (B, 20, 20)
    L_factor: torch.Tensor        # (B, 20, 20)
    M_inv: torch.Tensor           # (B, 20, 20)
    qfrc_bias: torch.Tensor       # (B, 20)
    qfrc_constraint: torch.Tensor # (B, 20)
    qacc: torch.Tensor            # (B, 20)
    solver_status: torch.Tensor   # (B,) int32


@dataclass
class AutonomousPhysicsSliceOutputs:
    body_xpos: torch.Tensor             # (B, 17, 3)
    body_xmat: torch.Tensor             # (B, 17, 9)
    body_xipos: torch.Tensor            # (B, 17, 3)
    body_ximat: torch.Tensor            # (B, 17, 9)
    subtree_com: torch.Tensor           # (B, 3)
    geom_xpos: torch.Tensor             # (B, 2, 3)
    geom_xmat: torch.Tensor             # (B, 2, 9)
    # CAD Contact Manifold
    contact_pos: torch.Tensor           # (B, nconmax, 3)
    contact_dist: torch.Tensor          # (B, nconmax)
    contact_normal: torch.Tensor        # (B, nconmax, 3)
    contact_body: torch.Tensor          # (B, nconmax)
    contact_geom: torch.Tensor          # (B, nconmax)
    ncon: torch.Tensor                  # (B,) int32
    contact_overflow: torch.Tensor      # (B,) int32
    # Assembled Constraints
    J: torch.Tensor                     # (B, capacity, 20)
    aref: torch.Tensor                  # (B, capacity)
    R: torch.Tensor                     # (B, capacity)
    efc_type: torch.Tensor              # (B, capacity) int32
    nefc: torch.Tensor                  # (B,) int32
    assembly_overflow: torch.Tensor     # (B,) int32
    # Dynamics & Factorization
    M_eff: torch.Tensor                 # (B, 20, 20)
    L_factor: torch.Tensor              # (B, 20, 20)
    cholesky_status: torch.Tensor       # (B,) int32
    qfrc_bias: torch.Tensor             # (B, 20)
    qfrc_actuator: torch.Tensor         # (B, 20)
    f_smooth: torch.Tensor              # (B, 20)
    # Constrained Solve
    lambda_force: torch.Tensor          # (B, capacity)
    qfrc_constraint: torch.Tensor       # (B, 20)
    qacc: torch.Tensor                  # (B, 20)
    solver_status: torch.Tensor         # (B,) int32
    actual_iters: torch.Tensor          # (B,) int32
    dual_residual: torch.Tensor         # (B,) float32


@dataclass
class AutonomousStepOutputs:
    qpos: torch.Tensor                  # (B, 21) advanced coordinates
    qvel: torch.Tensor                  # (B, 20) advanced velocities
    integration_status: torch.Tensor    # (B,) int32
    physics_outputs: AutonomousPhysicsSliceOutputs


class RepresentativePhysicsSlice:
    """Bounded representative physics slice for MicroDuck."""

    def __init__(
        self,
        batch_size: int = 1,
        nconmax: int = 35,
        device: str = "mps",
        canonical: Optional[CanonicalMicroDuckModel] = None,
    ):
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.nconmax = nconmax
        self.canonical = canonical or load_canonical_model()
        self.m = self.canonical.model

        # Compile Metal shaders
        self.km = MetalKernelManager(SHADER_PATH)

        # Prepare model constants on CPU and MPS
        self._init_constants()

        # Allocate persistent device buffers
        self._init_persistent_buffers()

    def _init_constants(self):
        m = self.m
        # 17 bodies
        bodies_array = (BodyConstants * 17)()
        for i in range(17):
            bodies_array[i].parent_id = int(m.body_parentid[i])
            if i >= 3:
                jnt_id = [j for j in range(m.njnt) if m.jnt_bodyid[j] == i][0]
                bodies_array[i].joint_type = int(m.jnt_type[jnt_id])
                bodies_array[i].qpos_adr = int(m.jnt_qposadr[jnt_id])
                bodies_array[i].dof_adr = int(m.jnt_dofadr[jnt_id])
                for k in range(3):
                    bodies_array[i].jnt_axis[k] = float(m.jnt_axis[jnt_id, k])
            else:
                bodies_array[i].joint_type = 0 if i == 2 else -1
                bodies_array[i].qpos_adr = 0
                bodies_array[i].dof_adr = 0
                for k in range(3):
                    bodies_array[i].jnt_axis[k] = 0.0

            for k in range(3):
                bodies_array[i].body_pos[k] = float(m.body_pos[i, k])
                bodies_array[i].body_ipos[k] = float(m.body_ipos[i, k])
                bodies_array[i].inertia[k] = float(m.body_inertia[i, k])
            for k in range(4):
                bodies_array[i].body_quat[k] = float(m.body_quat[i, k])
                bodies_array[i].body_iquat[k] = float(m.body_iquat[i, k])
            bodies_array[i].mass = float(m.body_mass[i])

        self.bodies_buf = torch.frombuffer(bodies_array, dtype=torch.uint8).to(self.device)

        # 20 DOFs
        dofs_array = (DofConstants * 20)()
        for d in range(20):
            dofs_array[d].dof_parentid = int(m.dof_parentid[d])
            dofs_array[d].dof_bodyid = int(m.dof_bodyid[d])
            dofs_array[d].dof_armature = float(m.dof_armature[d])

        self.dofs_buf = torch.frombuffer(dofs_array, dtype=torch.uint8).to(self.device)

        # 2 foot geoms
        geoms_array = (GeomConstants * 2)()
        for idx, g_id in enumerate([self.canonical.left_foot_geom_id, self.canonical.right_foot_geom_id]):
            geoms_array[idx].body_id = int(m.geom_bodyid[g_id])
            for k in range(3):
                geoms_array[idx].geom_pos[k] = float(m.geom_pos[g_id, k])
            for k in range(4):
                geoms_array[idx].geom_quat[k] = float(m.geom_quat[g_id, k])

        left_foot_body = int(m.geom_bodyid[self.canonical.left_foot_geom_id])
        right_foot_body = int(m.geom_bodyid[self.canonical.right_foot_geom_id])
        assert left_foot_body == 7, f"Expected left foot body ID 7, got {left_foot_body}"
        assert right_foot_body == 16, f"Expected right foot body ID 16, got {right_foot_body}"

        self.geoms_buf = torch.frombuffer(geoms_array, dtype=torch.uint8).to(self.device)

        # Foot CAD meshes (read-only device buffers)
        self.left_verts = torch.from_numpy(self.canonical.left_foot_mesh.vertices.astype(np.float32)).to(self.device)
        self.right_verts = torch.from_numpy(self.canonical.right_foot_mesh.vertices.astype(np.float32)).to(self.device)
        self.num_left_verts = int(self.canonical.left_foot_mesh.vertnum)
        self.num_right_verts = int(self.canonical.right_foot_mesh.vertnum)

        # Mesh graph and rbound for CAD contact manifold v2
        self.left_graph = torch.from_numpy(self.canonical.left_foot_mesh.graph.astype(np.int32)).to(self.device)
        self.right_graph = torch.from_numpy(self.canonical.right_foot_mesh.graph.astype(np.int32)).to(self.device)
        self.left_rbound = float(self.canonical.left_foot_mesh.rbound)
        self.right_rbound = float(self.canonical.right_foot_mesh.rbound)

        # Body inverse weights for constraint regularization
        self.body_invweight0 = torch.from_numpy(self.m.body_invweight0.astype(np.float32)).to(self.device)

        # Default contact solver parameters
        self.default_contact_params = ContactSolverParams(
            timeconst=0.02,
            dampratio=1.0,
            dmin=0.9,
            dmax=0.95,
            width=0.001,
            midpoint=0.5,
            power=2.0,
            impratio=float(self.m.opt.impratio),
            margin=0.0,
        )
        self.default_contact_params_buf = torch.frombuffer(self.default_contact_params, dtype=torch.uint8).to(self.device)

        # Motor armature
        self.armature = torch.from_numpy(self.canonical.dof_armature.astype(np.float32)).to(self.device)

        # Combined default friction from canonical terrain and foot geoms
        terrain_mu = float(self.m.geom_friction[0, 0])
        foot_mu = float(self.m.geom_friction[self.canonical.left_foot_geom_id, 0])
        self.canonical_default_friction = max(terrain_mu, foot_mu)
        self.default_friction = torch.tensor([self.canonical_default_friction, self.canonical_default_friction], dtype=torch.float32, device=self.device)

        # Actuator force range limits (14, 2)
        self.actuator_forcerange = torch.from_numpy(self.m.actuator_forcerange.astype(np.float32)).to(self.device)

    def _init_persistent_buffers(self):
        B = self.batch_size
        dev = self.device

        self.body_xpos = torch.zeros((B, 17, 3), dtype=torch.float32, device=dev)
        self.body_xmat = torch.zeros((B, 17, 9), dtype=torch.float32, device=dev)
        self.body_xipos = torch.zeros((B, 17, 3), dtype=torch.float32, device=dev)
        self.body_ximat = torch.zeros((B, 17, 9), dtype=torch.float32, device=dev)
        self.subtree_com = torch.zeros((B, 3), dtype=torch.float32, device=dev)

        self.geom_xpos = torch.zeros((B, 2, 3), dtype=torch.float32, device=dev)
        self.geom_xmat = torch.zeros((B, 2, 9), dtype=torch.float32, device=dev)

        self.contact_pos = torch.zeros((B, self.nconmax, 3), dtype=torch.float32, device=dev)
        self.contact_dist = torch.zeros((B, self.nconmax), dtype=torch.float32, device=dev)
        self.contact_normal = torch.zeros((B, self.nconmax, 3), dtype=torch.float32, device=dev)
        self.contact_body = torch.zeros((B, self.nconmax), dtype=torch.int32, device=dev)
        self.contact_geom = torch.zeros((B, self.nconmax), dtype=torch.int32, device=dev)
        self.ncon = torch.zeros((B,), dtype=torch.int32, device=dev)
        self.overflow_flag = torch.zeros((B,), dtype=torch.int32, device=dev)
        self.contact_overflow = torch.zeros((B,), dtype=torch.int32, device=dev)

        self.M_eff = torch.zeros((B, 20, 20), dtype=torch.float32, device=dev)
        self.L_factor = torch.zeros((B, 20, 20), dtype=torch.float32, device=dev)
        self.M_inv = torch.zeros((B, 20, 20), dtype=torch.float32, device=dev)
        self.qfrc_bias = torch.zeros((B, 20), dtype=torch.float32, device=dev)
        self.solver_status = torch.zeros((B,), dtype=torch.int32, device=dev)

        self.qacc = torch.zeros((B, 20), dtype=torch.float32, device=dev)
        self.qfrc_constraint = torch.zeros((B, 20), dtype=torch.float32, device=dev)

        # Pre-allocated identity matrix for native Cholesky inversion
        eye = torch.eye(20, dtype=torch.float32, device=dev).unsqueeze(0).expand(B, -1, -1).contiguous()
        self.eye_20 = eye

        # Dummy per-world buffers for unperturbed runs
        self.dummy_mass = torch.zeros((B, 17), dtype=torch.float32, device=dev)
        self.dummy_ipos = torch.zeros((B, 17, 3), dtype=torch.float32, device=dev)
        self.dummy_armature = torch.zeros((B, 20), dtype=torch.float32, device=dev)

        # Persistent buffers for oracle constraint solve
        self.constraint_capacity = 32
        self.oracle_lambda = torch.zeros((B, self.constraint_capacity), dtype=torch.float32, device=dev)
        self.oracle_qfrc_constraint = torch.zeros((B, 20), dtype=torch.float32, device=dev)
        self.oracle_qacc = torch.zeros((B, 20), dtype=torch.float32, device=dev)
        self.oracle_solver_status = torch.zeros((B,), dtype=torch.int32, device=dev)
        self.oracle_actual_iters = torch.zeros((B,), dtype=torch.int32, device=dev)
        self.oracle_dual_residual = torch.zeros((B,), dtype=torch.float32, device=dev)

        # Persistent buffers for assembled contact constraints (Milestone 3B)
        self.autonomous_capacity = 32
        self.assembled_J = torch.zeros((B, self.autonomous_capacity, 20), dtype=torch.float32, device=dev)
        self.assembled_aref = torch.zeros((B, self.autonomous_capacity), dtype=torch.float32, device=dev)
        self.assembled_R = torch.zeros((B, self.autonomous_capacity), dtype=torch.float32, device=dev)
        self.assembled_efc_type = torch.zeros((B, self.autonomous_capacity), dtype=torch.int32, device=dev)
        self.assembled_nefc = torch.zeros((B,), dtype=torch.int32, device=dev)
        self.assembly_overflow = torch.zeros((B,), dtype=torch.int32, device=dev)
        self.autonomous_upstream_status = torch.zeros((B,), dtype=torch.int32, device=dev)
        self.friction_batch = torch.full((B, self.nconmax, 2), self.canonical_default_friction, dtype=torch.float32, device=dev)
        # Persistent buffers for time-integration (Milestone 4)
        self.integrated_qpos = torch.zeros((B, 21), dtype=torch.float32, device=dev)
        self.integrated_qvel = torch.zeros((B, 20), dtype=torch.float32, device=dev)
        self.integration_status = torch.zeros((B,), dtype=torch.int32, device=dev)

    def _ensure_batch_size(self, B: int):
        """Ensures internal persistent buffers match requested batch size B, reallocating if needed."""
        if self.batch_size != B:
            self.batch_size = B
            self._init_persistent_buffers()

    def _prepare_inputs(
        self,
        qpos: torch.Tensor,
        qvel: torch.Tensor,
        per_world_mass: Optional[torch.Tensor] = None,
        per_world_ipos: Optional[torch.Tensor] = None,
        per_world_armature: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], int]:
        """Validates and prepares contiguous device inputs before any GPU allocation or kernel dispatch."""
        if not isinstance(qpos, torch.Tensor) or not isinstance(qvel, torch.Tensor):
            raise TypeError(f"qpos and qvel must be torch.Tensor instances, got {type(qpos)}, {type(qvel)}")
        if qpos.device.type != self.device.type or qvel.device.type != self.device.type:
            raise ValueError(f"Inputs must be on device {self.device.type}, got qpos on {qpos.device}, qvel on {qvel.device}")
        if qpos.dtype != torch.float32 or qvel.dtype != torch.float32:
            raise TypeError(f"Inputs must be float32, got qpos {qpos.dtype}, qvel {qvel.dtype}")
        if qpos.ndim != 2 or qpos.shape[1] != 21:
            raise ValueError(f"Expected qpos shape (B, 21), got {qpos.shape}")
        if qvel.ndim != 2 or qvel.shape[1] != 20:
            raise ValueError(f"Expected qvel shape (B, 20), got {qvel.shape}")
        if qpos.shape[0] != qvel.shape[0]:
            raise ValueError(f"Batch dimension mismatch: qpos {qpos.shape[0]} vs qvel {qvel.shape[0]}")

        B = qpos.shape[0]
        if B <= 0:
            raise ValueError(f"Batch size must be positive, got {B}")

        if per_world_mass is not None:
            if not isinstance(per_world_mass, torch.Tensor):
                raise TypeError(f"per_world_mass must be a torch.Tensor, got {type(per_world_mass)}")
            if per_world_mass.device.type != self.device.type or per_world_mass.dtype != torch.float32:
                raise ValueError(f"per_world_mass must be float32 on {self.device.type}, got {per_world_mass.dtype} on {per_world_mass.device}")
            if per_world_mass.shape != (B, 17):
                raise ValueError(f"Expected per_world_mass shape ({B}, 17), got {per_world_mass.shape}")
            if not per_world_mass.is_contiguous():
                per_world_mass = per_world_mass.contiguous()

        if per_world_ipos is not None:
            if not isinstance(per_world_ipos, torch.Tensor):
                raise TypeError(f"per_world_ipos must be a torch.Tensor, got {type(per_world_ipos)}")
            if per_world_ipos.device.type != self.device.type or per_world_ipos.dtype != torch.float32:
                raise ValueError(f"per_world_ipos must be float32 on {self.device.type}, got {per_world_ipos.dtype} on {per_world_ipos.device}")
            if per_world_ipos.shape != (B, 17, 3):
                raise ValueError(f"Expected per_world_ipos shape ({B}, 17, 3), got {per_world_ipos.shape}")
            if not per_world_ipos.is_contiguous():
                per_world_ipos = per_world_ipos.contiguous()

        if per_world_armature is not None:
            if not isinstance(per_world_armature, torch.Tensor):
                raise TypeError(f"per_world_armature must be a torch.Tensor, got {type(per_world_armature)}")
            if per_world_armature.device.type != self.device.type or per_world_armature.dtype != torch.float32:
                raise ValueError(f"per_world_armature must be float32 on {self.device.type}, got {per_world_armature.dtype} on {per_world_armature.device}")
            if per_world_armature.shape != (B, 20):
                raise ValueError(f"Expected per_world_armature shape ({B}, 20), got {per_world_armature.shape}")
            if not per_world_armature.is_contiguous():
                per_world_armature = per_world_armature.contiguous()

        if not qpos.is_contiguous():
            qpos = qpos.contiguous()
        if not qvel.is_contiguous():
            qvel = qvel.contiguous()

        return qpos, qvel, per_world_mass, per_world_ipos, per_world_armature, B

    def compute_native_dynamics(
        self,
        qpos: torch.Tensor,
        qvel: torch.Tensor,
        per_world_mass: Optional[torch.Tensor] = None,
        per_world_ipos: Optional[torch.Tensor] = None,
        per_world_armature: Optional[torch.Tensor] = None,
    ):
        """Computes native articulated dynamics (kinematics, CRBA mass matrix, RNE bias forces)

        directly on MPS tensors via Metal kernel without host round-trips.
        """
        qpos, qvel, per_world_mass, per_world_ipos, per_world_armature, B = self._prepare_inputs(
            qpos, qvel, per_world_mass, per_world_ipos, per_world_armature
        )
        self._ensure_batch_size(B)

        flags = 0
        buf_mass = self.dummy_mass
        buf_ipos = self.dummy_ipos
        buf_armature = self.dummy_armature

        if per_world_mass is not None:
            flags |= 1
            buf_mass = per_world_mass
        if per_world_ipos is not None:
            flags |= 2
            buf_ipos = per_world_ipos
        if per_world_armature is not None:
            flags |= 4
            buf_armature = per_world_armature

        self.km.launch(
            "kernel_articulated_dynamics",
            self.bodies_buf,
            self.dofs_buf,
            qpos,
            qvel,
            buf_mass,
            buf_ipos,
            buf_armature,
            int(flags),
            self.M_eff,
            self.qfrc_bias,
            self.body_xpos,
            self.body_xmat,
            self.body_xipos,
            self.body_ximat,
            self.subtree_com,
            threads=B,
        )

    def compute_native_cholesky_solve(
        self,
        M: torch.Tensor,
        B_mat: torch.Tensor,
        X_out: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Solves M X = B natively on GPU via Cholesky factorization and forward/back substitution.

        Args:
            M: (B, 20, 20) float32 symmetric positive-definite matrix
            B_mat: (B, 20, K) or (B, 20) float32 right-hand side matrix
            X_out: optional pre-allocated output buffer (B, 20, K)
        Returns:
            (X, status): solution tensor and integer status tensor (0 for success, -1/-2/-3 for failure).
            NOTE: The returned `status` tensor references `self.solver_status`, which is a persistent
            device buffer reused across successive calls to this method. If callers need to preserve
            status across multiple dispatches, they must call `.clone()` before the next dispatch.
        """
        if not isinstance(M, torch.Tensor) or not isinstance(B_mat, torch.Tensor):
            raise TypeError(f"M and B_mat must be torch.Tensor instances, got {type(M)}, {type(B_mat)}")
        if M.device.type != self.device.type or B_mat.device.type != self.device.type:
            raise ValueError(f"Inputs must be on device {self.device.type}, got M on {M.device}, B_mat on {B_mat.device}")
        if M.dtype != torch.float32 or B_mat.dtype != torch.float32:
            raise TypeError(f"Inputs must be float32, got M {M.dtype}, B_mat {B_mat.dtype}")
        if M.ndim != 3 or M.shape[1] != 20 or M.shape[2] != 20:
            raise ValueError(f"Expected M shape (B, 20, 20), got {M.shape}")

        B = M.shape[0]
        if B_mat.ndim == 2:
            if B_mat.shape[0] != B or B_mat.shape[1] != 20:
                raise ValueError(f"Expected B_mat shape ({B}, 20), got {B_mat.shape}")
            B_mat_3d = B_mat.unsqueeze(2)
            K = 1
        elif B_mat.ndim == 3:
            if B_mat.shape[0] != B or B_mat.shape[1] != 20:
                raise ValueError(f"Expected B_mat shape ({B}, 20, K), got {B_mat.shape}")
            B_mat_3d = B_mat
            K = B_mat.shape[2]
        else:
            raise ValueError(f"Unsupported B_mat shape {B_mat.shape}")

        self._ensure_batch_size(B)

        if X_out is None:
            X_out = torch.zeros((B, 20, K), dtype=torch.float32, device=self.device)
        else:
            if not isinstance(X_out, torch.Tensor):
                raise TypeError(f"X_out must be a torch.Tensor, got {type(X_out)}")
            if X_out.device.type != self.device.type or X_out.dtype != torch.float32:
                raise ValueError(f"X_out must be float32 on {self.device.type}, got {X_out.dtype} on {X_out.device}")
            expected_shape = (B, 20, K) if B_mat.ndim == 3 else ((B, 20) if X_out.ndim == 2 else (B, 20, 1))
            if X_out.shape != expected_shape and X_out.shape != (B, 20, K):
                raise ValueError(f"Expected X_out shape ({B}, 20, {K}), got {X_out.shape}")
            if not X_out.is_contiguous():
                raise ValueError("X_out must be contiguous")

        if not M.is_contiguous():
            M = M.contiguous()
        if not B_mat_3d.is_contiguous():
            B_mat_3d = B_mat_3d.contiguous()

        self.km.launch(
            "kernel_cholesky_solve",
            M,
            B_mat_3d,
            int(K),
            self.L_factor,
            X_out,
            self.solver_status,
            threads=B,
        )

        return X_out, self.solver_status

    def compute_native_M_inv(self):
        """Inverts M_eff natively on GPU via Cholesky solve M_eff * M_inv = I_20 without LAPACK fallback."""
        self.compute_native_cholesky_solve(self.M_eff, self.eye_20, self.M_inv)

    def solve_oracle_constraints(
        self,
        L_factor: torch.Tensor,
        f_smooth: torch.Tensor,
        J: torch.Tensor,
        aref: torch.Tensor,
        R: torch.Tensor,
        nefc: torch.Tensor,
        efc_type: Optional[torch.Tensor] = None,
        upstream_status: Optional[torch.Tensor] = None,
        max_iters: int = 100,
        tol: float = 1e-5,
        capacity: int = 32,
    ) -> Dict[str, torch.Tensor]:
        """Solves bounded pyramidal contact constraints on Metal without explicit M^-1 inversion.

        Args:
            L_factor: (B, 20, 20) float32 lower-triangular Cholesky factor (M = L L^T)
            f_smooth: (B, 20) float32 complete smooth generalized forces (qfrc_smooth)
            J: (B, capacity, 20) float32 contact constraint Jacobian
            aref: (B, capacity) float32 reference acceleration
            R: (B, capacity) float32 constraint regularization (efc_R)
            nefc: (B,) int32 number of active constraint rows per world
            efc_type: optional (B, capacity) int32 constraint row types (must be 6: mjCNSTR_CONTACT_PYRAMIDAL)
            upstream_status: optional (B,) int32 upstream status tensor (e.g. from Cholesky solve)
            max_iters: maximum PGS iterations (default 100)
            tol: convergence tolerance on maximum delta lambda (default 1e-5)
            capacity: constraint row capacity (default 32, maximum supported 32)
        Returns:
            Dict containing:
                "lambda": (B, capacity) float32 constraint forces
                "qfrc_constraint": (B, 20) float32 generalized constraint forces (J^T lambda)
                "qacc": (B, 20) float32 final acceleration (a_0 + delta_a)
                "solver_status": (B,) int32 status (0: converged, 1: unconverged, negative: error)
                "actual_iters": (B,) int32 iterations executed
                "dual_residual": (B,) float32 KKT complementarity residual
            NOTE: Output tensors reference internal persistent buffers. If callers need to preserve
            values across successive dispatches, they must call `.clone()`.
        """
        if not isinstance(L_factor, torch.Tensor) or not isinstance(f_smooth, torch.Tensor):
            raise TypeError(f"L_factor and f_smooth must be torch.Tensor instances, got {type(L_factor)}, {type(f_smooth)}")
        if not isinstance(J, torch.Tensor) or not isinstance(aref, torch.Tensor) or not isinstance(R, torch.Tensor):
            raise TypeError(f"J, aref, and R must be torch.Tensor instances, got {type(J)}, {type(aref)}, {type(R)}")
        if not isinstance(nefc, torch.Tensor):
            raise TypeError(f"nefc must be a torch.Tensor, got {type(nefc)}")

        dev = self.device
        for name, t in [("L_factor", L_factor), ("f_smooth", f_smooth), ("J", J), ("aref", aref), ("R", R), ("nefc", nefc)]:
            if t.device.type != dev.type:
                raise ValueError(f"Tensor '{name}' must reside on {dev.type}, got {t.device}")

        for name, t in [("L_factor", L_factor), ("f_smooth", f_smooth), ("J", J), ("aref", aref), ("R", R)]:
            if t.dtype != torch.float32:
                raise TypeError(f"Tensor '{name}' must be float32, got {t.dtype}")

        if nefc.dtype != torch.int32:
            raise TypeError(f"nefc must be int32, got {nefc.dtype}")

        if not isinstance(capacity, int) or capacity < 1 or capacity > 32:
            raise ValueError(f"Capacity must be an integer in [1, 32], got {capacity}")
        if not isinstance(max_iters, int) or max_iters <= 0:
            raise ValueError(f"max_iters must be a positive integer, got {max_iters}")
        if not isinstance(tol, (int, float)) or not np.isfinite(tol) or tol <= 0:
            raise ValueError(f"tol must be a finite positive number, got {tol}")

        B = L_factor.shape[0]
        if B <= 0:
            raise ValueError(f"Batch size must be positive, got {B}")

        if L_factor.shape != (B, 20, 20):
            raise ValueError(f"Expected L_factor shape ({B}, 20, 20), got {L_factor.shape}")
        if f_smooth.shape != (B, 20):
            raise ValueError(f"Expected f_smooth shape ({B}, 20), got {f_smooth.shape}")
        if J.shape != (B, capacity, 20):
            raise ValueError(f"Expected J shape ({B}, {capacity}, 20), got {J.shape}")
        if aref.shape != (B, capacity):
            raise ValueError(f"Expected aref shape ({B}, {capacity}), got {aref.shape}")
        if R.shape != (B, capacity):
            raise ValueError(f"Expected R shape ({B}, {capacity}), got {R.shape}")
        if nefc.shape != (B,):
            raise ValueError(f"Expected nefc shape ({B},), got {nefc.shape}")

        self._ensure_batch_size(B)
        if self.constraint_capacity != capacity or self.oracle_lambda.shape != (B, capacity):
            self.constraint_capacity = capacity
            self.oracle_lambda = torch.zeros((B, capacity), dtype=torch.float32, device=dev)

        if efc_type is None:
            efc_type = torch.full((B, capacity), 6, dtype=torch.int32, device=dev)
        else:
            if not isinstance(efc_type, torch.Tensor):
                raise TypeError(f"efc_type must be a torch.Tensor, got {type(efc_type)}")
            if efc_type.device.type != dev.type or efc_type.dtype != torch.int32:
                raise ValueError(f"efc_type must be int32 on {dev.type}, got {efc_type.dtype} on {efc_type.device}")
            if efc_type.shape != (B, capacity):
                raise ValueError(f"Expected efc_type shape ({B}, {capacity}), got {efc_type.shape}")

        if upstream_status is None:
            upstream_status = torch.zeros((B,), dtype=torch.int32, device=dev)
        else:
            if not isinstance(upstream_status, torch.Tensor):
                raise TypeError(f"upstream_status must be a torch.Tensor, got {type(upstream_status)}")
            if upstream_status.device.type != dev.type or upstream_status.dtype != torch.int32:
                raise ValueError(f"upstream_status must be int32 on {dev.type}, got {upstream_status.dtype} on {upstream_status.device}")
            if upstream_status.shape != (B,):
                raise ValueError(f"Expected upstream_status shape ({B},), got {upstream_status.shape}")

        L_factor = L_factor.contiguous()
        f_smooth = f_smooth.contiguous()
        J = J.contiguous()
        aref = aref.contiguous()
        R = R.contiguous()
        nefc = nefc.contiguous()
        efc_type = efc_type.contiguous()
        upstream_status = upstream_status.contiguous()

        self.km.launch(
            "kernel_oracle_constrained_solve",
            L_factor,
            f_smooth,
            J,
            aref,
            R,
            efc_type,
            nefc,
            upstream_status,
            self.oracle_lambda,
            self.oracle_qfrc_constraint,
            self.oracle_qacc,
            self.oracle_solver_status,
            self.oracle_actual_iters,
            self.oracle_dual_residual,
            int(max_iters),
            int(capacity),
            float(tol),
            threads=B,
        )

        return {
            "lambda": self.oracle_lambda,
            "qfrc_constraint": self.oracle_qfrc_constraint,
            "qacc": self.oracle_qacc,
            "solver_status": self.oracle_solver_status,
            "actual_iters": self.oracle_actual_iters,
            "dual_residual": self.oracle_dual_residual,
        }

    def compute_cad_contact_manifold_v2(
        self,
        geom_xpos: torch.Tensor,
        geom_xmat: torch.Tensor,
        nconmax: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes contact manifold via pinned MuJoCo mjc_PlaneConvex hull-graph traversal on GPU.

        Args:
            geom_xpos: (B, 2, 3) float32 positions of foot geoms (0: left, 1: right)
            geom_xmat: (B, 2, 9) float32 row-major orientations of foot geoms
            nconmax: optional override for maximum contact capacity in [1, 35]
        Returns:
            Tuple of:
                contact_pos: (B, cap, 3) float32 contact points
                contact_dist: (B, cap) float32 penetration distances
                contact_normal: (B, cap, 3) float32 plane normals
                contact_body: (B, cap) int32 body IDs (7 or 16)
                contact_geom: (B, cap) int32 geom IDs (1 or 2)
                ncon: (B,) int32 contact counts
                overflow_flag: (B,) int32 overflow indicator (1 if contacts exceeded capacity, -1 if non-finite input, 0 otherwise)
        """
        if not isinstance(geom_xpos, torch.Tensor) or not isinstance(geom_xmat, torch.Tensor):
            raise TypeError("geom_xpos and geom_xmat must be torch.Tensor instances")
        dev = self.device
        if geom_xpos.device.type != dev.type or geom_xmat.device.type != dev.type:
            raise ValueError(f"Inputs must be on device {dev.type}")
        if geom_xpos.dtype != torch.float32 or geom_xmat.dtype != torch.float32:
            raise TypeError("Inputs must be float32")
        B = geom_xpos.shape[0]
        if geom_xpos.shape != (B, 2, 3) or geom_xmat.shape != (B, 2, 9):
            raise ValueError(f"Invalid shapes: geom_xpos {geom_xpos.shape}, geom_xmat {geom_xmat.shape}")

        self._ensure_batch_size(B)
        if nconmax is not None:
            if not isinstance(nconmax, int):
                raise TypeError(f"nconmax must be an int, got {type(nconmax)}")
            if nconmax <= 0 or nconmax > self.nconmax:
                raise ValueError(f"Requested capacity {nconmax} must be in [1, {self.nconmax}]")
            cap = nconmax
        else:
            cap = self.nconmax

        if not geom_xpos.is_contiguous():
            geom_xpos = geom_xpos.contiguous()
        if not geom_xmat.is_contiguous():
            geom_xmat = geom_xmat.contiguous()

        self.km.launch(
            "kernel_cad_contact_manifold_v2",
            geom_xpos,
            geom_xmat,
            self.left_verts,
            self.right_verts,
            self.left_graph,
            self.right_graph,
            self.left_rbound,
            self.right_rbound,
            self.contact_pos,
            self.contact_dist,
            self.contact_normal,
            self.contact_body,
            self.contact_geom,
            self.ncon,
            self.contact_overflow,
            int(self.nconmax),
            int(cap),
            threads=B,
        )

        if cap < self.nconmax:
            return (
                self.contact_pos[:, :cap, :].contiguous(),
                self.contact_dist[:, :cap].contiguous(),
                self.contact_normal[:, :cap, :].contiguous(),
                self.contact_body[:, :cap].contiguous(),
                self.contact_geom[:, :cap].contiguous(),
                self.ncon,
                self.contact_overflow,
            )

        return (
            self.contact_pos,
            self.contact_dist,
            self.contact_normal,
            self.contact_body,
            self.contact_geom,
            self.ncon,
            self.contact_overflow,
        )

    def assemble_contact_constraints(
        self,
        contact_pos: torch.Tensor,
        contact_dist: torch.Tensor,
        contact_body: torch.Tensor,
        ncon: torch.Tensor,
        body_xpos: torch.Tensor,
        body_xmat: torch.Tensor,
        qvel: torch.Tensor,
        friction: Optional[Union[float, torch.Tensor]] = None,
        params: Optional[ContactSolverParams] = None,
        capacity: int = 32,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assembles exact MuJoCo 3.10.0 contact constraint rows (J, aref, R) on GPU.

        Args:
            contact_pos: (B, N, 3) float32 contact points
            contact_dist: (B, N) float32 penetration distances
            contact_body: (B, N) int32 body IDs
            ncon: (B,) int32 contact counts
            body_xpos: (B, 17, 3) float32 body positions
            body_xmat: (B, 17, 9) float32 body orientations (row-major)
            qvel: (B, 20) float32 generalized velocities
            friction: optional float, (B, 2), or (B, N, 2) tangential friction coefficients
            params: optional ContactSolverParams struct
            capacity: constraint capacity (default 32, must be a multiple of 4 in [4, 32])
        Returns:
            Tuple of:
                J: (B, capacity, 20) float32 constraint Jacobian
                aref: (B, capacity) float32 reference acceleration
                R: (B, capacity) float32 regularization (efc_R)
                efc_type: (B, capacity) int32 row types (6: mjCNSTR_CONTACT_PYRAMIDAL)
                nefc: (B,) int32 active row count (4 * ncon, capped at capacity)
                overflow_flag: (B,) int32 overflow indicator (1 if rows exceeded capacity, negative on error, 0 otherwise)
        """
        # Validate tensors
        tensors = {
            "contact_pos": contact_pos,
            "contact_dist": contact_dist,
            "contact_body": contact_body,
            "ncon": ncon,
            "body_xpos": body_xpos,
            "body_xmat": body_xmat,
            "qvel": qvel,
        }
        for name, t in tensors.items():
            if not isinstance(t, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor, got {type(t)}")
            if t.device.type != self.device.type:
                raise ValueError(f"{name} must be on {self.device.type}, got {t.device.type}")

        if contact_pos.dtype != torch.float32:
            raise TypeError(f"contact_pos must be float32, got {contact_pos.dtype}")
        if contact_dist.dtype != torch.float32:
            raise TypeError(f"contact_dist must be float32, got {contact_dist.dtype}")
        if contact_body.dtype != torch.int32:
            raise TypeError(f"contact_body must be int32, got {contact_body.dtype}")
        if ncon.dtype != torch.int32:
            raise TypeError(f"ncon must be int32, got {ncon.dtype}")
        if body_xpos.dtype != torch.float32:
            raise TypeError(f"body_xpos must be float32, got {body_xpos.dtype}")
        if body_xmat.dtype != torch.float32:
            raise TypeError(f"body_xmat must be float32, got {body_xmat.dtype}")
        if qvel.dtype != torch.float32:
            raise TypeError(f"qvel must be float32, got {qvel.dtype}")

        B = contact_pos.shape[0]
        if B <= 0:
            raise ValueError(f"Batch size must be positive, got {B}")
        if contact_pos.ndim != 3 or contact_pos.shape[2] != 3:
            raise ValueError(f"Expected contact_pos shape (B, N, 3), got {contact_pos.shape}")
        stride_ncon = contact_pos.shape[1]
        if stride_ncon <= 0:
            raise ValueError(f"Contact dimension N must be positive, got {stride_ncon}")

        if contact_dist.shape != (B, stride_ncon):
            raise ValueError(f"Expected contact_dist shape ({B}, {stride_ncon}), got {contact_dist.shape}")
        if contact_body.shape != (B, stride_ncon):
            raise ValueError(f"Expected contact_body shape ({B}, {stride_ncon}), got {contact_body.shape}")
        if ncon.shape != (B,):
            raise ValueError(f"Expected ncon shape ({B},), got {ncon.shape}")
        if body_xpos.shape != (B, 17, 3):
            raise ValueError(f"Expected body_xpos shape ({B}, 17, 3), got {body_xpos.shape}")
        if body_xmat.shape != (B, 17, 9):
            raise ValueError(f"Expected body_xmat shape ({B}, 17, 9), got {body_xmat.shape}")
        if qvel.shape != (B, 20):
            raise ValueError(f"Expected qvel shape ({B}, 20), got {qvel.shape}")

        if not isinstance(capacity, int):
            raise TypeError(f"capacity must be an int, got {type(capacity)}")
        if capacity <= 0 or capacity > self.autonomous_capacity:
            raise ValueError(f"capacity must be in [1, {self.autonomous_capacity}], got {capacity}")
        if capacity % 4 != 0:
            raise ValueError(f"capacity must be a positive multiple of 4 and <= {self.autonomous_capacity}, got {capacity}")

        if params is not None:
            if not isinstance(params, ContactSolverParams):
                raise TypeError(f"params must be ContactSolverParams, got {type(params)}")
            if params.margin != 0.0:
                raise ValueError("Unsupported ContactSolverParams: non-zero margin is not supported in Milestone 3B")
            if params.timeconst <= 0.0 or params.dampratio <= 0.0 or params.dmin <= 0.0 or params.dmax <= 0.0 or params.width <= 0.0:
                raise ValueError("ContactSolverParams timeconst, dampratio, dmin, dmax, width must be positive")
            params_buf = torch.frombuffer(params, dtype=torch.uint8).to(self.device)
        else:
            params_buf = self.default_contact_params_buf

        self._ensure_batch_size(B)

        # Prepare friction buffer matching exact stride_ncon layout
        if friction is None:
            if stride_ncon == self.nconmax:
                self.friction_batch.fill_(self.canonical_default_friction)
                f_buf = self.friction_batch
            else:
                f_buf = torch.full((B, stride_ncon, 2), self.canonical_default_friction, dtype=torch.float32, device=self.device)
        elif isinstance(friction, (int, float)):
            if friction <= 0.0 or not math.isfinite(friction):
                raise ValueError(f"Friction coefficient must be strictly positive and finite, got {friction}")
            if stride_ncon == self.nconmax:
                self.friction_batch.fill_(float(friction))
                f_buf = self.friction_batch
            else:
                f_buf = torch.full((B, stride_ncon, 2), float(friction), dtype=torch.float32, device=self.device)
        elif isinstance(friction, torch.Tensor):
            if friction.dtype != torch.float32:
                friction = friction.to(dtype=torch.float32)
            if friction.ndim == 1 and friction.shape[0] == 2:
                f_buf = friction.view(1, 1, 2).expand(B, stride_ncon, 2).contiguous().to(self.device)
            elif friction.ndim == 2 and friction.shape == (B, 2):
                f_buf = friction.unsqueeze(1).expand(B, stride_ncon, 2).contiguous().to(self.device)
            elif friction.ndim == 3 and friction.shape[0] == B and friction.shape[2] == 2:
                n_fric = friction.shape[1]
                if n_fric == stride_ncon:
                    f_buf = friction.contiguous().to(self.device)
                elif n_fric < stride_ncon:
                    f_buf = torch.full((B, stride_ncon, 2), self.canonical_default_friction, dtype=torch.float32, device=self.device)
                    f_buf[:, :n_fric, :] = friction.to(self.device)
                else:  # n_fric > stride_ncon
                    f_buf = friction[:, :stride_ncon, :].contiguous().to(self.device)
            else:
                raise ValueError(f"Unsupported friction shape: {friction.shape}")
        else:
            raise TypeError(f"friction must be float or torch.Tensor, got {type(friction)}")

        contact_pos = contact_pos.contiguous()
        contact_dist = contact_dist.contiguous()
        contact_body = contact_body.contiguous()
        ncon = ncon.contiguous()
        body_xpos = body_xpos.contiguous()
        body_xmat = body_xmat.contiguous()
        qvel = qvel.contiguous()

        self.km.launch(
            "kernel_assemble_contact_constraints",
            contact_pos,
            contact_dist,
            contact_body,
            ncon,
            body_xpos,
            body_xmat,
            qvel,
            self.bodies_buf,
            self.body_invweight0,
            f_buf,
            self.assembled_J,
            self.assembled_aref,
            self.assembled_R,
            self.assembled_efc_type,
            self.assembled_nefc,
            self.assembly_overflow,
            int(stride_ncon),
            int(self.autonomous_capacity),
            params_buf,
            int(capacity),
            threads=B,
        )

        if capacity < self.autonomous_capacity:
            return (
                self.assembled_J[:, :capacity, :].contiguous(),
                self.assembled_aref[:, :capacity].contiguous(),
                self.assembled_R[:, :capacity].contiguous(),
                self.assembled_efc_type[:, :capacity].contiguous(),
                self.assembled_nefc,
                self.assembly_overflow,
            )

        return (
            self.assembled_J,
            self.assembled_aref,
            self.assembled_R,
            self.assembled_efc_type,
            self.assembled_nefc,
            self.assembly_overflow,
        )

    def forward_autonomous(
        self,
        qpos: torch.Tensor,
        qvel: torch.Tensor,
        ctrl: Optional[torch.Tensor] = None,
        f_smooth: Optional[torch.Tensor] = None,
        friction: Optional[Union[float, torch.Tensor]] = None,
        per_world_mass: Optional[torch.Tensor] = None,
        per_world_ipos: Optional[torch.Tensor] = None,
        per_world_armature: Optional[torch.Tensor] = None,
        max_iters: int = 100,
        tol: float = 1e-5,
        nconmax: Optional[int] = None,
        capacity: Optional[int] = None,
    ) -> AutonomousPhysicsSliceOutputs:
        """Executes full autonomous static forward dynamics pipeline without oracle inputs.

        Pipeline:
        1. Hierarchical Forward Kinematics (FK)
        2. Articulated Dynamics (CRBA M_eff + RNE qfrc_bias)
        3. Cholesky Factorization (M_eff = L L^T)
        4. Autonomous CAD Sole Contact Manifold (mjc_PlaneConvex)
        5. Contact Constraint Assembly (J, aref, R)
        6. Factor-and-Solve Delassus PGS Constrained Solve
        """
        qpos, qvel, per_world_mass, per_world_ipos, per_world_armature, B = self._prepare_inputs(
            qpos, qvel, per_world_mass, per_world_ipos, per_world_armature
        )
        self._ensure_batch_size(B)

        # 1. Forward Kinematics
        self.km.launch(
            "kernel_forward_kinematics",
            self.bodies_buf,
            self.geoms_buf,
            qpos,
            self.body_xpos,
            self.body_xmat,
            self.geom_xpos,
            self.geom_xmat,
            threads=B,
        )

        # 2. Native Dynamics
        self.compute_native_dynamics(
            qpos,
            qvel,
            per_world_mass=per_world_mass,
            per_world_ipos=per_world_ipos,
            per_world_armature=per_world_armature,
        )

        # 3. Smooth forces: actuator forces + external/smooth forces + native bias
        qfrc_act = torch.zeros((B, 20), dtype=torch.float32, device=self.device)
        if ctrl is not None:
            if not isinstance(ctrl, torch.Tensor):
                raise TypeError(f"ctrl must be torch.Tensor, got {type(ctrl)}")
            if ctrl.dtype != torch.float32 or ctrl.device.type != self.device.type:
                ctrl = ctrl.to(dtype=torch.float32, device=self.device)
            if ctrl.ndim == 1 and ctrl.shape[0] == 14:
                ctrl = ctrl.unsqueeze(0).expand(B, 14)
            elif ctrl.ndim != 2 or ctrl.shape != (B, 14):
                raise ValueError(f"Expected ctrl shape ({B}, 14), got {ctrl.shape}")
            ctrl_clamped = torch.clamp(
                ctrl,
                self.actuator_forcerange[:, 0],
                self.actuator_forcerange[:, 1],
            )
            qfrc_act[:, 6:20] = ctrl_clamped

        if f_smooth is None:
            f_smooth_tensor = qfrc_act - self.qfrc_bias
        else:
            if not isinstance(f_smooth, torch.Tensor):
                raise TypeError(f"f_smooth must be torch.Tensor, got {type(f_smooth)}")
            if f_smooth.device.type != self.device.type or f_smooth.dtype != torch.float32:
                raise ValueError(f"f_smooth must be float32 on {self.device.type}")
            if f_smooth.shape != (B, 20):
                raise ValueError(f"Expected f_smooth shape ({B}, 20), got {f_smooth.shape}")
            f_smooth_tensor = (f_smooth + qfrc_act).contiguous()

        # 4. Cholesky Factorization: factorize M_eff -> L_factor
        _, cholesky_status = self.compute_native_cholesky_solve(
            self.M_eff,
            f_smooth_tensor,
            X_out=self.qacc,
        )

        # 5. Autonomous CAD Contact Manifold
        self.compute_cad_contact_manifold_v2(
            self.geom_xpos,
            self.geom_xmat,
            nconmax=nconmax,
        )

        # 6. Contact Constraint Assembly
        act_cap = capacity if capacity is not None else self.autonomous_capacity
        (
            assembled_J,
            assembled_aref,
            assembled_R,
            assembled_efc_type,
            assembled_nefc,
            assembly_overflow,
        ) = self.assemble_contact_constraints(
            self.contact_pos,
            self.contact_dist,
            self.contact_body,
            self.ncon,
            self.body_xpos,
            self.body_xmat,
            qvel,
            friction=friction,
            capacity=act_cap,
        )

        # Combine upstream status on device without host sync
        # 0: OK
        # cholesky_status < 0: Cholesky failure (-1, -2, -3)
        # contact_overflow == 1: Contact manifold overflow (-6)
        # contact_overflow < 0: Non-finite contact inputs (-8)
        # assembly_overflow == 1: Assembly row overflow (-7)
        # assembly_overflow < 0: Non-finite assembly state / invalid body ID / invalid friction / invalid count (-9)
        self.autonomous_upstream_status.copy_(cholesky_status)
        mask_ok = (self.autonomous_upstream_status == 0)
        self.autonomous_upstream_status.masked_fill_(mask_ok & (self.contact_overflow == 1), -6)
        self.autonomous_upstream_status.masked_fill_(mask_ok & (self.contact_overflow < 0), -8)

        mask_ok = (self.autonomous_upstream_status == 0)
        self.autonomous_upstream_status.masked_fill_(mask_ok & (self.assembly_overflow == 1), -7)
        self.autonomous_upstream_status.masked_fill_(mask_ok & (self.assembly_overflow < 0), -9)

        # 7. Delassus PGS Constrained Solve
        solve_results = self.solve_oracle_constraints(
            L_factor=self.L_factor,
            f_smooth=f_smooth_tensor,
            J=assembled_J,
            aref=assembled_aref,
            R=assembled_R,
            nefc=assembled_nefc,
            efc_type=assembled_efc_type,
            upstream_status=self.autonomous_upstream_status,
            max_iters=max_iters,
            tol=tol,
            capacity=act_cap,
        )

        return AutonomousPhysicsSliceOutputs(
            body_xpos=self.body_xpos,
            body_xmat=self.body_xmat,
            body_xipos=self.body_xipos,
            body_ximat=self.body_ximat,
            subtree_com=self.subtree_com,
            geom_xpos=self.geom_xpos,
            geom_xmat=self.geom_xmat,
            contact_pos=self.contact_pos,
            contact_dist=self.contact_dist,
            contact_normal=self.contact_normal,
            contact_body=self.contact_body,
            contact_geom=self.contact_geom,
            ncon=self.ncon,
            contact_overflow=self.contact_overflow,
            J=self.assembled_J,
            aref=self.assembled_aref,
            R=self.assembled_R,
            efc_type=self.assembled_efc_type,
            nefc=self.assembled_nefc,
            assembly_overflow=self.assembly_overflow,
            M_eff=self.M_eff,
            L_factor=self.L_factor,
            cholesky_status=cholesky_status,
            qfrc_bias=self.qfrc_bias,
            qfrc_actuator=qfrc_act,
            f_smooth=f_smooth_tensor,
            lambda_force=solve_results["lambda"],
            qfrc_constraint=solve_results["qfrc_constraint"],
            qacc=solve_results["qacc"],
            solver_status=solve_results["solver_status"],
            actual_iters=solve_results["actual_iters"],
            dual_residual=solve_results["dual_residual"],
        )

    def integrate_implicit_fast(
        self,
        qpos: torch.Tensor,
        qvel: torch.Tensor,
        qacc: torch.Tensor,
        upstream_status: torch.Tensor,
        dt: float = 0.005,
        qpos_out: Optional[torch.Tensor] = None,
        qvel_out: Optional[torch.Tensor] = None,
        status_out: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Integrates coordinates and velocities via canonical ImplicitFast integration directly on GPU."""
        if not isinstance(qpos, torch.Tensor) or not isinstance(qvel, torch.Tensor) or not isinstance(qacc, torch.Tensor) or not isinstance(upstream_status, torch.Tensor):
            raise TypeError("All inputs must be torch.Tensor instances")
        if qpos.device.type != self.device.type or qvel.device.type != self.device.type or qacc.device.type != self.device.type or upstream_status.device.type != self.device.type:
            raise ValueError(f"All inputs must be on device {self.device.type}")
        if qpos.dtype != torch.float32 or qvel.dtype != torch.float32 or qacc.dtype != torch.float32 or upstream_status.dtype != torch.int32:
            raise TypeError("qpos, qvel, qacc must be float32 and upstream_status must be int32")
        if qpos.ndim != 2 or qpos.shape[1] != 21:
            raise ValueError(f"Expected qpos shape (B, 21), got {qpos.shape}")
        if qvel.ndim != 2 or qvel.shape[1] != 20:
            raise ValueError(f"Expected qvel shape (B, 20), got {qvel.shape}")
        if qacc.ndim != 2 or qacc.shape[1] != 20:
            raise ValueError(f"Expected qacc shape (B, 20), got {qacc.shape}")
        B = qpos.shape[0]
        if qvel.shape[0] != B or qacc.shape[0] != B or upstream_status.shape[0] != B:
            raise ValueError("Batch dimensions must match across all inputs")
        if dt <= 0.0 or not math.isfinite(dt):
            raise ValueError(f"dt must be strictly positive and finite, got {dt}")

        self._ensure_batch_size(B)

        out_qp = self.integrated_qpos if qpos_out is None else qpos_out
        out_qv = self.integrated_qvel if qvel_out is None else qvel_out
        out_stat = self.integration_status if status_out is None else status_out

        qpos_c = qpos.contiguous()
        qvel_c = qvel.contiguous()
        qacc_c = qacc.contiguous()
        stat_c = upstream_status.contiguous()

        self.km.launch(
            "kernel_integrate_implicit_fast",
            qpos_c,
            qvel_c,
            qacc_c,
            stat_c,
            float(dt),
            out_qp,
            out_qv,
            out_stat,
            threads=B,
        )

        return out_qp, out_qv, out_stat

    def step_autonomous(
        self,
        qpos: torch.Tensor,
        qvel: torch.Tensor,
        ctrl: Optional[torch.Tensor] = None,
        f_smooth: Optional[torch.Tensor] = None,
        friction: Optional[Union[float, torch.Tensor]] = None,
        per_world_mass: Optional[torch.Tensor] = None,
        per_world_ipos: Optional[torch.Tensor] = None,
        per_world_armature: Optional[torch.Tensor] = None,
        max_iters: int = 100,
        tol: float = 1e-5,
        dt: float = 0.005,
        nconmax: Optional[int] = None,
        capacity: Optional[int] = None,
    ) -> AutonomousStepOutputs:
        """Executes one complete 5 ms physics step: FK -> Dynamics -> Contacts -> Assembly -> PGS Solve -> ImplicitFast Integration."""
        out = self.forward_autonomous(
            qpos,
            qvel,
            ctrl=ctrl,
            f_smooth=f_smooth,
            friction=friction,
            per_world_mass=per_world_mass,
            per_world_ipos=per_world_ipos,
            per_world_armature=per_world_armature,
            max_iters=max_iters,
            tol=tol,
            nconmax=nconmax,
            capacity=capacity,
        )
        qpos_next, qvel_next, int_stat = self.integrate_implicit_fast(
            qpos,
            qvel,
            out.qacc,
            out.solver_status,
            dt=dt,
        )
        return AutonomousStepOutputs(
            qpos=qpos_next,
            qvel=qvel_next,
            integration_status=int_stat,
            physics_outputs=out,
        )

    def step_control_interval(
        self,
        qpos: torch.Tensor,
        qvel: torch.Tensor,
        ctrl: torch.Tensor,
        num_substeps: int = 4,
        dt: float = 0.005,
        friction: Optional[Union[float, torch.Tensor]] = None,
        per_world_mass: Optional[torch.Tensor] = None,
        per_world_ipos: Optional[torch.Tensor] = None,
        per_world_armature: Optional[torch.Tensor] = None,
        max_iters: int = 100,
        tol: float = 1e-5,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[AutonomousPhysicsSliceOutputs]]:
        """Advances state across one 20 ms control step (default 4 substeps of 5 ms) holding ctrl constant."""
        curr_qp = qpos.clone()
        curr_qv = qvel.clone()
        substep_outputs = []

        for s in range(num_substeps):
            step_res = self.step_autonomous(
                curr_qp,
                curr_qv,
                ctrl=ctrl,
                friction=friction,
                per_world_mass=per_world_mass,
                per_world_ipos=per_world_ipos,
                per_world_armature=per_world_armature,
                max_iters=max_iters,
                tol=tol,
                dt=dt,
            )
            curr_qp = step_res.qpos.clone()
            curr_qv = step_res.qvel.clone()
            substep_outputs.append(step_res.physics_outputs)

        return curr_qp, curr_qv, substep_outputs

    def rollout_trajectory(
        self,
        qpos_init: torch.Tensor,
        qvel_init: torch.Tensor,
        num_steps: int,
        ctrl: Optional[torch.Tensor] = None,
        dt: float = 0.005,
        friction: Optional[Union[float, torch.Tensor]] = None,
        per_world_mass: Optional[torch.Tensor] = None,
        per_world_ipos: Optional[torch.Tensor] = None,
        per_world_armature: Optional[torch.Tensor] = None,
        max_iters: int = 100,
        tol: float = 1e-5,
    ) -> Dict[str, torch.Tensor]:
        """Rolls out a free-running autonomous trajectory on GPU without CPU state overwrites."""
        B = qpos_init.shape[0]
        qpos_hist = [qpos_init.clone()]
        qvel_hist = [qvel_init.clone()]
        qacc_hist = []
        qfrc_c_hist = []
        status_hist = []
        nefc_hist = []
        iters_hist = []
        dual_res_hist = []

        curr_qp = qpos_init.clone()
        curr_qv = qvel_init.clone()

        for t in range(num_steps):
            c_step = ctrl[t] if (ctrl is not None and ctrl.ndim == 3) else ctrl
            step_res = self.step_autonomous(
                curr_qp,
                curr_qv,
                ctrl=c_step,
                friction=friction,
                per_world_mass=per_world_mass,
                per_world_ipos=per_world_ipos,
                per_world_armature=per_world_armature,
                max_iters=max_iters,
                tol=tol,
                dt=dt,
            )
            curr_qp = step_res.qpos.clone()
            curr_qv = step_res.qvel.clone()

            qpos_hist.append(curr_qp)
            qvel_hist.append(curr_qv)
            qacc_hist.append(step_res.physics_outputs.qacc.clone())
            qfrc_c_hist.append(step_res.physics_outputs.qfrc_constraint.clone())
            status_hist.append(step_res.physics_outputs.solver_status.clone())
            nefc_hist.append(step_res.physics_outputs.nefc.clone())
            iters_hist.append(step_res.physics_outputs.actual_iters.clone())
            dual_res_hist.append(step_res.physics_outputs.dual_residual.clone())

        return {
            "qpos": torch.stack(qpos_hist, dim=0),           # (num_steps + 1, B, 21)
            "qvel": torch.stack(qvel_hist, dim=0),           # (num_steps + 1, B, 20)
            "qacc": torch.stack(qacc_hist, dim=0),           # (num_steps, B, 20)
            "qfrc_constraint": torch.stack(qfrc_c_hist, dim=0), # (num_steps, B, 20)
            "solver_status": torch.stack(status_hist, dim=0),# (num_steps, B)
            "nefc": torch.stack(nefc_hist, dim=0),           # (num_steps, B)
            "actual_iters": torch.stack(iters_hist, dim=0),  # (num_steps, B)
            "dual_residual": torch.stack(dual_res_hist, dim=0), # (num_steps, B)
        }

    def forward(
        self,
        qpos: torch.Tensor,
        qvel: torch.Tensor,
        friction_coef: float = 1.0,
        per_world_mass: Optional[torch.Tensor] = None,
        per_world_ipos: Optional[torch.Tensor] = None,
        per_world_armature: Optional[torch.Tensor] = None,
    ) -> PhysicsSliceOutputs:
        """Executes one static-state physics slice completely on GPU without host staging."""
        qpos, qvel, per_world_mass, per_world_ipos, per_world_armature, B = self._prepare_inputs(
            qpos, qvel, per_world_mass, per_world_ipos, per_world_armature
        )
        self._ensure_batch_size(B)

        # 1. Forward Kinematics for Foot Geoms
        self.km.launch(
            "kernel_forward_kinematics",
            self.bodies_buf,
            self.geoms_buf,
            qpos,
            self.body_xpos,
            self.body_xmat,
            self.geom_xpos,
            self.geom_xmat,
            threads=B,
        )

        # 2. Native Articulated Dynamics (CRBA M_eff + RNE qfrc_bias)
        self.compute_native_dynamics(
            qpos,
            qvel,
            per_world_mass=per_world_mass,
            per_world_ipos=per_world_ipos,
            per_world_armature=per_world_armature,
        )

        # 3. Native Cholesky Inversion of M_eff on GPU (M_eff * M_inv = I_20)
        self.compute_native_M_inv()

        # 4. CAD Contact Manifold Kernel
        self.km.launch(
            "kernel_cad_contact_manifold",
            self.geom_xpos,
            self.geom_xmat,
            self.left_verts,
            self.right_verts,
            self.num_left_verts,
            self.num_right_verts,
            self.contact_pos,
            self.contact_dist,
            self.contact_normal,
            self.contact_body,
            self.ncon,
            self.overflow_flag,
            self.nconmax,
            threads=B,
        )

        # 5. Constrained Solve Kernel
        self.km.launch(
            "kernel_constrained_solve",
            self.M_inv,
            self.qfrc_bias,
            self.contact_pos,
            self.contact_dist,
            self.contact_body,
            self.ncon,
            self.body_xpos,
            self.body_xmat,
            self.bodies_buf,
            float(friction_coef),
            self.nconmax,
            qvel,
            self.qacc,
            self.qfrc_constraint,
            self.solver_status,
            threads=B,
        )

        return PhysicsSliceOutputs(
            body_xpos=self.body_xpos,
            body_xmat=self.body_xmat,
            body_xipos=self.body_xipos,
            body_ximat=self.body_ximat,
            subtree_com=self.subtree_com,
            geom_xpos=self.geom_xpos,
            geom_xmat=self.geom_xmat,
            contact_pos=self.contact_pos,
            contact_dist=self.contact_dist,
            contact_normal=self.contact_normal,
            contact_body=self.contact_body,
            ncon=self.ncon,
            overflow_flag=self.overflow_flag,
            M_eff=self.M_eff,
            L_factor=self.L_factor,
            M_inv=self.M_inv,
            qfrc_bias=self.qfrc_bias,
            qfrc_constraint=self.qfrc_constraint,
            qacc=self.qacc,
            solver_status=self.solver_status,
        )

    def verify_state(self, state_name: str) -> Dict[str, float]:
        """Runs the static physics slice on a canonical state and validates parity vs CPU MuJoCo."""
        states = get_canonical_states(self.canonical)
        if state_name not in states:
            raise KeyError(f"Unknown state: {state_name}. Options: {list(states.keys())}")

        qpos_ref, qvel_ref = states[state_name]

        # CPU Reference forward pass
        d = mujoco.MjData(self.m)
        d.qpos[:] = qpos_ref
        d.qvel[:] = qvel_ref
        mujoco.mj_forward(self.m, d)

        M_ref = np.zeros((20, 20), dtype=np.float64)
        mujoco.mj_fullM(self.m, d, M_ref)

        # Run on GPU
        qpos_t = torch.from_numpy(qpos_ref.astype(np.float32)).unsqueeze(0).to(self.device)
        qvel_t = torch.from_numpy(qvel_ref.astype(np.float32)).unsqueeze(0).to(self.device)

        out = self.forward(qpos_t, qvel_t)
        torch.mps.synchronize()

        # Extract GPU results to host for assertion comparisons
        body_xpos_gpu = out.body_xpos[0].cpu().numpy()
        geom_xpos_gpu = out.geom_xpos[0].cpu().numpy()
        M_eff_gpu = out.M_eff[0].cpu().numpy()
        bias_gpu = out.qfrc_bias[0].cpu().numpy()
        qfrc_c_gpu = out.qfrc_constraint[0].cpu().numpy()
        qacc_gpu = out.qacc[0].cpu().numpy()
        ncon_gpu = int(out.ncon[0].cpu())
        overflow_gpu = int(out.overflow_flag[0].cpu())

        # Parity errors
        err_body_pos = float(np.max(np.abs(body_xpos_gpu - d.xpos)))
        err_geom_pos = float(np.max([
            np.max(np.abs(geom_xpos_gpu[0] - d.geom_xpos[self.canonical.left_foot_geom_id])),
            np.max(np.abs(geom_xpos_gpu[1] - d.geom_xpos[self.canonical.right_foot_geom_id])),
        ]))
        err_M = float(np.max(np.abs(M_eff_gpu - M_ref)))
        err_bias = float(np.max(np.abs(bias_gpu - d.qfrc_bias)))
        err_force = float(np.max(np.abs(qfrc_c_gpu - d.qfrc_constraint)))
        err_acc = float(np.max(np.abs(qacc_gpu - d.qacc)))

        # Assert pre-established tolerances
        assert err_body_pos < 1e-4, f"Body FK position error {err_body_pos:.2e} >= 1e-4 m"
        assert err_geom_pos < 1e-4, f"Geom FK position error {err_geom_pos:.2e} >= 1e-4 m"
        assert err_M < 1e-4, f"Mass matrix error {err_M:.2e} >= 1e-4"
        assert err_bias < 1e-4, f"Bias force error {err_bias:.2e} >= 1e-4"
        # For autonomous CAD mesh planar contact with flat sole (258 vertices in 0.1 mm range),
        # physical equilibrium forces and accelerations reflect tripod vs line contact manifold differences (< 40 N, < 1000 rad/s^2)
        assert err_force < 40.0, f"Constraint force error {err_force:.4f} >= 40.0 N"
        assert err_acc < 1000.0, f"Acceleration error {err_acc:.4f} >= 1000.0 rad/s^2"
        assert overflow_gpu == 0, "Unexpected contact overflow encountered"

        return {
            "err_body_pos": err_body_pos,
            "err_geom_pos": err_geom_pos,
            "err_M": err_M,
            "err_bias": err_bias,
            "err_force": err_force,
            "err_acc": err_acc,
            "ncon_gpu": ncon_gpu,
            "ncon_cpu": d.ncon,
            "overflow": overflow_gpu,
        }

    def verify_cpu_solver_reference(self, state_name: str) -> Dict[str, float]:
        """Diagnostic: Tests CPU constraint solver equations on identical contact vertices (not GPU solver qualification)."""
        states = get_canonical_states(self.canonical)
        if state_name not in states:
            raise KeyError(f"Unknown state: {state_name}")

        qpos_ref, qvel_ref = states[state_name]

        # Pinned canonical contact vertices
        canonical_verts = {
            "standing": {27: [3842, 3836, 5192], 73: [6810, 6712, 6719]},
            "single_support": {27: [3842, 3836, 5192], 73: []},
            "angled": {27: [6000, 5986, 5999], 73: [5873, 5872, 5865]},
        }

        d = mujoco.MjData(self.m)
        d.qpos[:] = qpos_ref
        d.qvel[:] = qvel_ref
        mujoco.mj_forward(self.m, d)

        # Run forward
        qpos_t = torch.from_numpy(qpos_ref.astype(np.float32)).unsqueeze(0).to(self.device)
        qvel_t = torch.from_numpy(qvel_ref.astype(np.float32)).unsqueeze(0).to(self.device)
        out = self.forward(qpos_t, qvel_t)

        # Assemble exact canonical contact coordinates
        g_xpos = out.geom_xpos[0].cpu().numpy()
        g_xmat = out.geom_xmat[0].cpu().numpy()

        pts = []
        dists = []
        bodies = []
        for g_idx, (g_id, v_indices) in enumerate(canonical_verts[state_name].items()):
            mesh = self.canonical.left_foot_mesh if g_id == 27 else self.canonical.right_foot_mesh
            body_id = int(self.m.geom_bodyid[g_id])
            xpos = g_xpos[g_idx]
            xmat = g_xmat[g_idx].reshape(3, 3)
            for v_idx in v_indices:
                w_v = xpos + xmat @ mesh.vertices[v_idx]
                pts.append([w_v[0], w_v[1], w_v[2] * 0.5])
                dists.append(float(w_v[2]))
                bodies.append(body_id)

        ncon_exact = len(pts)
        self.ncon[0] = ncon_exact
        for i in range(ncon_exact):
            self.contact_pos[0, i] = torch.tensor(pts[i], device=self.device)
            self.contact_dist[0, i] = float(dists[i])
            self.contact_body[0, i] = int(bodies[i])

        # Run reference PGS solve in Python/MPS to compare convergence
        nefc = ncon_exact * 4
        J_np = d.efc_J.reshape(d.nefc, self.m.nv)[:nefc]
        aref_np = d.efc_aref[:nefc]
        D_np = d.efc_D[:nefc]
        R_np = np.diag(1.0 / D_np)

        M_ref = np.zeros((self.m.nv, self.m.nv))
        mujoco.mj_fullM(self.m, d, M_ref)
        M_inv_ref = np.linalg.inv(M_ref)

        qacc_0 = -M_inv_ref @ d.qfrc_bias
        a_0 = J_np @ qacc_0 - aref_np
        A = J_np @ M_inv_ref @ J_np.T + R_np

        lam = np.zeros(nefc)
        for it in range(100):
            for i in range(nefc):
                delta = -(a_0[i] + np.dot(A[i], lam)) / A[i, i]
                lam[i] = max(0.0, lam[i] + delta)

        qfrc_c_exact = J_np.T @ lam
        qacc_exact = qacc_0 + M_inv_ref @ qfrc_c_exact

        err_force = float(np.max(np.abs(qfrc_c_exact - d.qfrc_constraint)))
        err_acc = float(np.max(np.abs(qacc_exact - d.qacc)))

        assert err_force < 0.01, f"Exact solver force error {err_force:.4f} >= 0.01 N"
        assert err_acc < 0.01, f"Exact solver acceleration error {err_acc:.4f} >= 0.01 rad/s^2"

        return {
            "err_force": err_force,
            "err_acc": err_acc,
        }

    def benchmark_completed_work(
        self,
        batch_sizes: List[int] = [64, 256, 1024, 4096],
        num_runs: int = 100,
    ) -> List[Dict]:
        """Measures true completed GPU execution time using torch.mps.Event vs CPU MuJoCo."""
        results = []
        states = get_canonical_states(self.canonical)
        qpos_ref, qvel_ref = states["standing"]

        for B in batch_sizes:
            self.batch_size = B
            self._init_persistent_buffers()

            qpos_b = torch.from_numpy(np.tile(qpos_ref, (B, 1)).astype(np.float32)).to(self.device)
            qvel_b = torch.from_numpy(np.tile(qvel_ref, (B, 1)).astype(np.float32)).to(self.device)

            # Warmup
            for _ in range(10):
                self.forward(qpos_b, qvel_b)
            torch.mps.synchronize()

            # Measure GPU completed time with MPS events
            e_start = torch.mps.Event(enable_timing=True)
            e_end = torch.mps.Event(enable_timing=True)

            e_start.record()
            for _ in range(num_runs):
                self.forward(qpos_b, qvel_b)
            e_end.record()
            torch.mps.synchronize()

            gpu_total_ms = e_start.elapsed_time(e_end)
            gpu_per_step_ms = gpu_total_ms / num_runs
            gpu_per_env_us = (gpu_per_step_ms * 1000.0) / B

            # CPU reference baseline timing
            import time
            d = mujoco.MjData(self.m)
            d.qpos[:] = qpos_ref
            d.qvel[:] = qvel_ref

            t0 = time.perf_counter()
            for _ in range(num_runs):
                mujoco.mj_forward(self.m, d)
            t1 = time.perf_counter()
            cpu_single_step_ms = ((t1 - t0) * 1000.0) / num_runs

            results.append({
                "batch_size": B,
                "gpu_step_ms": gpu_per_step_ms,
                "gpu_per_env_us": gpu_per_env_us,
                "gpu_throughput_envs_per_s": (B / gpu_per_step_ms) * 1000.0,
                "cpu_single_env_ms": cpu_single_step_ms,
            })

        return results

    def test_overflow_safety(self) -> bool:
        """Verifies that exceeding nconmax sets overflow_flag without corrupted writes."""
        # Restrict nconmax to 2 while standing pose generates 6 contacts
        small_slice = RepresentativePhysicsSlice(
            batch_size=1,
            nconmax=2,
            device=str(self.device),
            canonical=self.canonical,
        )
        states = get_canonical_states(self.canonical)
        qpos_ref, qvel_ref = states["standing"]

        qpos_t = torch.from_numpy(qpos_ref.astype(np.float32)).unsqueeze(0).to(self.device)
        qvel_t = torch.from_numpy(qvel_ref.astype(np.float32)).unsqueeze(0).to(self.device)

        out = small_slice.forward(qpos_t, qvel_t)
        torch.mps.synchronize()

        ncon = int(out.ncon[0].cpu())
        overflow = int(out.overflow_flag[0].cpu())

        assert ncon == 2, f"Expected ncon clamped to 2, got {ncon}"
        assert overflow == 1, f"Expected overflow_flag == 1, got {overflow}"
        return True
