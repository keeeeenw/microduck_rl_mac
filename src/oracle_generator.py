"""Independent Deterministic CPU MuJoCo Oracle Generator for MicroDuck.

Generates ground-truth kinematics, mass matrix (CRBA), bias forces (RNE),
and reference Cholesky linear solves independently of any Metal code.
Saves input states, model parameters, and expected outputs for exact verification.
"""

from pathlib import Path
from typing import Dict, Any, List
import json
import numpy as np
import mujoco

WORKSPACE = Path("/Users/zixiao/workspace/microduck")
CANONICAL_XML = WORKSPACE / "mlx-assessment" / "results" / "microduck_canonical_flat.xml"
EXTERNAL_DIR = Path("/Volumes/T7/ChatGPOExtension/unified-metal/oracle_corpus")
LOCAL_DIR = WORKSPACE / "unified-metal" / "corpus"


class OracleGenerator:
    """Generates independent CPU reference states and outputs."""

    def __init__(self, xml_path: Path = CANONICAL_XML):
        self.xml_path = xml_path
        self.m = mujoco.MjModel.from_xml_path(str(xml_path))
        self.d = mujoco.MjData(self.m)

    def _eval_state(
        self,
        m: mujoco.MjModel,
        qpos: np.ndarray,
        qvel: np.ndarray,
        qfrc_applied: np.ndarray = None,
        ctrl: np.ndarray = None,
    ) -> Dict[str, Any]:
        """Evaluates a single state on CPU MuJoCo to extract all target quantities."""
        d = mujoco.MjData(m)
        d.qpos[:] = qpos
        d.qvel[:] = qvel
        if qfrc_applied is not None:
            d.qfrc_applied[:] = qfrc_applied
        if ctrl is not None:
            d.ctrl[:] = ctrl
        mujoco.mj_forward(m, d)

        # Full mass matrix including configured armature
        M = np.zeros((m.nv, m.nv), dtype=np.float64)
        mujoco.mj_fullM(m, d, M)

        # Cholesky factorization of M
        L = np.linalg.cholesky(M)

        # Reference solves: M x = b
        b_bias = d.qfrc_bias.copy()
        x_bias = np.linalg.solve(M, b_bias)
        x_inv = np.linalg.inv(M)

        # Deterministic multi-RHS test fixture (20 x 4)
        rng = np.random.default_rng(12345)
        B_multi = rng.standard_normal((m.nv, 4))
        X_multi = np.linalg.solve(M, B_multi)

        # Check Cholesky residual
        res_L = np.max(np.abs(L @ L.T - M))
        assert res_L < 1e-12, f"CPU Cholesky residual {res_L} is non-trivial"

        cond_M = float(np.linalg.cond(M))

        # Constraint and contact manifold extraction
        nefc = int(d.nefc)
        if nefc > 0:
            efc_J = d.efc_J[:nefc * m.nv].reshape(nefc, m.nv).copy()
            efc_aref = d.efc_aref[:nefc].copy()
            efc_D = d.efc_D[:nefc].copy()
            efc_R = d.efc_R[:nefc].copy()
            efc_type = d.efc_type[:nefc].copy()
            efc_id = d.efc_id[:nefc].copy()
            efc_force = d.efc_force[:nefc].copy()
            # Verify that only pyramidal contact constraints (type 6) are present in this gate
            assert np.all(efc_type == 6), f"Unsupported constraint types found: {set(efc_type)}"
            # Verify reciprocal consistency of R and D
            assert np.max(np.abs(efc_R - 1.0 / efc_D)) < 1e-5, "R and 1/D inconsistency"
        else:
            efc_J = np.zeros((0, m.nv), dtype=np.float64)
            efc_aref = np.zeros((0,), dtype=np.float64)
            efc_D = np.zeros((0,), dtype=np.float64)
            efc_R = np.zeros((0,), dtype=np.float64)
            efc_type = np.zeros((0,), dtype=np.int32)
            efc_id = np.zeros((0,), dtype=np.int32)
            efc_force = np.zeros((0,), dtype=np.float64)

        # Solver configuration metadata
        solver_type = int(m.opt.solver)           # mjSOL_NEWTON = 2
        cone_type = int(m.opt.cone)               # mjCONE_PYRAMIDAL = 0
        solver_iterations = int(m.opt.iterations) # 10
        solver_ls_iterations = int(m.opt.ls_iterations) # 20
        solver_tolerance = float(m.opt.tolerance)

        ncon = int(d.ncon)
        if ncon > 0:
            contact_geom1 = np.array([d.contact[i].geom1 for i in range(ncon)], dtype=np.int32)
            contact_geom2 = np.array([d.contact[i].geom2 for i in range(ncon)], dtype=np.int32)
            contact_pos = np.array([d.contact[i].pos for i in range(ncon)], dtype=np.float64).reshape(-1, 3)
            contact_dist = np.array([d.contact[i].dist for i in range(ncon)], dtype=np.float64)
            contact_frame = np.array([d.contact[i].frame for i in range(ncon)], dtype=np.float64).reshape(-1, 9)
            contact_dim = np.array([d.contact[i].dim for i in range(ncon)], dtype=np.int32)
            contact_friction = np.array([d.contact[i].friction for i in range(ncon)], dtype=np.float64).reshape(-1, 5)
            contact_solref = np.array([d.contact[i].solref for i in range(ncon)], dtype=np.float64).reshape(-1, 2)
            contact_solimp = np.array([d.contact[i].solimp for i in range(ncon)], dtype=np.float64).reshape(-1, 5)
        else:
            contact_geom1 = np.zeros((0,), dtype=np.int32)
            contact_geom2 = np.zeros((0,), dtype=np.int32)
            contact_pos = np.zeros((0, 3), dtype=np.float64)
            contact_dist = np.zeros((0,), dtype=np.float64)
            contact_frame = np.zeros((0, 9), dtype=np.float64)
            contact_dim = np.zeros((0,), dtype=np.int32)
            contact_friction = np.zeros((0, 5), dtype=np.float64)
            contact_solref = np.zeros((0, 2), dtype=np.float64)
            contact_solimp = np.zeros((0, 5), dtype=np.float64)

        efc_pos = d.efc_pos[:nefc].copy() if nefc > 0 else np.zeros((0,), dtype=np.float64)
        efc_vel = d.efc_vel[:nefc].copy() if nefc > 0 else np.zeros((0,), dtype=np.float64)
        efc_margin = d.efc_margin[:nefc].copy() if nefc > 0 else np.zeros((0,), dtype=np.float64)
        efc_KBIP = d.efc_KBIP[:nefc].copy() if nefc > 0 else np.zeros((0, 4), dtype=np.float64)

        return {
            "qpos": qpos.copy(),
            "qvel": qvel.copy(),
            # Model parameters (self-contained domain randomization inputs)
            "per_world_mass": m.body_mass.copy(),
            "per_world_ipos": m.body_ipos.copy(),
            "per_world_armature": m.dof_armature.copy(),
            "body_invweight0": m.body_invweight0.copy(),
            "geom_rbound": m.geom_rbound.copy(),
            "cond_M": np.array(cond_M, dtype=np.float64),
            # Kinematics
            "xpos": d.xpos.copy(),        # (nbody, 3)
            "xquat": d.xquat.copy(),      # (nbody, 4)
            "xmat": d.xmat.reshape(-1, 9).copy(), # (nbody, 9) row-major
            "xipos": d.xipos.copy(),      # (nbody, 3) CoM of each body
            "ximat": d.ximat.reshape(-1, 9).copy(), # (nbody, 9) CoM orientation row-major
            "geom_xpos": d.geom_xpos.copy(),
            "geom_xmat": d.geom_xmat.reshape(-1, 9).copy(),
            "subtree_com": d.subtree_com.copy(), # (nbody, 3)
            # Dynamics
            "M": M,                       # (20, 20)
            "qfrc_bias": d.qfrc_bias.copy(), # (20,)
            "cvel": d.cvel.copy(),        # (nbody, 6) spatial velocity
            "cdof": d.cdof.copy(),        # (nv, 6) spatial motion dof
            "cinert": d.cinert.copy(),    # (nbody, 10) spatial inertia
            "crb": d.crb.copy(),          # (nbody, 10) composite inertia
            # Factorization & Solves
            "L": L,                       # (20, 20)
            "x_bias": x_bias,             # (20,)
            "x_inv": x_inv,               # (20, 20)
            "B_multi": B_multi,           # (20, 4)
            "X_multi": X_multi,           # (20, 4)
            # Contacts
            "ncon": np.array(ncon, dtype=np.int32),
            "contact_geom1": contact_geom1,
            "contact_geom2": contact_geom2,
            "contact_pos": contact_pos,
            "contact_dist": contact_dist,
            "contact_frame": contact_frame,
            "contact_dim": contact_dim,
            "contact_friction": contact_friction,
            "contact_solref": contact_solref,
            "contact_solimp": contact_solimp,
            # Constraints & Solvers
            "nefc": np.array(nefc, dtype=np.int32),
            "efc_J": efc_J,
            "efc_aref": efc_aref,
            "efc_D": efc_D,
            "efc_R": efc_R,
            "efc_type": efc_type,
            "efc_id": efc_id,
            "efc_force": efc_force,
            "efc_pos": efc_pos,
            "efc_vel": efc_vel,
            "efc_margin": efc_margin,
            "efc_KBIP": efc_KBIP,
            "qfrc_smooth": d.qfrc_smooth.copy(),
            "qacc_smooth": d.qacc_smooth.copy(),
            "qfrc_constraint": d.qfrc_constraint.copy(),
            "qacc": d.qacc.copy(),
            "qfrc_applied": d.qfrc_applied.copy(),
            "solver_type": np.array(solver_type, dtype=np.int32),
            "cone_type": np.array(cone_type, dtype=np.int32),
            "solver_iterations": np.array(solver_iterations, dtype=np.int32),
            "solver_ls_iterations": np.array(solver_ls_iterations, dtype=np.int32),
            "solver_tolerance": np.array(solver_tolerance, dtype=np.float64),
        }

    def generate_corpus(self) -> Dict[str, Dict[str, Any]]:
        """Constructs the comprehensive oracle corpus across all asserted scenarios."""
        m = self.m
        corpus = {}

        # ---------------------------------------------------------------------
        # Group A: Original 15 Scenarios (Labeled Stress / Regression Cases)
        # ---------------------------------------------------------------------
        qpos_std = m.key_qpos[0].copy()
        qvel_zero = np.zeros(m.nv, dtype=np.float64)
        corpus["standing_zero_vel"] = self._eval_state(m, qpos_std, qvel_zero)

        rng = np.random.default_rng(42)
        qvel_std_mov = rng.uniform(-0.3, 0.3, size=m.nv)
        corpus["standing_moving_vel"] = self._eval_state(m, qpos_std, qvel_std_mov)

        qpos_single = m.key_qpos[0].copy()
        j_r_pitch = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "robot/right_hip_pitch")
        j_r_knee = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "robot/right_knee")
        qpos_single[m.jnt_qposadr[j_r_pitch]] = -1.2
        qpos_single[m.jnt_qposadr[j_r_knee]] = 1.5
        corpus["single_support_zero_vel"] = self._eval_state(m, qpos_single, qvel_zero)

        qvel_single = np.full(m.nv, 0.08, dtype=np.float64)
        corpus["single_support_moving_vel"] = self._eval_state(m, qpos_single, qvel_single)

        qpos_tilted = m.key_qpos[0].copy()
        angle = np.radians(15.0)
        qpos_tilted[3] = np.cos(angle / 2.0)
        qpos_tilted[4] = np.sin(angle / 2.0)
        qpos_tilted[5] = 0.0
        qpos_tilted[6] = 0.0
        corpus["tilted_landing"] = self._eval_state(m, qpos_tilted, qvel_zero)

        qpos_air = m.key_qpos[0].copy()
        qpos_air[2] = 0.35
        qvel_air = np.zeros(m.nv, dtype=np.float64)
        qvel_air[0] = 0.5
        qvel_air[2] = -0.2
        corpus["airborne"] = self._eval_state(m, qpos_air, qvel_air)

        qpos_tumbling = m.key_qpos[0].copy()
        qpos_tumbling[2] = 0.25
        qvel_tumbling = np.zeros(m.nv, dtype=np.float64)
        qvel_tumbling[3] = 1.2
        qvel_tumbling[4] = -0.8
        qvel_tumbling[5] = 2.0
        corpus["nonzero_base_angvel"] = self._eval_state(m, qpos_tumbling, qvel_tumbling)

        qpos_sep = m.key_qpos[0].copy()
        qpos_sep[2] += 0.015
        corpus["near_contact_separation"] = self._eval_state(m, qpos_sep, qvel_zero)

        qpos_onset = m.key_qpos[0].copy()
        qpos_onset[2] -= 0.005
        corpus["contact_onset"] = self._eval_state(m, qpos_onset, qvel_zero)

        m_rand = mujoco.MjModel.from_xml_path(str(self.xml_path))
        base_id = mujoco.mj_name2id(m_rand, mujoco.mjtObj.mjOBJ_BODY, "robot/trunk_base")
        m_rand.body_ipos[base_id] += np.array([0.005, -0.004, 0.003])
        m_rand.body_mass[base_id] *= 1.05
        m_rand.dof_armature[6:] *= 1.10
        corpus["randomized_model_standing"] = self._eval_state(m_rand, qpos_std, qvel_std_mov)

        qpos_crouch = m.key_qpos[0].copy()
        j_l_pitch = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "robot/left_hip_pitch")
        j_l_knee = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "robot/left_knee")
        qpos_crouch[m.jnt_qposadr[j_l_pitch]] = -0.7
        qpos_crouch[m.jnt_qposadr[j_r_pitch]] = -0.7
        qpos_crouch[m.jnt_qposadr[j_l_knee]] = 1.1
        qpos_crouch[m.jnt_qposadr[j_r_knee]] = 1.1
        corpus["crouched_pose"] = self._eval_state(m, qpos_crouch, qvel_zero)

        qpos_asym = m.key_qpos[0].copy()
        qpos_asym[m.jnt_qposadr[j_l_pitch]] = 0.4
        qpos_asym[m.jnt_qposadr[j_r_pitch]] = -0.8
        qpos_asym[m.jnt_qposadr[j_l_knee]] = 0.6
        qpos_asym[m.jnt_qposadr[j_r_knee]] = 1.3
        qvel_asym = np.zeros(m.nv, dtype=np.float64)
        qvel_asym[6:12] = [0.5, -0.4, 0.3, -0.2, 0.6, -0.5]
        corpus["asymmetric_pose"] = self._eval_state(m, qpos_asym, qvel_asym)

        qpos_rotmot = m.key_qpos[0].copy()
        qpos_rotmot[2] = 0.28
        cr, sr = np.cos(np.radians(6.0)), np.sin(np.radians(6.0))
        cp, sp = np.cos(np.radians(-4.0)), np.sin(np.radians(-4.0))
        cy, sy = np.cos(np.radians(10.0)), np.sin(np.radians(10.0))
        qpos_rotmot[3] = cr * cp * cy + sr * sp * sy
        qpos_rotmot[4] = sr * cp * cy - cr * sp * sy
        qpos_rotmot[5] = cr * sp * cy + sr * cp * sy
        qpos_rotmot[6] = cr * cp * sy - sr * sp * cy
        qvel_rotmot = np.zeros(m.nv, dtype=np.float64)
        qvel_rotmot[0:3] = [0.35, -0.25, 0.15]
        qvel_rotmot[3:6] = [-0.5, 0.8, -1.0]
        qvel_rotmot[6:14] = 0.15
        corpus["combined_rotation_motion"] = self._eval_state(m, qpos_rotmot, qvel_rotmot)

        m_cond = mujoco.MjModel.from_xml_path(str(self.xml_path))
        m_cond.dof_armature[6:] = 0.0005
        trunk_id = mujoco.mj_name2id(m_cond, mujoco.mjtObj.mjOBJ_BODY, "robot/trunk_base")
        m_cond.body_mass[trunk_id] = 1.25
        corpus["high_condition_mass_matrix"] = self._eval_state(m_cond, qpos_crouch, qvel_zero)

        qfrc_applied_15 = np.zeros(m.nv, dtype=np.float64)
        qfrc_applied_15[0] = 5.0
        qfrc_applied_15[6:10] = [2.0, -1.5, 1.0, -0.5]
        corpus["nonzero_applied_force"] = self._eval_state(
            m, qpos_std, qvel_zero, qfrc_applied=qfrc_applied_15
        )

        # ---------------------------------------------------------------------
        # Group B: Realistic Contact-Boundary & Impedance-Transition Fixtures
        # Measured touching height: z_touch = 0.11718236219874847 m
        # ---------------------------------------------------------------------
        z_touch = 0.11718236219874847

        # B1: Nominal standing realistic (2 mm penetration, saturated imp = 0.95)
        qpos_nom = m.key_qpos[0].copy()
        qpos_nom[2] = z_touch - 0.0020
        corpus["nominal_standing_realistic"] = self._eval_state(m, qpos_nom, qvel_zero)
        assert corpus["nominal_standing_realistic"]["ncon"] == 4

        # B2: Truly separated airborne (2 mm above ground, ncon = 0, nefc = 0)
        qpos_sep2 = m.key_qpos[0].copy()
        qpos_sep2[2] = z_touch + 0.0020
        corpus["boundary_separated_2mm"] = self._eval_state(m, qpos_sep2, qvel_zero)
        assert corpus["boundary_separated_2mm"]["ncon"] == 0

        # B3: Exact contact onset (0.00 mm clearance, touch onset, imp = 0.90)
        qpos_onset0 = m.key_qpos[0].copy()
        qpos_onset0[2] = z_touch
        corpus["boundary_onset_exact"] = self._eval_state(m, qpos_onset0, qvel_zero)
        assert corpus["boundary_onset_exact"]["ncon"] >= 1

        # B4: Shallow penetration inside impedance transition (0.1 mm, imp ~ 0.901)
        qpos_trans1 = m.key_qpos[0].copy()
        qpos_trans1[2] = z_touch - 0.0001
        corpus["impedance_transition_shallow"] = self._eval_state(m, qpos_trans1, qvel_zero)
        assert 0.90 < corpus["impedance_transition_shallow"]["efc_KBIP"][0, 2] < 0.91

        # B5: Midpoint of impedance transition (0.5 mm, imp ~ 0.925)
        qpos_trans2 = m.key_qpos[0].copy()
        qpos_trans2[2] = z_touch - 0.0005
        corpus["impedance_transition_mid"] = self._eval_state(m, qpos_trans2, qvel_zero)
        assert 0.92 < corpus["impedance_transition_mid"]["efc_KBIP"][0, 2] < 0.93

        # B6: Transition edge (1.0 mm, imp = 0.95)
        qpos_trans3 = m.key_qpos[0].copy()
        qpos_trans3[2] = z_touch - 0.0010
        corpus["impedance_transition_edge"] = self._eval_state(m, qpos_trans3, qvel_zero)
        assert np.isclose(corpus["impedance_transition_edge"]["efc_KBIP"][0, 2], 0.95, atol=1e-4)

        # B7: Toe only contact realistic (pitched forward 8 deg, adjusted height)
        qpos_toe = m.key_qpos[0].copy()
        ang_toe = np.radians(8.0)
        qpos_toe[3] = np.cos(ang_toe / 2)
        qpos_toe[5] = np.sin(ang_toe / 2)
        qpos_toe[2] = 0.5 - 0.3802918170417287 - 0.0015 # 1.5 mm toe penetration
        corpus["toe_only_contact_realistic"] = self._eval_state(m, qpos_toe, qvel_zero)
        assert corpus["toe_only_contact_realistic"]["ncon"] == 2

        # B8: Heel only contact realistic (pitched backward 8 deg, adjusted height)
        qpos_heel = m.key_qpos[0].copy()
        ang_heel = np.radians(-8.0)
        qpos_heel[3] = np.cos(ang_heel / 2)
        qpos_heel[5] = np.sin(ang_heel / 2)
        # Measured touching height pitched backward is 0.11775557203169412
        qpos_heel[2] = 0.11775557203169412 - 0.0015
        corpus["heel_only_contact_realistic"] = self._eval_state(m, qpos_heel, qvel_zero)
        assert corpus["heel_only_contact_realistic"]["ncon"] == 2

        # B9: Sliding lateral velocity (nominal standing with lateral velocity)
        qvel_lateral = np.zeros(m.nv, dtype=np.float64)
        qvel_lateral[1] = 0.25 # vy = 0.25 m/s
        qvel_lateral[6:12] = [0.1, -0.1, 0.1, -0.1, 0.1, -0.1]
        corpus["sliding_lateral_velocity"] = self._eval_state(m, qpos_nom, qvel_lateral)

        # B10: Varied friction model (mu = 0.6)
        m_fric = mujoco.MjModel.from_xml_path(str(self.xml_path))
        # Override ground and foot friction
        g_plane = mujoco.mj_name2id(m_fric, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
        g_lfoot = mujoco.mj_name2id(m_fric, mujoco.mjtObj.mjOBJ_GEOM, "robot/left_foot_collision")
        g_rfoot = mujoco.mj_name2id(m_fric, mujoco.mjtObj.mjOBJ_GEOM, "robot/right_foot_collision")
        m_fric.geom_friction[g_plane, 0] = 0.6
        m_fric.geom_friction[g_lfoot, 0] = 0.6
        m_fric.geom_friction[g_rfoot, 0] = 0.6
        corpus["randomized_friction_corpus"] = self._eval_state(m_fric, qpos_nom, qvel_zero)
        assert np.isclose(corpus["randomized_friction_corpus"]["contact_friction"][0, 0], 0.6)

        return corpus

    def save_corpus(self, corpus: Dict[str, Dict[str, Any]]):
        """Saves oracle corpus to external drive (if mounted) and local repository."""
        target_dirs = []
        if EXTERNAL_DIR.parent.exists():
            EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
            target_dirs.append(EXTERNAL_DIR)
        LOCAL_DIR.mkdir(parents=True, exist_ok=True)
        target_dirs.append(LOCAL_DIR)

        meta = {
            "num_scenarios": len(corpus),
            "scenarios": list(corpus.keys()),
            "nv": self.m.nv,
            "nq": self.m.nq,
            "nbody": self.m.nbody,
        }

        for out_dir in target_dirs:
            # Save metadata
            (out_dir / "corpus_metadata.json").write_text(json.dumps(meta, indent=2))
            # Save numpy arrays per scenario
            for name, data in corpus.items():
                np_file = out_dir / f"{name}.npz"
                np.savez_compressed(np_file, **data)
            print(f"Saved {len(corpus)} oracle fixtures to {out_dir}")


if __name__ == "__main__":
    gen = OracleGenerator()
    print("Generating comprehensive CPU MuJoCo oracle corpus...")
    corpus = gen.generate_corpus()
    gen.save_corpus(corpus)
    print("Oracle generation complete.")
