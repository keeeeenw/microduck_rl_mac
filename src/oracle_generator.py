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

    def _eval_state(self, m: mujoco.MjModel, qpos: np.ndarray, qvel: np.ndarray) -> Dict[str, Any]:
        """Evaluates a single state on CPU MuJoCo to extract all target quantities."""
        d = mujoco.MjData(m)
        d.qpos[:] = qpos
        d.qvel[:] = qvel
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

        return {
            "qpos": qpos.copy(),
            "qvel": qvel.copy(),
            # Kinematics
            "xpos": d.xpos.copy(),        # (nbody, 3)
            "xquat": d.xquat.copy(),      # (nbody, 4)
            "xmat": d.xmat.reshape(-1, 9).copy(), # (nbody, 9) row-major
            "xipos": d.xipos.copy(),      # (nbody, 3) CoM of each body
            "ximat": d.ximat.reshape(-1, 9).copy(), # (nbody, 9) CoM orientation row-major
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
        }

    def generate_corpus(self) -> Dict[str, Dict[str, Any]]:
        """Constructs the comprehensive oracle corpus across all asserted scenarios."""
        m = self.m
        corpus = {}

        # 1. Standing zero velocity
        qpos_std = m.key_qpos[0].copy()
        qvel_zero = np.zeros(m.nv, dtype=np.float64)
        corpus["standing_zero_vel"] = self._eval_state(m, qpos_std, qvel_zero)

        # 2. Standing moving velocity
        rng = np.random.default_rng(42)
        qvel_std_mov = rng.uniform(-0.3, 0.3, size=m.nv)
        corpus["standing_moving_vel"] = self._eval_state(m, qpos_std, qvel_std_mov)

        # 3. Single support zero velocity (lift right leg)
        qpos_single = m.key_qpos[0].copy()
        j_r_pitch = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "robot/right_hip_pitch")
        j_r_knee = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "robot/right_knee")
        qpos_single[m.jnt_qposadr[j_r_pitch]] = -1.2
        qpos_single[m.jnt_qposadr[j_r_knee]] = 1.5
        corpus["single_support_zero_vel"] = self._eval_state(m, qpos_single, qvel_zero)

        # 4. Single support moving velocity
        qvel_single = np.full(m.nv, 0.08, dtype=np.float64)
        corpus["single_support_moving_vel"] = self._eval_state(m, qpos_single, qvel_single)

        # 5. Tilted landing (15 deg base roll)
        qpos_tilted = m.key_qpos[0].copy()
        angle = np.radians(15.0)
        qpos_tilted[3] = np.cos(angle / 2.0)
        qpos_tilted[4] = np.sin(angle / 2.0)
        qpos_tilted[5] = 0.0
        qpos_tilted[6] = 0.0
        corpus["tilted_landing"] = self._eval_state(m, qpos_tilted, qvel_zero)

        # 6. Airborne (height z = 0.35m, moving linear velocity)
        qpos_air = m.key_qpos[0].copy()
        qpos_air[2] = 0.35
        qvel_air = np.zeros(m.nv, dtype=np.float64)
        qvel_air[0] = 0.5   # vx
        qvel_air[2] = -0.2  # vz falling
        corpus["airborne"] = self._eval_state(m, qpos_air, qvel_air)

        # 7. Nonzero base angular velocity (tumbling / turning in air)
        qpos_tumbling = m.key_qpos[0].copy()
        qpos_tumbling[2] = 0.25
        qvel_tumbling = np.zeros(m.nv, dtype=np.float64)
        qvel_tumbling[3] = 1.2   # wx
        qvel_tumbling[4] = -0.8  # wy
        qvel_tumbling[5] = 2.0   # wz
        corpus["nonzero_base_angvel"] = self._eval_state(m, qpos_tumbling, qvel_tumbling)

        # 8. Near contact separation (sole 1.0 mm above ground)
        qpos_sep = m.key_qpos[0].copy()
        qpos_sep[2] += 0.015 # slightly lifted so feet are near but separated
        corpus["near_contact_separation"] = self._eval_state(m, qpos_sep, qvel_zero)

        # 9. Contact onset (feet penetrating ground plane)
        qpos_onset = m.key_qpos[0].copy()
        qpos_onset[2] -= 0.005 # pressed into ground
        corpus["contact_onset"] = self._eval_state(m, qpos_onset, qvel_zero)

        # 10. Randomized model: perturbed trunk CoM and armature
        m_rand = mujoco.MjModel.from_xml_path(str(self.xml_path))
        base_id = mujoco.mj_name2id(m_rand, mujoco.mjtObj.mjOBJ_BODY, "robot/trunk_base")
        m_rand.body_ipos[base_id] += np.array([0.005, -0.004, 0.003])
        m_rand.body_mass[base_id] *= 1.05
        m_rand.dof_armature[6:] *= 1.10 # +10% armature on actuated joints
        corpus["randomized_model_standing"] = self._eval_state(m_rand, qpos_std, qvel_std_mov)

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
