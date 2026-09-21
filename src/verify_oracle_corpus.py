"""Independent Verification of Native Dynamics and Solves vs CPU MuJoCo Oracle Corpus.

Tests all 10 corpus scenarios against candidate Metal kernels:
- Body forward kinematics: positions (xpos) and orientations (xmat)
- Articulated dynamics: CRBA mass matrix (M_eff) and RNE bias forces (qfrc_bias)
- Native Cholesky factorization and linear solves: relative residual ||M x - b|| / (||M|| ||x|| + ||b||)
"""

import sys
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.representative_physics_slice import RepresentativePhysicsSlice
from src.canonical_model_loader import load_canonical_model

CORPUS_DIR = Path("/Users/zixiao/workspace/microduck/unified-metal/corpus")


def run_oracle_verification() -> List[Dict[str, Any]]:
    canonical = load_canonical_model()
    engine = RepresentativePhysicsSlice(batch_size=1, canonical=canonical)

    scenarios = [
        "standing_zero_vel",
        "standing_moving_vel",
        "single_support_zero_vel",
        "single_support_moving_vel",
        "tilted_landing",
        "airborne",
        "nonzero_base_angvel",
        "near_contact_separation",
        "contact_onset",
        "randomized_model_standing",
    ]

    results = []

    print(f"{'Scenario':<28} | {'err_xpos (m)':<12} | {'err_xmat':<10} | {'err_M':<10} | {'err_bias':<10} | {'rel_residual':<12} | {'Status'}")
    print("-" * 105)

    for sc in scenarios:
        npz = np.load(CORPUS_DIR / f"{sc}.npz")
        qpos = torch.from_numpy(npz["qpos"].astype(np.float32)).unsqueeze(0).to(engine.device)
        qvel = torch.from_numpy(npz["qvel"].astype(np.float32)).unsqueeze(0).to(engine.device)

        p_mass = None
        p_ipos = None
        p_arm = None
        if "randomized" in sc:
            base_id = 2
            m_mass = np.copy(canonical.body_mass)
            m_mass[base_id] *= 1.05
            m_ipos = np.copy(canonical.model.body_ipos)
            m_ipos[base_id] += np.array([0.005, -0.004, 0.003])
            m_arm = np.copy(canonical.dof_armature)
            m_arm[6:] *= 1.10
            p_mass = torch.from_numpy(m_mass.astype(np.float32)).unsqueeze(0).to(engine.device)
            p_ipos = torch.from_numpy(m_ipos.astype(np.float32)).unsqueeze(0).to(engine.device)
            p_arm = torch.from_numpy(m_arm.astype(np.float32)).unsqueeze(0).to(engine.device)

        engine.compute_native_dynamics(qpos, qvel, per_world_mass=p_mass, per_world_ipos=p_ipos, per_world_armature=p_arm)
        X_bias, status = engine.compute_native_cholesky_solve(engine.M_eff, engine.qfrc_bias)
        torch.mps.synchronize()

        xpos_gpu = engine.body_xpos[0].cpu().numpy()
        xmat_gpu = engine.body_xmat[0].cpu().numpy()
        M_gpu = engine.M_eff[0].cpu().numpy()
        bias_gpu = engine.qfrc_bias[0].cpu().numpy()
        X_gpu = X_bias[0].cpu().numpy()

        err_xpos = float(np.max(np.abs(xpos_gpu - npz["xpos"])))
        err_xmat = float(np.max(np.abs(xmat_gpu - npz["xmat"])))
        err_M = float(np.max(np.abs(M_gpu - npz["M"])))
        err_bias = float(np.max(np.abs(bias_gpu - npz["qfrc_bias"])))

        b_norm = np.linalg.norm(bias_gpu)
        denom = np.linalg.norm(M_gpu) * np.linalg.norm(X_gpu) + b_norm
        rel_residual = float(np.linalg.norm(M_gpu @ X_gpu - bias_gpu[:, None]) / denom) if denom > 0 else 0.0

        # Established tolerances
        pass_xpos = err_xpos < 1e-6
        pass_xmat = err_xmat < 1e-6
        pass_M = err_M < 1e-6
        pass_bias = err_bias < 1e-5
        pass_solve = rel_residual < 1e-6
        status_ok = pass_xpos and pass_xmat and pass_M and pass_bias and pass_solve

        rec = {
            "scenario": sc,
            "err_xpos": err_xpos,
            "err_xmat": err_xmat,
            "err_M": err_M,
            "err_bias": err_bias,
            "rel_residual": rel_residual,
            "status": "PASS" if status_ok else "FAIL",
        }
        results.append(rec)

        print(f"{sc:<28} | {err_xpos:<12.2e} | {err_xmat:<10.2e} | {err_M:<10.2e} | {err_bias:<10.2e} | {rel_residual:<12.2e} | {rec['status']}")

    return results


if __name__ == "__main__":
    run_oracle_verification()
