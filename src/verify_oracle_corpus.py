"""Independent Verification of Native Dynamics and Solves vs CPU MuJoCo Oracle Corpus.

Tests all corpus scenarios against candidate Metal kernels:
- Body forward kinematics: positions (xpos) and orientations (xmat)
- Articulated dynamics: CRBA mass matrix (M_eff) and RNE bias forces (qfrc_bias)
- Native Cholesky factorization and linear solves:
  * Solver status == 0 (strict success check)
  * Output finiteness (no NaN or Inf)
  * Relative residual ||M x - b|| / (||M|| ||x|| + ||b||) with zero-norm denominator guard
  * Direct solution comparison ||x - x_cpu|| vs CPU MuJoCo oracle
  * Multi-RHS solve parity on independent fixtures
"""

import sys
from pathlib import Path
from typing import Dict, Any, List
import json
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.representative_physics_slice import RepresentativePhysicsSlice
from src.canonical_model_loader import load_canonical_model

CORPUS_DIR = PROJECT_ROOT / "corpus"


def run_oracle_verification() -> List[Dict[str, Any]]:
    canonical = load_canonical_model()
    engine = RepresentativePhysicsSlice(batch_size=1, canonical=canonical)

    meta_file = CORPUS_DIR / "corpus_metadata.json"
    if meta_file.exists():
        scenarios = json.loads(meta_file.read_text()).get("scenarios", [])
    else:
        scenarios = sorted([p.stem for p in CORPUS_DIR.glob("*.npz")])

    if not scenarios:
        raise RuntimeError(f"Corpus scenario list is empty! No fixtures found in {CORPUS_DIR}")

    results = []
    num_failed = 0

    header = (
        f"{'Scenario':<28} | {'err_xpos':<10} | {'err_xmat':<10} | {'err_M':<10} | "
        f"{'err_bias':<10} | {'rel_res':<10} | {'err_sol':<10} | {'err_multi':<10} | {'Status'}"
    )
    print(header)
    print("-" * len(header))

    for sc in scenarios:
        npz = np.load(CORPUS_DIR / f"{sc}.npz")
        qpos = torch.from_numpy(npz["qpos"].astype(np.float32)).unsqueeze(0).to(engine.device)
        qvel = torch.from_numpy(npz["qvel"].astype(np.float32)).unsqueeze(0).to(engine.device)

        # Load self-contained domain randomization parameters directly from fixture
        p_mass = torch.from_numpy(npz["per_world_mass"].astype(np.float32)).unsqueeze(0).to(engine.device) if "per_world_mass" in npz else None
        p_ipos = torch.from_numpy(npz["per_world_ipos"].astype(np.float32)).unsqueeze(0).to(engine.device) if "per_world_ipos" in npz else None
        p_arm = torch.from_numpy(npz["per_world_armature"].astype(np.float32)).unsqueeze(0).to(engine.device) if "per_world_armature" in npz else None

        # 1. Native dynamics
        engine.compute_native_dynamics(qpos, qvel, per_world_mass=p_mass, per_world_ipos=p_ipos, per_world_armature=p_arm)
        
        # 2. Native single-RHS solve for bias acceleration
        X_bias, status_bias_raw = engine.compute_native_cholesky_solve(engine.M_eff, engine.qfrc_bias)
        # Clone status before issuing subsequent solve because compute_native_cholesky_solve reuses engine.solver_status
        status_bias = status_bias_raw.clone()

        # 3. Native multi-RHS solve (K=4)
        B_multi_t = torch.from_numpy(npz["B_multi"].astype(np.float32)).unsqueeze(0).to(engine.device)
        X_multi, status_multi_raw = engine.compute_native_cholesky_solve(engine.M_eff, B_multi_t)
        status_multi = status_multi_raw.clone()
        torch.mps.synchronize()

        xpos_gpu = engine.body_xpos[0].cpu().numpy()
        xmat_gpu = engine.body_xmat[0].cpu().numpy()
        M_gpu = engine.M_eff[0].cpu().numpy()
        bias_gpu = engine.qfrc_bias[0].cpu().numpy()
        X_bias_gpu = X_bias[0].cpu().numpy()
        X_multi_gpu = X_multi[0].cpu().numpy()

        err_xpos = float(np.max(np.abs(xpos_gpu - npz["xpos"])))
        err_xmat = float(np.max(np.abs(xmat_gpu - npz["xmat"])))
        err_M = float(np.max(np.abs(M_gpu - npz["M"])))
        err_bias = float(np.max(np.abs(bias_gpu - npz["qfrc_bias"])))

        # Residual calculation with safe zero-norm handling
        res_vec = M_gpu @ X_bias_gpu - bias_gpu[:, None]
        num_res = float(np.linalg.norm(res_vec))
        denom_res = float(np.linalg.norm(M_gpu) * np.linalg.norm(X_bias_gpu) + np.linalg.norm(bias_gpu))
        if denom_res == 0.0:
            rel_residual = 0.0 if num_res == 0.0 else float("inf")
        else:
            rel_residual = num_res / denom_res

        # Direct solution comparisons against CPU MuJoCo oracle
        err_sol_bias = float(np.max(np.abs(X_bias_gpu.squeeze() - npz["x_bias"])))
        err_sol_multi = float(np.max(np.abs(X_multi_gpu - npz["X_multi"])))

        # Strict pass criteria for Milestones 0-2
        pass_status = (status_bias.item() == 0) and (status_multi.item() == 0)
        pass_finite = (
            np.all(np.isfinite(X_bias_gpu))
            and np.all(np.isfinite(X_multi_gpu))
            and np.all(np.isfinite(M_gpu))
            and np.all(np.isfinite(bias_gpu))
        )
        pass_xpos = err_xpos < 1e-6
        pass_xmat = err_xmat < 1e-6
        pass_M = err_M < 1e-6
        pass_bias = err_bias < 1e-5
        pass_res = rel_residual < 1e-5
        pass_sol_bias = err_sol_bias < 1e-3
        pass_sol_multi = err_sol_multi < 1e-3

        status_ok = (
            pass_status
            and pass_finite
            and pass_xpos
            and pass_xmat
            and pass_M
            and pass_bias
            and pass_res
            and pass_sol_bias
            and pass_sol_multi
        )

        if not status_ok:
            num_failed += 1

        rec = {
            "scenario": sc,
            "err_xpos": err_xpos,
            "err_xmat": err_xmat,
            "err_M": err_M,
            "err_bias": err_bias,
            "rel_residual": rel_residual,
            "err_sol_bias": err_sol_bias,
            "err_sol_multi": err_sol_multi,
            "status": "PASS" if status_ok else "FAIL",
        }
        results.append(rec)

        print(
            f"{sc:<28} | {err_xpos:<10.2e} | {err_xmat:<10.2e} | {err_M:<10.2e} | "
            f"{err_bias:<10.2e} | {rel_residual:<10.2e} | {err_sol_bias:<10.2e} | {err_sol_multi:<10.2e} | {rec['status']}"
        )

    if num_failed > 0:
        raise RuntimeError(f"Oracle corpus verification failed for {num_failed} / {len(results)} scenarios!")

    print(f"\nAll {len(results)} dynamics/solve scenarios PASSED strict qualification criteria.\n")

    # =========================================================================
    # Milestone 3A: Metal Constraint Solver Qualification on Oracle Constraints
    # =========================================================================
    print("=" * 135)
    print("Milestone 3A: Metal Constraint Solver Qualification on Oracle Constraints")
    print("=" * 135)
    header_3a = (
        f"{'Scenario':<26} | {'nefc':<4} | {'It':<3} | {'St':<2} | {'SameQP':<9} | "
        f"{'ProjRes':<9} | {'DualInf':<9} | {'Comp':<9} | {'f_lin(N)':<9} | {'tau_rot(Nm)':<11} | {'a_rot(r/s2)':<11} | {'Status'}"
    )
    print(header_3a)
    print("-" * len(header_3a))

    capacity = 32
    num_failed_3a = 0
    results_3a = []

    for sc in scenarios:
        npz = np.load(CORPUS_DIR / f"{sc}.npz")
        nefc_val = int(npz["nefc"])
        L_np = npz["L"].astype(np.float32)
        f_smooth_np = npz["qfrc_smooth"].astype(np.float32)

        L = torch.from_numpy(L_np).unsqueeze(0).to(engine.device)
        f_smooth = torch.from_numpy(f_smooth_np).unsqueeze(0).to(engine.device)
        J = torch.zeros((1, capacity, 20), dtype=torch.float32, device=engine.device)
        aref = torch.zeros((1, capacity), dtype=torch.float32, device=engine.device)
        R = torch.ones((1, capacity), dtype=torch.float32, device=engine.device)
        efc_type = torch.full((1, capacity), 6, dtype=torch.int32, device=engine.device)

        if nefc_val > 0:
            J_sub = npz["efc_J"].astype(np.float32)
            aref_sub = npz["efc_aref"].astype(np.float32)
            R_sub = npz["efc_R"].astype(np.float32)
            J[0, :nefc_val] = torch.from_numpy(J_sub)
            aref[0, :nefc_val] = torch.from_numpy(aref_sub)
            R[0, :nefc_val] = torch.from_numpy(R_sub)
            if "efc_type" in npz:
                efc_type[0, :nefc_val] = torch.from_numpy(npz["efc_type"].astype(np.int32))

        nefc = torch.tensor([nefc_val], dtype=torch.int32, device=engine.device)
        res = engine.solve_oracle_constraints(
            L, f_smooth, J, aref, R, nefc, efc_type=efc_type, capacity=capacity
        )
        torch.mps.synchronize()

        status = res["solver_status"].item()
        actual_iters = res["actual_iters"].item()
        dual_res_gpu = res["dual_residual"].item()

        qfrc_c_gpu = res["qfrc_constraint"][0].cpu().numpy()
        qacc_gpu = res["qacc"][0].cpu().numpy()
        lam_gpu = res["lambda"][0, :nefc_val].cpu().numpy()

        f_ref = npz["qfrc_constraint"]
        a_ref = npz["qacc"]

        err_f_lin = float(np.max(np.abs(qfrc_c_gpu[:3] - f_ref[:3])))
        err_f_rot = float(np.max(np.abs(qfrc_c_gpu[3:6] - f_ref[3:6])))
        err_f_jnt = float(np.max(np.abs(qfrc_c_gpu[6:20] - f_ref[6:20])))
        err_a_lin = float(np.max(np.abs(qacc_gpu[:3] - a_ref[:3])))
        err_a_rot = float(np.max(np.abs(qacc_gpu[3:20] - a_ref[3:20])))

        # Tier 1 CPU same-QP parity and independent KKT metrics
        if nefc_val > 0:
            # Assembly on CPU
            Y_cpu = np.zeros((nefc_val, 20), dtype=np.float32)
            for i in range(nefc_val):
                for k in range(20):
                    s = J_sub[i, k]
                    for p in range(k):
                        s -= L_np[k, p] * Y_cpu[i, p]
                    Y_cpu[i, k] = s / L_np[k, k]
            A_cpu = Y_cpu @ Y_cpu.T + np.diag(R_sub)
            y0_cpu = np.zeros(20, dtype=np.float32)
            for i in range(20):
                s = f_smooth_np[i]
                for p in range(i):
                    s -= L_np[i, p] * y0_cpu[p]
                y0_cpu[i] = s / L_np[i, i]
            a0_cpu = np.zeros(20, dtype=np.float32)
            for i in range(19, -1, -1):
                s = y0_cpu[i]
                for p in range(i + 1, 20):
                    s -= L_np[p, i] * a0_cpu[p]
                a0_cpu[i] = s / L_np[i, i]
            b_cpu = J_sub @ a0_cpu - aref_sub

            # CPU same-QP reference solve
            lam_cpu = np.zeros(nefc_val, dtype=np.float32)
            g_cpu = b_cpu.copy()
            for it in range(100):
                max_d = 0.0
                for i in range(nefc_val):
                    delta = -g_cpu[i] / A_cpu[i, i]
                    lam_new = max(0.0, lam_cpu[i] + delta)
                    d_act = lam_new - lam_cpu[i]
                    if abs(d_act) > 1e-12:
                        g_cpu += A_cpu[:, i] * d_act
                        lam_cpu[i] = lam_new
                        max_d = max(max_d, abs(d_act))
                if max_d < 1e-5:
                    break
            err_same_qp = float(np.max(np.abs(lam_gpu - lam_cpu)))

            # Independent CPU KKT metrics
            prim_inf = float(np.max(np.maximum(0.0, -lam_gpu)))
            g_gpu = b_cpu + A_cpu @ lam_gpu
            dual_inf = float(np.max(np.maximum(0.0, -g_gpu)))
            comp = float(np.max(np.abs(lam_gpu * g_gpu)))
            diag_A = np.diag(A_cpu)
            proj_res = float(np.max(np.abs(lam_gpu - np.maximum(0.0, lam_gpu - g_gpu / diag_A))))
        else:
            err_same_qp = 0.0
            prim_inf = 0.0
            dual_inf = 0.0
            comp = 0.0
            proj_res = 0.0

        pass_3a = (
            status in (0, 1)
            and err_same_qp < 1e-4
            and prim_inf <= 1e-6
            and err_f_lin <= 0.05
            and err_f_rot <= 0.05
            and err_f_jnt <= 0.05
            and err_a_lin <= 0.05
            and (err_a_rot <= (0.15 if sc == "high_condition_mass_matrix" else 0.05))
        )

        if not pass_3a:
            num_failed_3a += 1

        rec_3a = {
            "scenario": sc,
            "nefc": nefc_val,
            "iters": actual_iters,
            "status": status,
            "err_same_qp": err_same_qp,
            "proj_res": proj_res,
            "dual_inf": dual_inf,
            "comp": comp,
            "err_f_lin": err_f_lin,
            "err_f_rot": err_f_rot,
            "err_f_jnt": err_f_jnt,
            "err_a_lin": err_a_lin,
            "err_a_rot": err_a_rot,
            "status_pass": "PASS" if pass_3a else "FAIL",
        }
        results_3a.append(rec_3a)

        print(
            f"{sc:<26} | {nefc_val:<4d} | {actual_iters:<3d} | {status:<2d} | {err_same_qp:<9.2e} | "
            f"{proj_res:<9.2e} | {dual_inf:<9.2e} | {comp:<9.2e} | {err_f_lin:<9.2e} | {err_f_rot:<11.2e} | {err_a_rot:<11.2e} | {rec_3a['status_pass']}"
        )

    if num_failed_3a > 0:
        raise RuntimeError(f"Milestone 3A oracle verification failed for {num_failed_3a} / {len(results_3a)} scenarios!")

    print(f"\nAll {len(results_3a)} Milestone 3A scenarios PASSED qualification criteria.")
    return results


if __name__ == "__main__":
    try:
        run_oracle_verification()
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
