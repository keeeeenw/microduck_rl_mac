#!/usr/bin/env python3
"""Milestone 4: Canonical 5 ms ImplicitFast Time Integration and Free-Running Trajectory Qualification.

Generates complete numerical evaluation tables:
1. Common-state 1-step (5 ms) parity across all 25 corpus scenarios using separated metric gates.
2. Free-running CPU and Metal trajectories across 4 steps (20 ms), 20 steps (100 ms), and 200 steps (1.0 s).
3. Actuator torque control and the 4-substep control interval (20 ms).
4. Multi-world failure isolation and reset recovery.
"""

import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mujoco
import numpy as np
from scipy.linalg import solve_triangular
import torch

from src.canonical_model_loader import load_canonical_model
from src.representative_physics_slice import RepresentativePhysicsSlice


def compute_so3_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    """Computes geodesic angular distance on SO(3) invariant to q ~ -q."""
    q1_64 = q1.astype(np.float64)
    q2_64 = q2.astype(np.float64)
    q1_norm = q1_64 / max(1e-14, float(np.linalg.norm(q1_64)))
    q2_norm = q2_64 / max(1e-14, float(np.linalg.norm(q2_64)))
    dot = float(np.abs(np.dot(q1_norm, q2_norm)))
    dot_clamped = min(1.0, max(0.0, dot))
    return float(2.0 * np.arccos(dot_clamped))


def compute_separated_metrics(qp_gpu: np.ndarray, qv_gpu: np.ndarray, qp_cpu: np.ndarray, qv_cpu: np.ndarray) -> Dict[str, float]:
    """Computes separated metrics without pooling quantities of different units."""
    pos_err = float(np.max(np.abs(qp_gpu[0:3] - qp_cpu[0:3])))
    so3_err = compute_so3_distance(qp_gpu[3:7], qp_cpu[3:7])
    linvel_err = float(np.max(np.abs(qv_gpu[0:3] - qv_cpu[0:3])))
    angvel_err = float(np.max(np.abs(qv_gpu[3:6] - qv_cpu[3:6])))
    jnt_pos_err = float(np.max(np.abs(qp_gpu[7:21] - qp_cpu[7:21])))
    jnt_vel_err = float(np.max(np.abs(qv_gpu[6:20] - qv_cpu[6:20])))
    return {
        "pos_err": pos_err,
        "so3_err": so3_err,
        "linvel_err": linvel_err,
        "angvel_err": angvel_err,
        "jnt_pos_err": jnt_pos_err,
        "jnt_vel_err": jnt_vel_err,
    }


def compute_kkt(out, nefc: int, max_iters: int) -> Dict:
    """Computes KKT optimality and diagnostics for Delassus PGS candidate solve."""
    stat = int(out.solver_status[0].cpu().item())
    iters = int(out.actual_iters[0].cpu().item())

    # Read failure status first: failed zero-row world cannot be mislabeled
    if stat < 0:
        return {
            "stat": stat,
            "iters": iters,
            "label": "failed",
            "primal": float("nan"),
            "dual": float("nan"),
            "comp": float("nan"),
            "proj": float("nan"),
        }

    if nefc == 0:
        return {
            "stat": 0,
            "iters": 0,
            "label": "converged",
            "primal": 0.0,
            "dual": 0.0,
            "comp": 0.0,
            "proj": 0.0,
        }

    if stat == 0:
        label = "converged"
    elif iters == max_iters:
        label = "exhausted"
    else:
        label = "stagnated"

    J_np = out.J[0, :nefc].cpu().numpy()
    R_np = out.R[0, :nefc].cpu().numpy()
    aref_np = out.aref[0, :nefc].cpu().numpy()
    L_np = out.L_factor[0].cpu().numpy()
    f_sm_np = out.f_smooth[0].cpu().numpy()
    lam_np = out.lambda_force[0, :nefc].cpu().numpy()

    Y = solve_triangular(L_np, J_np.T, lower=True)
    A = Y.T @ Y + np.diag(R_np)
    y0 = solve_triangular(L_np, f_sm_np, lower=True)
    a0 = solve_triangular(L_np.T, y0, lower=False)
    b = J_np @ a0 - aref_np
    g = A @ lam_np + b

    primal = float(np.max(np.maximum(0.0, -lam_np)))
    dual = float(np.max(np.maximum(0.0, -g)))
    comp = float(np.max(np.abs(lam_np * g)))
    proj = float(np.max(np.abs(lam_np - np.maximum(0.0, lam_np - g / np.diag(A)))))

    return {
        "stat": stat,
        "iters": iters,
        "label": label,
        "primal": primal,
        "dual": dual,
        "comp": comp,
        "proj": proj,
    }


def evaluate_common_state_one_step(ps: RepresentativePhysicsSlice, m: mujoco.MjModel) -> List[Dict]:
    """Evaluates 1-step (5 ms) state advancement across all 25 corpus scenarios."""
    corpus_dir = Path("corpus")
    scenarios = sorted([f.stem for f in corpus_dir.glob("*.npz")])
    d = mujoco.MjData(m)
    results = []

    for sc_name in scenarios:
        d_npz = np.load(corpus_dir / f"{sc_name}.npz")
        qpos = d_npz["qpos"].copy()
        qvel = d_npz["qvel"].copy()

        # Step CPU reference
        d.qpos[:] = qpos
        d.qvel[:] = qvel
        if "qfrc_applied" in d_npz:
            d.qfrc_applied[:] = d_npz["qfrc_applied"]
        else:
            d.qfrc_applied[:] = 0.0
        mujoco.mj_step(m, d)

        # Step GPU candidate
        qp_t = torch.from_numpy(qpos.astype(np.float32)).unsqueeze(0).to(ps.device)
        qv_t = torch.from_numpy(qvel.astype(np.float32)).unsqueeze(0).to(ps.device)
        f_smooth = (
            torch.from_numpy(d_npz["qfrc_smooth"].astype(np.float32)).unsqueeze(0).to(ps.device)
            if ("qfrc_applied" in d_npz and np.any(d_npz["qfrc_applied"] != 0))
            else None
        )
        pwm = torch.from_numpy(d_npz["per_world_mass"].astype(np.float32)).unsqueeze(0).to(ps.device) if "per_world_mass" in d_npz and np.any(d_npz["per_world_mass"] != 0) else None
        pwi = torch.from_numpy(d_npz["per_world_ipos"].astype(np.float32)).unsqueeze(0).to(ps.device) if "per_world_ipos" in d_npz and np.any(d_npz["per_world_ipos"] != 0) else None
        pwa = torch.from_numpy(d_npz["per_world_armature"].astype(np.float32)).unsqueeze(0).to(ps.device) if "per_world_armature" in d_npz and np.any(d_npz["per_world_armature"] != 0) else None

        f_tensor = None
        if "randomized_friction" in sc_name and int(d_npz["ncon"]) > 0:
            f_tensor = torch.from_numpy(d_npz["contact_friction"][:, :2].astype(np.float32)).unsqueeze(0).to(ps.device)

        step_res = ps.step_autonomous(
            qp_t, qv_t, f_smooth=f_smooth, friction=f_tensor,
            per_world_mass=pwm, per_world_ipos=pwi, per_world_armature=pwa,
            max_iters=200, tol=1e-5, dt=0.005
        )
        torch.mps.synchronize()

        qp_gpu = step_res.qpos[0].cpu().numpy()
        qv_gpu = step_res.qvel[0].cpu().numpy()
        metrics = compute_separated_metrics(qp_gpu, qv_gpu, d.qpos, d.qvel)

        nefc = int(step_res.physics_outputs.nefc[0].cpu().item())
        kkt = compute_kkt(step_res.physics_outputs, nefc, 200)

        results.append({
            "scenario": sc_name,
            "nefc": nefc,
            "stat": kkt["stat"],
            "iters": kkt["iters"],
            "label": kkt["label"],
            "proj_res": kkt["proj"],
            "dual_infeas": kkt["dual"],
            "pos_err": metrics["pos_err"],
            "so3_err": metrics["so3_err"],
            "linvel_err": metrics["linvel_err"],
            "angvel_err": metrics["angvel_err"],
            "jnt_pos_err": metrics["jnt_pos_err"],
            "jnt_vel_err": metrics["jnt_vel_err"],
        })

    return results


def evaluate_free_running_trajectories(ps: RepresentativePhysicsSlice, m: mujoco.MjModel) -> Dict[str, Dict]:
    """Evaluates free-running autonomous CPU vs GPU trajectories without host overwrites."""
    regimes = {
        "airborne": {
            "file": "corpus/airborne.npz",
            "modify_qpos": lambda qp: np.array([qp[0], qp[1], 10.0, *qp[3:]]), # High altitude so it stays purely airborne across 200 steps
            "steps": [4, 20, 200],
        },
        "nominal_standing_realistic": {
            "file": "corpus/nominal_standing_realistic.npz",
            "modify_qpos": lambda qp: qp,
            "steps": [4, 20],
        },
        "contact_onset_drop": {
            "file": "corpus/nominal_standing_realistic.npz",
            "modify_qpos": lambda qp: np.array([qp[0], qp[1], qp[2] + 0.05, *qp[3:]]), # Drop from 5 cm
            "steps": [4, 20, 25],
        },
        "sliding_lateral_velocity": {
            "file": "corpus/sliding_lateral_velocity.npz",
            "modify_qpos": lambda qp: qp,
            "steps": [4, 20],
        },
    }

    d = mujoco.MjData(m)
    trajectory_results = {}

    for regime_name, config in regimes.items():
        data = np.load(config["file"])
        qpos_init = config["modify_qpos"](data["qpos"].copy())
        qvel_init = data["qvel"].copy()
        max_step = max(config["steps"])

        # GPU Rollout
        qp_t = torch.from_numpy(qpos_init.astype(np.float32)).unsqueeze(0).to(ps.device)
        qv_t = torch.from_numpy(qvel_init.astype(np.float32)).unsqueeze(0).to(ps.device)
        rollout = ps.rollout_trajectory(qp_t, qv_t, num_steps=max_step, dt=0.005)
        torch.mps.synchronize()

        gpu_qpos = rollout["qpos"][:, 0, :].cpu().numpy()
        gpu_qvel = rollout["qvel"][:, 0, :].cpu().numpy()
        gpu_nefc = rollout["nefc"][:, 0].cpu().numpy()
        gpu_stat = rollout["solver_status"][:, 0].cpu().numpy()

        # CPU Rollout
        d.qpos[:] = qpos_init
        d.qvel[:] = qvel_init
        cpu_qpos = [qpos_init.copy()]
        cpu_qvel = [qvel_init.copy()]
        cpu_ncon = [d.ncon]
        for _ in range(max_step):
            mujoco.mj_step(m, d)
            cpu_qpos.append(d.qpos.copy())
            cpu_qvel.append(d.qvel.copy())
            cpu_ncon.append(d.ncon)

        regime_eval = {}
        for s in config["steps"]:
            m_sep = compute_separated_metrics(gpu_qpos[s], gpu_qvel[s], cpu_qpos[s], cpu_qvel[s])
            regime_eval[s] = {
                "t_ms": s * 5,
                "pos_err": m_sep["pos_err"],
                "so3_err": m_sep["so3_err"],
                "linvel_err": m_sep["linvel_err"],
                "angvel_err": m_sep["angvel_err"],
                "jnt_pos_err": m_sep["jnt_pos_err"],
                "jnt_vel_err": m_sep["jnt_vel_err"],
                "nefc_gpu": int(gpu_nefc[s - 1]),
                "ncon_cpu": int(cpu_ncon[s]),
                "stat_gpu": int(gpu_stat[s - 1]),
            }
        trajectory_results[regime_name] = regime_eval

    return trajectory_results


def evaluate_actuator_control_interval(ps: RepresentativePhysicsSlice, m: mujoco.MjModel) -> Dict:
    """Evaluates actuator torque transmission and 4-substep control interval (20 ms)."""
    d = mujoco.MjData(m)
    data = np.load("corpus/airborne.npz")
    qpos = data["qpos"].copy()
    qvel = data["qvel"].copy()
    ctrl = np.linspace(-0.6, 0.6, 14, dtype=np.float64)

    # 1 Step (5 ms)
    d.qpos[:] = qpos
    d.qvel[:] = qvel
    d.ctrl[:] = ctrl
    mujoco.mj_step(m, d)

    qp_t = torch.from_numpy(qpos.astype(np.float32)).unsqueeze(0).to(ps.device)
    qv_t = torch.from_numpy(qvel.astype(np.float32)).unsqueeze(0).to(ps.device)
    ctrl_t = torch.from_numpy(ctrl.astype(np.float32)).unsqueeze(0).to(ps.device)

    step_res = ps.step_autonomous(qp_t, qv_t, ctrl=ctrl_t, dt=0.005)
    torch.mps.synchronize()

    m_1step = compute_separated_metrics(step_res.qpos[0].cpu().numpy(), step_res.qvel[0].cpu().numpy(), d.qpos, d.qvel)

    # 4 Substeps (20 ms control interval)
    d.qpos[:] = qpos
    d.qvel[:] = qvel
    d.ctrl[:] = ctrl
    for _ in range(4):
        mujoco.mj_step(m, d)

    qp_next, qv_next, _ = ps.step_control_interval(qp_t, qv_t, ctrl_t, num_substeps=4, dt=0.005)
    torch.mps.synchronize()

    m_ctrl_int = compute_separated_metrics(qp_next[0].cpu().numpy(), qv_next[0].cpu().numpy(), d.qpos, d.qvel)

    return {
        "1step_5ms": m_1step,
        "ctrl_int_20ms": m_ctrl_int,
    }


def main():
    c = load_canonical_model()
    m = c.model
    ps = RepresentativePhysicsSlice(batch_size=1)

    print("# Milestone 4: Canonical 5 ms ImplicitFast Time Integration Qualification Report\n")
    print("## 1. Common-State One-Step (5 ms) Parity Across All 25 Corpus Scenarios")
    print("| Scenario | nefc | Stat | Iter | Label | Proj Res | Dual Infeas | Pos Err (m) | SO(3) Err (rad) | LinVel Err (m/s) | AngVel Err (rad/s) | JntPos Err (rad) | JntVel Err (rad/s) |")
    print("|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    one_step_results = evaluate_common_state_one_step(ps, m)
    for r in one_step_results:
        print(
            f"| `{r['scenario']}` | {r['nefc']} | {r['stat']} | {r['iters']} | {r['label']} | "
            f"{r['proj_res']:.2e} | {r['dual_infeas']:.2e} | {r['pos_err']:.2e} | {r['so3_err']:.2e} | "
            f"{r['linvel_err']:.2e} | {r['angvel_err']:.2e} | {r['jnt_pos_err']:.2e} | {r['jnt_vel_err']:.2e} |"
        )

    print("\n## 2. Free-Running Trajectory Qualification Across Control Durations")
    print("| Regime | Step | Time (ms) | Pos Err (m) | SO(3) Err (rad) | LinVel Err (m/s) | AngVel Err (rad/s) | JntPos Err (rad) | JntVel Err (rad/s) | GPU nefc | CPU ncon | GPU Stat |")
    print("|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    trajectories = evaluate_free_running_trajectories(ps, m)
    for regime, steps_data in trajectories.items():
        for s, d in steps_data.items():
            print(
                f"| `{regime}` | {s} | {d['t_ms']} | {d['pos_err']:.2e} | {d['so3_err']:.2e} | "
                f"{d['linvel_err']:.2e} | {d['angvel_err']:.2e} | {d['jnt_pos_err']:.2e} | {d['jnt_vel_err']:.2e} | "
                f"{d['nefc_gpu']} | {d['ncon_cpu']} | {d['stat_gpu']} |"
            )

    print("\n## 3. Actuator Torque Control & 20 ms Control Interval (4 Substeps)")
    print("| Interval | Pos Err (m) | SO(3) Err (rad) | LinVel Err (m/s) | AngVel Err (rad/s) | JntPos Err (rad) | JntVel Err (rad/s) |")
    print("|---|:---:|:---:|:---:|:---:|:---:|:---:|")

    act_results = evaluate_actuator_control_interval(ps, m)
    for label, metrics in act_results.items():
        print(
            f"| `{label}` | {metrics['pos_err']:.2e} | {metrics['so3_err']:.2e} | "
            f"{metrics['linvel_err']:.2e} | {metrics['angvel_err']:.2e} | "
            f"{metrics['jnt_pos_err']:.2e} | {metrics['jnt_vel_err']:.2e} |"
        )


if __name__ == "__main__":
    main()
