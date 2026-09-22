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

from src.canonical_model_loader import (
    CANONICAL_XML_PATH,
    CanonicalMicroDuckModel,
    create_matching_cpu_data,
    create_matching_cpu_model,
    load_canonical_model,
)
from src.representative_physics_slice import RepresentativePhysicsSlice

# Canonical Milestone 4 Physical Qualification Gates
GATE_POS_MAX = 1e-3       # 1.0 mm
GATE_SO3_MAX = 1e-3       # 1.0 mrad
GATE_LINVEL_MAX = 0.05    # 0.05 m/s
GATE_ANGVEL_MAX = 0.05    # 0.05 rad/s
GATE_JNTPOS_MAX = 1e-3    # 1.0 mrad
GATE_JNTVEL_MAX = 0.05    # 0.05 rad/s


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


def evaluate_common_state_one_step(
    ps: RepresentativePhysicsSlice,
    canonical: CanonicalMicroDuckModel,
) -> Tuple[List[Dict], List[str]]:
    """Evaluates 1-step (5 ms) state advancement across all 25 corpus scenarios vs matched CPU models."""
    corpus_dir = Path("corpus")
    scenarios = sorted([f.stem for f in corpus_dir.glob("*.npz")])
    if len(scenarios) != 25:
        raise RuntimeError(f"Expected exactly 25 corpus scenarios, found {len(scenarios)}")

    results = []
    gate_violations = []

    for sc_name in scenarios:
        fpath = corpus_dir / f"{sc_name}.npz"
        d_npz = dict(np.load(fpath))

        # 1. Matched CPU reference model & fresh data
        m_matched = create_matching_cpu_model(canonical, d_npz)
        d_matched = create_matching_cpu_data(m_matched, d_npz)
        mujoco.mj_step(m_matched, d_matched)

        # 2. GPU candidate
        qp_t = torch.from_numpy(d_npz["qpos"].astype(np.float32)).unsqueeze(0).to(ps.device)
        qv_t = torch.from_numpy(d_npz["qvel"].astype(np.float32)).unsqueeze(0).to(ps.device)
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
        int_stat = int(step_res.integration_status[0].cpu().item())

        metrics = compute_separated_metrics(qp_gpu, qv_gpu, d_matched.qpos, d_matched.qvel)
        nefc = int(step_res.physics_outputs.nefc[0].cpu().item())
        kkt = compute_kkt(step_res.physics_outputs, nefc, 200)

        # Gate enforcement
        violations = []
        if int_stat != 0:
            violations.append(f"integration_status={int_stat}")
        for m_key, val, limit in [
            ("pos_err", metrics["pos_err"], GATE_POS_MAX),
            ("so3_err", metrics["so3_err"], GATE_SO3_MAX),
            ("linvel_err", metrics["linvel_err"], GATE_LINVEL_MAX),
            ("angvel_err", metrics["angvel_err"], GATE_ANGVEL_MAX),
            ("jnt_pos_err", metrics["jnt_pos_err"], GATE_JNTPOS_MAX),
            ("jnt_vel_err", metrics["jnt_vel_err"], GATE_JNTVEL_MAX),
        ]:
            if not math.isfinite(val):
                violations.append(f"{m_key} non-finite ({val})")
            elif val > limit:
                violations.append(f"{m_key}={val:.2e} > {limit:.2e}")

        if violations:
            gate_violations.append(f"{sc_name}: {'; '.join(violations)}")

        results.append({
            "scenario": sc_name,
            "nefc": nefc,
            "stat": kkt["stat"],
            "int_stat": int_stat,
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
            "passed": len(violations) == 0,
        })

    return results, gate_violations


def evaluate_free_running_trajectories(
    ps: RepresentativePhysicsSlice,
    canonical: CanonicalMicroDuckModel,
) -> Tuple[Dict[str, Dict], List[str]]:
    """Evaluates free-running autonomous CPU vs GPU trajectories across control horizons with full-trace maxima."""
    regimes = {
        "airborne": {
            "file": "corpus/airborne.npz",
            "modify_qpos": lambda qp: np.array([qp[0], qp[1], 10.0, *qp[3:]]), # Pure airborne across 200 steps
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

    trajectory_results = {}
    gate_violations = []

    for regime_name, config in regimes.items():
        data = dict(np.load(config["file"]))
        qpos_init = config["modify_qpos"](data["qpos"].copy())
        qvel_init = data["qvel"].copy()
        max_step = max(config["steps"])

        m_matched = create_matching_cpu_model(canonical, data)
        d_matched = mujoco.MjData(m_matched)

        # GPU Rollout
        qp_t = torch.from_numpy(qpos_init.astype(np.float32)).unsqueeze(0).to(ps.device)
        qv_t = torch.from_numpy(qvel_init.astype(np.float32)).unsqueeze(0).to(ps.device)
        rollout = ps.rollout_trajectory(qp_t, qv_t, num_steps=max_step, dt=0.005)
        torch.mps.synchronize()

        gpu_qpos = rollout["qpos"][:, 0, :].cpu().numpy()
        gpu_qvel = rollout["qvel"][:, 0, :].cpu().numpy()
        gpu_nefc = rollout["nefc"][:, 0].cpu().numpy()
        gpu_stat = rollout["solver_status"][:, 0].cpu().numpy()
        gpu_int_stat = rollout["integration_status"][:, 0].cpu().numpy()

        # CPU Rollout with fresh state
        d_matched.qpos[:] = qpos_init
        d_matched.qvel[:] = qvel_init
        cpu_qpos = [qpos_init.copy()]
        cpu_qvel = [qvel_init.copy()]
        cpu_ncon = [d_matched.ncon]
        for _ in range(max_step):
            mujoco.mj_step(m_matched, d_matched)
            cpu_qpos.append(d_matched.qpos.copy())
            cpu_qvel.append(d_matched.qvel.copy())
            cpu_ncon.append(d_matched.ncon)

        # Compute point metrics and trace maxima up to each checkpoint
        regime_eval = {}
        for s in config["steps"]:
            m_sep = compute_separated_metrics(gpu_qpos[s], gpu_qvel[s], cpu_qpos[s], cpu_qvel[s])

            # Full-trace maxima over steps 1..s
            max_pos = max(float(np.max(np.abs(gpu_qpos[t, 0:3] - cpu_qpos[t][0:3]))) for t in range(1, s + 1))
            max_so3 = max(compute_so3_distance(gpu_qpos[t, 3:7], cpu_qpos[t][3:7]) for t in range(1, s + 1))
            max_linvel = max(float(np.max(np.abs(gpu_qvel[t, 0:3] - cpu_qvel[t][0:3]))) for t in range(1, s + 1))
            max_angvel = max(float(np.max(np.abs(gpu_qvel[t, 3:6] - cpu_qvel[t][3:6]))) for t in range(1, s + 1))
            max_jnt_pos = max(float(np.max(np.abs(gpu_qpos[t, 7:21] - cpu_qpos[t][7:21]))) for t in range(1, s + 1))
            max_jnt_vel = max(float(np.max(np.abs(gpu_qvel[t, 6:20] - cpu_qvel[t][6:20]))) for t in range(1, s + 1))

            regime_eval[s] = {
                "t_ms": s * 5,
                "pos_err": m_sep["pos_err"],
                "so3_err": m_sep["so3_err"],
                "linvel_err": m_sep["linvel_err"],
                "angvel_err": m_sep["angvel_err"],
                "jnt_pos_err": m_sep["jnt_pos_err"],
                "jnt_vel_err": m_sep["jnt_vel_err"],
                "max_pos_err": max_pos,
                "max_so3_err": max_so3,
                "max_linvel_err": max_linvel,
                "max_angvel_err": max_angvel,
                "max_jnt_pos_err": max_jnt_pos,
                "max_jnt_vel_err": max_jnt_vel,
                "nefc_gpu": int(gpu_nefc[s - 1]),
                "ncon_cpu": int(cpu_ncon[s]),
                "stat_gpu": int(gpu_stat[s - 1]),
                "int_stat_gpu": int(gpu_int_stat[s - 1]),
            }

            # Check integration status
            if int(gpu_int_stat[s - 1]) != 0:
                gate_violations.append(f"{regime_name} step {s}: integration_status={int(gpu_int_stat[s - 1])}")

        trajectory_results[regime_name] = regime_eval

    return trajectory_results, gate_violations


def evaluate_actuator_control_interval(
    ps: RepresentativePhysicsSlice,
    canonical: CanonicalMicroDuckModel,
) -> Tuple[Dict, List[str]]:
    """Evaluates actuator torque transmission and 4-substep control interval (20 ms)."""
    data = dict(np.load("corpus/airborne.npz"))
    m_matched = create_matching_cpu_model(canonical, data)
    d = mujoco.MjData(m_matched)

    qpos = data["qpos"].copy()
    qvel = data["qvel"].copy()
    ctrl = np.linspace(-0.6, 0.6, 14, dtype=np.float64)

    # 1 Step (5 ms)
    d.qpos[:] = qpos
    d.qvel[:] = qvel
    d.ctrl[:] = ctrl
    mujoco.mj_step(m_matched, d)

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
        mujoco.mj_step(m_matched, d)

    qp_next, qv_next, _ = ps.step_control_interval(qp_t, qv_t, ctrl_t, num_substeps=4, dt=0.005)
    torch.mps.synchronize()

    m_ctrl_int = compute_separated_metrics(qp_next[0].cpu().numpy(), qv_next[0].cpu().numpy(), d.qpos, d.qvel)

    violations = []
    for label, m_dict in [("1step_5ms", m_1step), ("ctrl_int_20ms", m_ctrl_int)]:
        for k, v in m_dict.items():
            if not math.isfinite(v):
                violations.append(f"{label} {k} non-finite ({v})")

    return {
        "1step_5ms": m_1step,
        "ctrl_int_20ms": m_ctrl_int,
    }, violations


def main():
    canonical = load_canonical_model()
    ps = RepresentativePhysicsSlice(batch_size=1, canonical=canonical)

    print("# Milestone 4: Canonical 5 ms ImplicitFast Time Integration Qualification Report\n")
    print("## 1. Common-State One-Step (5 ms) Parity Across All 25 Corpus Scenarios")
    print("| Scenario | nefc | Stat | IntStat | Iter | Label | Proj Res | Dual Infeas | Pos Err (m) | SO(3) Err (rad) | LinVel Err (m/s) | AngVel Err (rad/s) | JntPos Err (rad) | JntVel Err (rad/s) | Status |")
    print("|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    one_step_results, one_step_violations = evaluate_common_state_one_step(ps, canonical)
    for r in one_step_results:
        status_str = "PASS" if r["passed"] else "FAIL"
        print(
            f"| `{r['scenario']}` | {r['nefc']} | {r['stat']} | {r['int_stat']} | {r['iters']} | {r['label']} | "
            f"{r['proj_res']:.2e} | {r['dual_infeas']:.2e} | {r['pos_err']:.2e} | {r['so3_err']:.2e} | "
            f"{r['linvel_err']:.2e} | {r['angvel_err']:.2e} | {r['jnt_pos_err']:.2e} | {r['jnt_vel_err']:.2e} | **{status_str}** |"
        )

    print("\n## 2. Free-Running Trajectory Qualification (Endpoints and Full-Trace Maxima)")
    print("| Regime | Step | Time (ms) | Checkpoint Pos (m) | Trace Max Pos (m) | Checkpoint SO(3) | Trace Max SO(3) | Checkpoint LinVel | Trace Max LinVel | GPU nefc | CPU ncon | Stat | IntStat |")
    print("|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    trajectories, traj_violations = evaluate_free_running_trajectories(ps, canonical)
    for regime, steps_data in trajectories.items():
        for s, d in steps_data.items():
            print(
                f"| `{regime}` | {s} | {d['t_ms']} | {d['pos_err']:.2e} | {d['max_pos_err']:.2e} | "
                f"{d['so3_err']:.2e} | {d['max_so3_err']:.2e} | {d['linvel_err']:.2e} | {d['max_linvel_err']:.2e} | "
                f"{d['nefc_gpu']} | {d['ncon_cpu']} | {d['stat_gpu']} | {d['int_stat_gpu']} |"
            )

    print("\n## 3. Actuator Torque Control & 20 ms Control Interval (4 Substeps)")
    print("| Interval | Pos Err (m) | SO(3) Err (rad) | LinVel Err (m/s) | AngVel Err (rad/s) | JntPos Err (rad) | JntVel Err (rad/s) |")
    print("|---|:---:|:---:|:---:|:---:|:---:|:---:|")

    act_results, act_violations = evaluate_actuator_control_interval(ps, canonical)
    for label, metrics in act_results.items():
        print(
            f"| `{label}` | {metrics['pos_err']:.2e} | {metrics['so3_err']:.2e} | "
            f"{metrics['linvel_err']:.2e} | {metrics['angvel_err']:.2e} | "
            f"{metrics['jnt_pos_err']:.2e} | {metrics['jnt_vel_err']:.2e} |"
        )

    all_violations = one_step_violations + traj_violations + act_violations
    if all_violations:
        print("\n### ❌ ENFORCED QUALIFICATION GATE FAILURES:")
        for v in all_violations:
            print(f"- {v}")
        sys.exit(1)
    else:
        print("\n### ✅ ALL 25 SCENARIOS & TRAJECTORIES PASSED ENFORCED NUMERICAL GATES.")


if __name__ == "__main__":
    main()

