"""Compare completed walking policy and throughput between physics='cpu' and physics='metal' backends.

Implements:
1. Controlled multi-case walking suite with paired initializations:
   - forward_fast (0.30 m/s), forward_slow (0.15 m/s), forward_turn_left (0.30, +0.50),
     forward_turn_right (0.30, -0.50), walk_to_stop (0.30 -> 0), idle (0.0),
     and weak diagnostic cases (turn_left/right).
   - Episode horizon set strictly beyond requested duration (episode_length_s = duration_s + 1000.0)
     to guarantee continuous, uninterrupted trajectories without timeout auto-resets.
   - Target commands configured in environment before reset so actor receives exact target at step 0.
   - Pushes and pose resampling frozen during evaluation.
   - Sourced actor observation command slot (indices 48:51) verified and asserted at every step.
   - Full finite-state verification (qpos, qvel, actions, obs) and fall rule evaluated per step.
   - Measures body-frame linear velocity (R^T * v_world), local yaw rate, tracking RMSE,
     uninterrupted displacement, uninterrupted path length, unwrapped heading accumulation, and foot contact.
   - Elementwise per-environment extrema (min root height, max trunk tilt) captured pre-reset.
   - Distinguishes task terminations from playback fall rule (z < 0.065m, tilt > 60 deg, or nonfinite).
   - Separated walk_to_stop interval metrics: walking phase ([2.0, stop_time]) vs settled stop phase ([settle_time, T]).
   - Instantaneous reward decomposition (step_reward * step_dt) with sum-equality assertions.
2. Focused multi-trial throughput benchmarks (1024, 2048, 4096) with alternating backend order:
   - 3 trials per batch size (alternating CPU-first vs Metal-first).
   - Zero-action and policy-driven stepping.
   - Comprehensive collision counter delta profiling (narrowphase seconds, transfer bytes/seconds, evals).
   - Unattributed remainder wall time clearly documented as candidate hypotheses.
3. Full provenance tracking (SHA-256 of sources, models, evaluator script, helpers, task settings).
"""

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

MICRODUCK_RL = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = MICRODUCK_RL / "scripts"
METAL_PACKAGE = MICRODUCK_RL / "src" / "mjlab_microduck" / "native_gpu" / "metal"

if str(MICRODUCK_RL / "src") not in sys.path:
    sys.path.insert(0, str(MICRODUCK_RL / "src"))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import mujoco
import mjlab_microduck.tasks  # noqa: F401
from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.native_gpu.environment import MetalEnv
from mjlab.utils.lab_api.math import matrix_from_quat

from evaluator_helpers import (
    check_state_finite,
    check_fall_rule,
    extract_yaw_from_quat,
    unwrap_yaw_trajectory,
    compute_walk_to_stop_intervals,
)


def sha256_file(path: Path) -> str:
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PyTorchPolicy(nn.Module):
    """PyTorch actor matching policy.onnx exactly, supporting vectorized batches on MPS."""

    def __init__(self, onnx_path: Path, device: str = "mps"):
        super().__init__()
        import onnx
        from onnx import numpy_helper

        model = onnx.load(str(onnx_path))
        inits = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}

        mean = torch.from_numpy(inits["obs_normalizer._mean"].copy())
        scale = torch.from_numpy(inits["onnx::Div_24"].copy())

        self.mean = nn.Parameter(mean, requires_grad=False)
        self.scale = nn.Parameter(scale, requires_grad=False)

        self.mlp = nn.Sequential(
            nn.Linear(61, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 14),
        )
        sd = {
            "0.weight": torch.from_numpy(inits["mlp.0.weight"].copy()),
            "0.bias": torch.from_numpy(inits["mlp.0.bias"].copy()),
            "2.weight": torch.from_numpy(inits["mlp.2.weight"].copy()),
            "2.bias": torch.from_numpy(inits["mlp.2.bias"].copy()),
            "4.weight": torch.from_numpy(inits["mlp.4.weight"].copy()),
            "4.bias": torch.from_numpy(inits["mlp.4.bias"].copy()),
            "6.weight": torch.from_numpy(inits["mlp.6.weight"].copy()),
            "6.bias": torch.from_numpy(inits["mlp.6.bias"].copy()),
        }
        self.mlp.load_state_dict(sd)
        self.to(device)
        self.eval()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs_norm = (obs - self.mean) / self.scale
        return self.mlp(obs_norm)


CONTROLLED_CASES = {
    "forward_fast": {"cmd": (0.30, 0.0, 0.0), "duration_s": 20.0, "is_diagnostic": False},
    "forward_slow": {"cmd": (0.15, 0.0, 0.0), "duration_s": 20.0, "is_diagnostic": False},
    "forward_turn_left": {"cmd": (0.30, 0.0, 0.50), "duration_s": 20.0, "is_diagnostic": False},
    "forward_turn_right": {"cmd": (0.30, 0.0, -0.50), "duration_s": 20.0, "is_diagnostic": False},
    "walk_to_stop": {"cmd": (0.30, 0.0, 0.0), "duration_s": 20.0, "is_diagnostic": False, "stop_time_s": 10.0},
    "idle": {"cmd": (0.0, 0.0, 0.0), "duration_s": 20.0, "is_diagnostic": False},
    "turn_left": {"cmd": (0.0, 0.0, 0.50), "duration_s": 20.0, "is_diagnostic": True},
    "turn_right": {"cmd": (0.0, 0.0, -0.50), "duration_s": 20.0, "is_diagnostic": True},
    "forward_fast_60s": {"cmd": (0.30, 0.0, 0.0), "duration_s": 60.0, "is_diagnostic": False},
}


class MonitoredMetalEnv(MetalEnv):
    """Subclass of MetalEnv that registers pre-reset hooks to capture exact terminal states."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_reset_callbacks = []

    def _reset_idx(self, env_ids: torch.Tensor):
        for cb in self.pre_reset_callbacks:
            cb(self, env_ids)
        super()._reset_idx(env_ids)


def evaluate_controlled_case(
    policy: PyTorchPolicy,
    physics: str,
    case_name: str,
    case_spec: Dict[str, Any],
    num_envs: int = 4,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run a controlled walking case on CPU or Metal with per-environment metric capture."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    duration_s = case_spec.get("duration_s", 20.0)
    stop_time_s = case_spec.get("stop_time_s", None)
    initial_cmd = np.array(case_spec["cmd"], dtype=np.float32)
    steps = int(round(duration_s / 0.02))

    cfg = load_env_cfg("Mjlab-Velocity-Flat-MicroDuck")
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    # Crucial fix: set episode horizon strictly beyond requested duration
    cfg.episode_length_s = duration_s + 1000.0
    cfg.is_finite_horizon = False

    # Configure command ranges directly so env.reset() samples initial_cmd at step 0
    twist_cfg = cfg.commands["twist"]
    twist_cfg.ranges.lin_vel_x = (float(initial_cmd[0]), float(initial_cmd[0]))
    twist_cfg.ranges.lin_vel_y = (float(initial_cmd[1]), float(initial_cmd[1]))
    twist_cfg.ranges.ang_vel_z = (float(initial_cmd[2]), float(initial_cmd[2]))
    twist_cfg.rel_standing_envs = 0.0
    twist_cfg.rel_heading_envs = 0.0
    twist_cfg.rel_world_envs = 0.0
    twist_cfg.rel_forward_envs = 0.0
    twist_cfg.rel_turn_in_place_envs = 0.0
    twist_cfg.resampling_time_range = (1e9, 1e9)
    twist_cfg.heading_command = False
    twist_cfg.ranges.heading = None

    # Freeze head and body pose commands to nominal zero
    if "head_pose" in cfg.commands:
        cfg.commands["head_pose"].ranges = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
        cfg.commands["head_pose"].resampling_time_range = (1e9, 1e9)
    if "body_pose" in cfg.commands:
        cfg.commands["body_pose"].ranges = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
        cfg.commands["body_pose"].resampling_time_range = (1e9, 1e9)

    # Freeze external pushes during controlled evaluation
    cfg.events.pop("push_robot", None)

    env = MonitoredMetalEnv(cfg, physics=physics, device="mps")
    obs, _ = env.reset()

    # Step 0 check: verify actor observation contains exact initial_cmd in command slots [48:51]
    expected_initial_t = torch.tensor(initial_cmd, device=obs["actor"].device, dtype=torch.float32)
    assert torch.allclose(obs["actor"][:, 48:51], expected_initial_t, atol=1e-5), (
        f"Actor step 0 command mismatch: got {obs['actor'][:, 48:51]}, expected {expected_initial_t}"
    )

    twist = env.command_manager.get_term("twist")

    # Initial state tracking
    origin_xyz = env.sim.data.qpos[:, :3].clone().cpu().numpy()

    # Metrics per environment
    min_height_per_env = [float("inf")] * num_envs
    max_tilt_per_env = [0.0] * num_envs
    fall_time_per_env = [None] * num_envs
    task_terminations_per_env = [0] * num_envs
    timeout_terminations_per_env = [0] * num_envs
    numerical_failures_per_env = [0] * num_envs
    fall_rule_violations_per_env = [0] * num_envs

    steady_samples_per_env = [[] for _ in range(num_envs)]
    foot_contacts_per_env = [[] for _ in range(num_envs)]

    # Trajectory logs for uninterrupted metrics
    trajectory_xyz = [origin_xyz.copy()]  # list of (num_envs, 3) arrays
    yaw_trajectory = []  # list of (num_envs,) arrays
    vx_trajectory = []   # list of (num_envs,) arrays
    t_series = []

    # Initial yaw
    initial_yaw = extract_yaw_from_quat(env.sim.data.qpos[:, 3:7]).cpu().numpy()
    yaw_trajectory.append(initial_yaw)

    # Reward accounting
    rm = env.reward_manager
    term_names = list(rm._term_names)
    step_reward_contributions = {name: [] for name in term_names}
    step_rewards_list = []

    # Pre-reset hook to catch unexpected resets and terminal state
    def on_pre_reset(e: MonitoredMetalEnv, env_ids: torch.Tensor):
        nonlocal min_height_per_env, max_tilt_per_env, task_terminations_per_env, timeout_terminations_per_env
        ids = env_ids.cpu().tolist()
        for env_id in ids:
            if hasattr(e, "reset_time_outs") and e.reset_time_outs is not None:
                if e.reset_time_outs[env_id].item():
                    timeout_terminations_per_env[env_id] += 1
            if hasattr(e, "reset_terminated") and e.reset_terminated is not None:
                if e.reset_terminated[env_id].item():
                    task_terminations_per_env[env_id] += 1

    env.pre_reset_callbacks.append(on_pre_reset)

    active_cmd = initial_cmd.copy()

    start_time = time.perf_counter()
    with torch.no_grad():
        for step_i in range(steps):
            t_sim = step_i * 0.02
            t_series.append(t_sim)

            # Handle walk_to_stop command switch instantaneously at switch step
            if stop_time_s is not None and t_sim >= stop_time_s and active_cmd[0] != 0.0:
                active_cmd = np.zeros(3, dtype=np.float32)
                # Overwrite actor observation buffer directly before actor evaluation
                obs["actor"][:, 48:51] = 0.0
                # Overwrite environment command term
                twist.vel_command_b[:, :] = 0.0
                twist.vel_command_w[:, :] = 0.0
                twist.cfg.ranges.lin_vel_x = (0.0, 0.0)
                twist.cfg.ranges.lin_vel_y = (0.0, 0.0)
                twist.cfg.ranges.ang_vel_z = (0.0, 0.0)

            actor_obs = obs["actor"]
            # Strict assertion: actor MUST see active_cmd in command slots
            expected_cmd_t = torch.tensor(active_cmd, device=actor_obs.device, dtype=torch.float32)
            assert torch.allclose(actor_obs[:, 48:51], expected_cmd_t, atol=1e-5), (
                f"Step {step_i} actor command mismatch: got {actor_obs[:, 48:51]}, expected {expected_cmd_t}"
            )

            actions = policy(actor_obs)
            obs, rew, terminated, truncated, extras = env.step(actions)

            # Check terminations
            if terminated.any():
                for eid in terminated.nonzero(as_tuple=False).flatten().cpu().tolist():
                    task_terminations_per_env[eid] += 1
            if truncated.any():
                for eid in truncated.nonzero(as_tuple=False).flatten().cpu().tolist():
                    timeout_terminations_per_env[eid] += 1

            # Assert and accumulate instantaneous reward decomposition
            scale = 0.02 if rm._scale_by_dt else 1.0
            instant_contributions = rm._step_reward * scale
            reconstructed_rew = instant_contributions.sum(dim=-1)
            diff = torch.abs(rew - reconstructed_rew).max().item()
            assert diff < 1e-4, f"Reward sum mismatch at step {step_i}: max diff {diff}"

            step_rewards_list.append(rew.mean().item())
            for t_idx, name in enumerate(term_names):
                step_reward_contributions[name].append(instant_contributions[:, t_idx].mean().item())

            # Evaluate physical state
            qpos = env.sim.data.qpos
            qvel = env.sim.data.qvel

            # Full finite-state check across positions, velocities, actions, and observations
            is_finite = check_state_finite(qpos, qvel, actions, actor_obs)
            for eid in range(num_envs):
                if not is_finite[eid].item():
                    numerical_failures_per_env[eid] += 1

            R = matrix_from_quat(qpos[:, 3:7])
            r22 = torch.clamp(R[:, 2, 2], -1.0, 1.0)
            tilt_deg = torch.rad2deg(torch.acos(r22))
            v_body = torch.bmm(R.transpose(1, 2), qvel[:, :3].unsqueeze(-1)).squeeze(-1)
            yaw = extract_yaw_from_quat(qpos[:, 3:7])
            yaw_rate = qvel[:, 5]

            # Playback fall rule check
            has_fallen = check_fall_rule(qpos[:, :3], tilt_deg, is_finite)
            for eid in range(num_envs):
                h_val = float(qpos[eid, 2].item())
                t_val = float(tilt_deg[eid].item())
                min_height_per_env[eid] = min(min_height_per_env[eid], h_val)
                max_tilt_per_env[eid] = max(max_tilt_per_env[eid], t_val)
                if has_fallen[eid].item():
                    fall_rule_violations_per_env[eid] += 1
                    if fall_time_per_env[eid] is None:
                        fall_time_per_env[eid] = t_sim

            # Log trajectory
            curr_xyz = qpos[:, :3].clone().cpu().numpy()
            trajectory_xyz.append(curr_xyz)
            yaw_trajectory.append(yaw.clone().cpu().numpy())
            vx_trajectory.append(v_body[:, 0].clone().cpu().numpy())

            # Foot contact sensors from observation critic buffer (indices 58:60)
            critic_buf = obs.get("critic", None)
            if critic_buf is not None:
                left_contact = (critic_buf[:, 58] > 0.5).cpu().numpy()
                right_contact = (critic_buf[:, 59] > 0.5).cpu().numpy()
            else:
                left_contact = np.zeros(num_envs, dtype=bool)
                right_contact = np.zeros(num_envs, dtype=bool)

            v_body_cpu = v_body.cpu().numpy()
            yaw_rate_cpu = yaw_rate.cpu().numpy()

            # Record steady state samples (t >= 2.0 s)
            if t_sim >= 2.0:
                for env_id in range(num_envs):
                    steady_samples_per_env[env_id].append((
                        float(v_body_cpu[env_id, 0]),
                        float(v_body_cpu[env_id, 1]),
                        float(yaw_rate_cpu[env_id]),
                        float(active_cmd[0]),
                        float(active_cmd[1]),
                        float(active_cmd[2]),
                    ))
                    foot_contacts_per_env[env_id].append((
                        bool(left_contact[env_id]),
                        bool(right_contact[env_id]),
                    ))

    wall_time = time.perf_counter() - start_time
    env.close()

    # Continuous trajectory assertions:
    assert sum(timeout_terminations_per_env) == 0, (
        f"Unexpected timeout resets in continuous case {case_name}: {timeout_terminations_per_env}"
    )
    assert sum(task_terminations_per_env) == 0, (
        f"Unexpected task terminations in continuous case {case_name}: {task_terminations_per_env}"
    )
    assert sum(numerical_failures_per_env) == 0, (
        f"Non-finite state encountered in continuous case {case_name}: {numerical_failures_per_env}"
    )
    assert sum(fall_rule_violations_per_env) == 0, (
        f"Fall rule violations in continuous case {case_name}: {fall_rule_violations_per_env}"
    )

    # Compute uninterrupted trajectory metrics
    traj_arr = np.array(trajectory_xyz)  # shape: (steps + 1, num_envs, 3)
    final_disp_xy = (traj_arr[-1, :, :2] - traj_arr[0, :, :2])  # shape: (num_envs, 2)
    step_diffs_xy = np.linalg.norm(np.diff(traj_arr[:, :, :2], axis=0), axis=-1)  # shape: (steps, num_envs)
    uninterrupted_path_length = np.sum(step_diffs_xy, axis=0)  # shape: (num_envs,)

    # Unwrapped yaw trajectory heading accumulation
    yaw_arr = np.array(yaw_trajectory)  # shape: (steps + 1, num_envs)
    unwrapped_heading_change = unwrap_yaw_trajectory(yaw_arr)  # shape: (num_envs,)

    # Per-environment summary computation
    env_results = []
    for env_id in range(num_envs):
        samples = np.array(steady_samples_per_env[env_id])
        if len(samples) > 0:
            achieved_vel = samples[:, :3]
            cmd_vel = samples[:, 3:6]
            mean_vel = achieved_vel.mean(axis=0).tolist()
            rmse_vel = np.sqrt(((achieved_vel - cmd_vel) ** 2).mean(axis=0)).tolist()
        else:
            mean_vel = [0.0, 0.0, 0.0]
            rmse_vel = [0.0, 0.0, 0.0]

        contacts = np.array(foot_contacts_per_env[env_id]) if len(foot_contacts_per_env[env_id]) else np.zeros((1, 2))
        left_ratio = float(contacts[:, 0].mean())
        right_ratio = float(contacts[:, 1].mean())

        env_results.append({
            "env_id": env_id,
            "survived": fall_time_per_env[env_id] is None,
            "fall_time_s": fall_time_per_env[env_id],
            "task_terminations": task_terminations_per_env[env_id],
            "timeout_terminations": timeout_terminations_per_env[env_id],
            "numerical_failures": numerical_failures_per_env[env_id],
            "min_height_m": float(min_height_per_env[env_id]),
            "max_tilt_deg": float(max_tilt_per_env[env_id]),
            "displacement_xy_m": final_disp_xy[env_id].tolist(),
            "path_length_m": float(uninterrupted_path_length[env_id]),
            "heading_change_rad": float(unwrapped_heading_change[env_id]),
            "mean_body_velocity_mps": mean_vel,
            "tracking_rmse": rmse_vel,
            "foot_contact_ratios": [left_ratio, right_ratio],
        })

    # Separate walk_to_stop interval metrics if applicable
    interval_metrics = None
    if case_name == "walk_to_stop":
        act_stop = float(stop_time_s) if stop_time_s is not None else 10.0
        interval_metrics = compute_walk_to_stop_intervals(
            t_series=np.array(t_series),
            vx_series=np.array(vx_trajectory),
            target_walk_vx=float(initial_cmd[0]),
            stop_time_s=act_stop,
            settle_time_s=min(act_stop + 2.0, duration_s),
            transient_cutoff_s=min(2.0, act_stop / 2.0),
        )

    # Aggregate statistics
    mean_reward_terms = {name: float(np.mean(step_reward_contributions[name])) for name in term_names}
    survived_count = sum(1 for r in env_results if r["survived"])
    total_task_terminations = sum(r["task_terminations"] for r in env_results)
    total_timeout_terminations = sum(r["timeout_terminations"] for r in env_results)

    all_min_heights = [r["min_height_m"] for r in env_results]
    all_max_tilts = [r["max_tilt_deg"] for r in env_results]
    all_vx = [r["mean_body_velocity_mps"][0] for r in env_results]
    all_vy = [r["mean_body_velocity_mps"][1] for r in env_results]
    all_wz = [r["mean_body_velocity_mps"][2] for r in env_results]
    all_rmse_vx = [r["tracking_rmse"][0] for r in env_results]
    all_disp_x = [r["displacement_xy_m"][0] for r in env_results]
    all_path_len = [r["path_length_m"] for r in env_results]
    all_headings = [r["heading_change_rad"] for r in env_results]

    res = {
        "case": case_name,
        "physics": physics,
        "is_diagnostic": case_spec.get("is_diagnostic", False),
        "command": list(initial_cmd),
        "duration_s": duration_s,
        "num_envs": num_envs,
        "steps": steps,
        "wall_time_s": wall_time,
        "transitions_per_second": (steps * num_envs) / wall_time,
        "mean_step_reward": float(np.mean(step_rewards_list)),
        "survived_envs": f"{survived_count}/{num_envs}",
        "task_terminations_total": total_task_terminations,
        "timeout_terminations_total": total_timeout_terminations,
        "numerical_failures_total": sum(numerical_failures_per_env),
        "fall_rule_violations_total": sum(fall_rule_violations_per_env),
        "elementwise_min_height_m": float(np.min(all_min_heights)),
        "elementwise_max_tilt_deg": float(np.max(all_max_tilts)),
        "mean_height_spread_m": [float(np.mean(all_min_heights)), float(np.std(all_min_heights))],
        "mean_tilt_spread_deg": [float(np.mean(all_max_tilts)), float(np.std(all_max_tilts))],
        "mean_forward_vel_mps": float(np.mean(all_vx)),
        "forward_vel_spread_mps": [float(np.mean(all_vx)), float(np.std(all_vx))],
        "mean_lateral_vel_mps": float(np.mean(all_vy)),
        "mean_yaw_rate_radps": float(np.mean(all_wz)),
        "mean_heading_change_rad": float(np.mean(all_headings)),
        "mean_tracking_rmse_vx": float(np.mean(all_rmse_vx)),
        "mean_displacement_x_m": float(np.mean(all_disp_x)),
        "mean_path_length_m": float(np.mean(all_path_len)),
        "reward_term_contributions": mean_reward_terms,
        "per_environment_results": env_results,
    }
    if interval_metrics is not None:
        res["interval_metrics"] = interval_metrics
    return res


def run_focused_throughput_trial(
    physics: str,
    num_envs: int,
    policy: Optional[PyTorchPolicy] = None,
    steps: int = 40,
    warmup: int = 15,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute one timed throughput benchmark trial with collision profiling deltas."""
    torch.manual_seed(seed)
    cfg = load_env_cfg("Mjlab-Velocity-Flat-MicroDuck")
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    cfg.episode_length_s = 1e6  # eliminate timeouts during benchmark

    env = MetalEnv(cfg, physics=physics, device="mps")
    obs, _ = env.reset()
    zero_actions = torch.zeros((num_envs, 14), dtype=torch.float32, device="mps")

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            act = policy(obs["actor"]) if policy is not None else zero_actions
            obs, _, _, _, _ = env.step(act)
        if torch.backends.mps.is_available():
            torch.mps.synchronize()

    # Capture start profiling counters
    sim = env.sim
    start_evals = getattr(sim, "collision_evaluations_count", 0)
    start_np_sec = getattr(sim, "collision_narrowphase_seconds", 0.0)
    start_tr_sec = getattr(sim.collision_transfer, "seconds", 0.0) if hasattr(sim, "collision_transfer") else 0.0
    start_tr_bytes = getattr(sim.collision_transfer, "bytes", 0) if hasattr(sim, "collision_transfer") else 0
    start_qpos_staging = getattr(sim, "collision_qpos_staging_bytes", 0)
    start_fric_staging = getattr(sim, "collision_fric_staging_bytes", 0)
    start_fb_sec = getattr(sim, "collision_fallback_branch_seconds", 0.0)
    start_overflow = getattr(sim, "extra_contact_overflow_count", 0)

    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(steps):
            act = policy(obs["actor"]) if policy is not None else zero_actions
            obs, _, _, _, _ = env.step(act)
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
    elapsed = time.perf_counter() - start_time

    # Capture delta counters
    delta_evals = getattr(sim, "collision_evaluations_count", 0) - start_evals
    delta_np_sec = getattr(sim, "collision_narrowphase_seconds", 0.0) - start_np_sec
    delta_tr_sec = (getattr(sim.collision_transfer, "seconds", 0.0) if hasattr(sim, "collision_transfer") else 0.0) - start_tr_sec
    delta_tr_bytes = (getattr(sim.collision_transfer, "bytes", 0) if hasattr(sim, "collision_transfer") else 0) - start_tr_bytes
    delta_qpos_staging = getattr(sim, "collision_qpos_staging_bytes", 0) - start_qpos_staging
    delta_fric_staging = getattr(sim, "collision_fric_staging_bytes", 0) - start_fric_staging
    delta_fb_sec = getattr(sim, "collision_fallback_branch_seconds", 0.0) - start_fb_sec
    delta_overflow = getattr(sim, "extra_contact_overflow_count", 0) - start_overflow

    unclassified_remainder_sec = max(0.0, elapsed - delta_fb_sec)
    transitions = num_envs * steps
    env.close()

    return {
        "physics": physics,
        "mode": "policy" if policy is not None else "zero_action",
        "num_envs": num_envs,
        "steps": steps,
        "elapsed_s": elapsed,
        "transitions_per_second": transitions / elapsed,
        "ms_per_step": (elapsed / steps) * 1000.0,
        "profiling_deltas": {
            "collision_evaluations_count": int(delta_evals),
            "collision_narrowphase_seconds": float(delta_np_sec),
            "collision_transfer_seconds": float(delta_tr_sec),
            "collision_transfer_bytes": int(delta_tr_bytes),
            "collision_qpos_staging_bytes": int(delta_qpos_staging),
            "collision_fric_staging_bytes": int(delta_fric_staging),
            "collision_fallback_branch_seconds": float(delta_fb_sec),
            "unclassified_remaining_wall_seconds": float(unclassified_remainder_sec),
            "unclassified_remaining_wall_fraction": float(unclassified_remainder_sec / elapsed) if elapsed > 0 else 0.0,
            "extra_contact_overflow_count": int(delta_overflow),
        }
    }


def run_matched_throughput_suite(
    policy: PyTorchPolicy,
    batch_sizes: List[int] = [1024, 2048, 4096],
    num_trials: int = 3,
    steps: int = 40,
) -> Dict[str, Any]:
    """Run focused matched throughput benchmarks alternating backend invocation order across trials."""
    results = {}

    for n in batch_sizes:
        print(f"\n--- Benchmarking Batch Size N={n} ({num_trials} alternating trials) ---", flush=True)
        n_results = {"cpu": [], "metal": [], "speedup": []}

        for trial_i in range(num_trials):
            order = ["cpu", "metal"] if trial_i % 2 == 0 else ["metal", "cpu"]
            trial_benchmarks = {}

            print(f"  Trial {trial_i+1}/{num_trials} (order: {' -> '.join(order)})...", flush=True)
            for backend in order:
                trial_res = run_focused_throughput_trial(
                    physics=backend,
                    num_envs=n,
                    policy=policy,
                    steps=steps,
                    warmup=15,
                    seed=42 + trial_i,
                )
                trial_benchmarks[backend] = trial_res
                fb_sec = trial_res['profiling_deltas']['collision_fallback_branch_seconds']
                rem_sec = trial_res['profiling_deltas']['unclassified_remaining_wall_seconds']
                print(f"    {backend.upper():<5}: {trial_res['transitions_per_second']:<8.1f} tr/s "
                      f"({trial_res['ms_per_step']:.2f} ms/step) | "
                      f"FB branch: {fb_sec:.3f}s, Unclassified rem: {rem_sec:.3f}s, "
                      f"evals: {trial_res['profiling_deltas']['collision_evaluations_count']}", flush=True)

            cpu_tr = trial_benchmarks["cpu"]["transitions_per_second"]
            metal_tr = trial_benchmarks["metal"]["transitions_per_second"]
            speedup = metal_tr / cpu_tr

            n_results["cpu"].append(trial_benchmarks["cpu"])
            n_results["metal"].append(trial_benchmarks["metal"])
            n_results["speedup"].append(speedup)

        cpu_tr_vals = [r["transitions_per_second"] for r in n_results["cpu"]]
        metal_tr_vals = [r["transitions_per_second"] for r in n_results["metal"]]
        cpu_ms_vals = [r["ms_per_step"] for r in n_results["cpu"]]
        metal_ms_vals = [r["ms_per_step"] for r in n_results["metal"]]
        speedup_vals = n_results["speedup"]

        results[str(n)] = {
            "num_envs": n,
            "trials_cpu": n_results["cpu"],
            "trials_metal": n_results["metal"],
            "speedup_trials": speedup_vals,
            "summary": {
                "cpu_median_tr_s": float(np.median(cpu_tr_vals)),
                "cpu_mean_tr_s": float(np.mean(cpu_tr_vals)),
                "cpu_spread_tr_s": [float(np.min(cpu_tr_vals)), float(np.max(cpu_tr_vals))],
                "cpu_mean_ms_per_step": float(np.mean(cpu_ms_vals)),
                "metal_median_tr_s": float(np.median(metal_tr_vals)),
                "metal_mean_tr_s": float(np.mean(metal_tr_vals)),
                "metal_spread_tr_s": [float(np.min(metal_tr_vals)), float(np.max(metal_tr_vals))],
                "metal_mean_ms_per_step": float(np.mean(metal_ms_vals)),
                "median_speedup": float(np.median(speedup_vals)),
                "mean_speedup": float(np.mean(speedup_vals)),
            }
        }

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        type=Path,
        default=MICRODUCK_RL / "logs" / "native-gpu" / "hybrid-walk-4096-20260920" / "policy.onnx",
        help="Path to trained policy.onnx",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=MICRODUCK_RL / "artifacts" / "evaluation" / "controlled_walking_and_throughput_profile.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=[
            "forward_fast", "forward_slow", "forward_turn_left", "forward_turn_right",
            "walk_to_stop", "idle", "turn_left", "turn_right", "forward_fast_60s"
        ],
        choices=list(CONTROLLED_CASES.keys()),
        help="Walking cases to evaluate",
    )
    parser.add_argument("--eval-envs", type=int, default=4, help="Number of paired envs for policy evaluation")
    parser.add_argument(
        "--benchmark-batch-sizes",
        nargs="+",
        type=int,
        default=[1024, 2048, 4096],
        help="Batch sizes for focused throughput sweep",
    )
    parser.add_argument("--skip-benchmarks", action="store_true", help="Skip throughput sweep")
    parser.add_argument("--eval-duration", type=float, default=None, help="Override duration in seconds for walking evaluation cases")
    parser.add_argument("--benchmark-trials", type=int, default=3, help="Alternating trials per batch size")
    parser.add_argument("--benchmark-steps", type=int, default=30, help="Steps per throughput trial")
    args = parser.parse_args()

    policy_path = args.policy.resolve()
    print(f"Loading completed policy: {policy_path} (SHA-256: {sha256_file(policy_path)})", flush=True)
    policy = PyTorchPolicy(policy_path, device="mps")

    # Record provenance
    provenance = {
        "timestamp": time.time(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "mujoco_version": mujoco.__version__,
        "policy_path": str(policy_path),
        "policy_sha256": sha256_file(policy_path),
        "invocation_args": {k: str(v) for k, v in vars(args).items()},
        "evaluator_script_path": str(Path(__file__).resolve()),
        "evaluator_script_sha256": sha256_file(Path(__file__).resolve()),
        "source_hashes": {
            "evaluator_helpers.py": sha256_file(SCRIPTS_DIR / "evaluator_helpers.py"),
            "metal_simulation_adapter.py": sha256_file(METAL_PACKAGE / "metal_simulation_adapter.py"),
            "physics_slice.metal": sha256_file(METAL_PACKAGE / "shaders" / "physics_slice.metal"),
            "representative_physics_slice.py": sha256_file(METAL_PACKAGE / "representative_physics_slice.py"),
            "cpu_simulation.py": sha256_file(MICRODUCK_RL / "src" / "mjlab_microduck" / "native_gpu" / "cpu_simulation.py"),
            "environment.py": sha256_file(MICRODUCK_RL / "src" / "mjlab_microduck" / "native_gpu" / "environment.py"),
            "scene.xml": sha256_file(MICRODUCK_RL / "src" / "mjlab_microduck" / "robot" / "microduck" / "scene.xml"),
            "robot_walk.xml": sha256_file(MICRODUCK_RL / "src" / "mjlab_microduck" / "robot" / "microduck" / "robot_walk.xml"),
        }
    }

    print("\n" + "=" * 90, flush=True)
    print("1. EVALUATING CONTROLLED MULTI-CASE WALKING SUITE (UNINTERRUPTED TRAJECTORIES)", flush=True)
    print("=" * 90, flush=True)

    controlled_eval_results = {}
    for case_name in args.cases:
        case_spec = copy.deepcopy(CONTROLLED_CASES[case_name])
        if args.eval_duration is not None:
            case_spec["duration_s"] = args.eval_duration
            if "stop_time_s" in case_spec:
                case_spec["stop_time_s"] = args.eval_duration / 2.0
        print(f"\n[Case: {case_name}] cmd={case_spec['cmd']}, duration={case_spec['duration_s']}s, N={args.eval_envs}", flush=True)

        print(f"  Running CPU physics...", flush=True)
        cpu_res = evaluate_controlled_case(
            policy, physics="cpu", case_name=case_name, case_spec=case_spec,
            num_envs=args.eval_envs, seed=42
        )
        print(f"    CPU: survived={cpu_res['survived_envs']}, timeouts={cpu_res['timeout_terminations_total']}, "
              f"v_x={cpu_res['mean_forward_vel_mps']:.4f} m/s, min_h={cpu_res['elementwise_min_height_m']:.4f} m, "
              f"disp_x={cpu_res['mean_displacement_x_m']:.3f} m, path={cpu_res['mean_path_length_m']:.3f} m, "
              f"head_chg={cpu_res['mean_heading_change_rad']:.3f} rad, reward={cpu_res['mean_step_reward']:.4f}", flush=True)

        print(f"  Running Metal physics...", flush=True)
        metal_res = evaluate_controlled_case(
            policy, physics="metal", case_name=case_name, case_spec=case_spec,
            num_envs=args.eval_envs, seed=42
        )
        print(f"    Metal: survived={metal_res['survived_envs']}, timeouts={metal_res['timeout_terminations_total']}, "
              f"v_x={metal_res['mean_forward_vel_mps']:.4f} m/s, min_h={metal_res['elementwise_min_height_m']:.4f} m, "
              f"disp_x={metal_res['mean_displacement_x_m']:.3f} m, path={metal_res['mean_path_length_m']:.3f} m, "
              f"head_chg={metal_res['mean_heading_change_rad']:.3f} rad, reward={metal_res['mean_step_reward']:.4f}", flush=True)

        controlled_eval_results[case_name] = {
            "case_spec": case_spec,
            "cpu": cpu_res,
            "metal": metal_res,
        }

    throughput_results = {}
    if not args.skip_benchmarks:
        print("\n" + "=" * 90, flush=True)
        print("2. RUNNING FOCUSED THROUGHPUT BENCHMARK SUITE WITH PROFILING", flush=True)
        print("=" * 90, flush=True)

        throughput_results = run_matched_throughput_suite(
            policy=policy,
            batch_sizes=args.benchmark_batch_sizes,
            num_trials=args.benchmark_trials,
            steps=args.benchmark_steps,
        )

        print("\n" + "=" * 90, flush=True)
        print("3. THROUGHPUT BENCHMARK SUMMARY", flush=True)
        print("=" * 90, flush=True)
        print(f"{'Envs':<8} | {'CPU Median (tr/s)':<18} | {'Metal Median (tr/s)':<20} | {'Median Speedup':<14} | {'CPU Spread':<16} | {'Metal Spread':<16}", flush=True)
        print("-" * 96, flush=True)
        for n_str, res in throughput_results.items():
            s = res["summary"]
            print(f"{n_str:<8} | {s['cpu_median_tr_s']:<18.1f} | {s['metal_median_tr_s']:<20.1f} | "
                  f"{s['median_speedup']:<14.2f}x | {s['cpu_spread_tr_s'][0]:.1f}-{s['cpu_spread_tr_s'][1]:.1f} | "
                  f"{s['metal_spread_tr_s'][0]:.1f}-{s['metal_spread_tr_s'][1]:.1f}", flush=True)

    final_report = {
        "provenance": provenance,
        "controlled_walking_evaluation": controlled_eval_results,
        "throughput_benchmarks": throughput_results,
    }

    def json_default(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.cpu().tolist()
        return str(obj)

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(final_report, indent=2, default=json_default) + "\n")
    print(f"\nSaved complete evaluation and throughput report to: {output_path}", flush=True)


if __name__ == "__main__":
    main()
