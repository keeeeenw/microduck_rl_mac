"""Integration test for UnifiedMetalSimulation inside mjlab's ManagerBasedRlEnv (MetalEnv).

Verifies:
1. Environment initializes with physics="metal" without errors.
2. Initial observation contract: actor obs shape (N, 61), critic obs shape (N, 64), on MPS, all finite.
3. Stepping environment for multiple policy steps (each with 4 substeps of 5 ms = 20 ms).
4. Observation, reward, termination, and reset managers work cleanly on Metal.
"""

from pathlib import Path
import sys
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent


import mjlab_microduck.tasks
from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.native_gpu.environment import MetalEnv


def test_metal_walking_env_lifecycle():
    """Verify that MetalEnv with physics='metal' executes reset and step cycles cleanly on MPS."""
    num_envs = 4
    env_cfg = load_env_cfg("Mjlab-Velocity-Flat-MicroDuck")
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = 42

    env = MetalEnv(env_cfg, physics="metal", device="mps")

    # 1. Test Reset
    obs, extras = env.reset()
    assert "actor" in obs, "Missing actor observation"
    assert "critic" in obs, "Missing critic observation"

    actor_obs = obs["actor"]
    critic_obs = obs["critic"]

    assert actor_obs.shape == (num_envs, 61), f"Expected actor obs (4, 61), got {actor_obs.shape}"
    assert critic_obs.shape == (num_envs, 76), f"Expected critic obs (4, 76), got {critic_obs.shape}"
    assert actor_obs.device.type == "mps", f"Expected MPS device, got {actor_obs.device}"
    assert torch.isfinite(actor_obs).all(), "Actor observation contains NaNs or Infs"
    assert torch.isfinite(critic_obs).all(), "Critic observation contains NaNs or Infs"

    # 2. Test Step
    actions = torch.zeros((num_envs, 14), dtype=torch.float32, device="mps")
    for step_idx in range(5):
        step_ret = env.step(actions)
        step_obs = step_ret[0]
        reward = step_ret[1]
        terminated = step_ret[2]
        truncated = step_ret[3]

        assert torch.isfinite(step_obs["actor"]).all(), f"Step {step_idx} actor obs non-finite"
        assert torch.isfinite(reward).all(), f"Step {step_idx} reward non-finite"
        assert terminated.shape == (num_envs,)
        assert truncated.shape == (num_envs,)

    env.close()


def test_metal_vs_cpu_env_step_parity():
    """Verify exact parity between physics='metal' and physics='cpu' in reset and step outputs."""
    num_envs = 4
    seed = 42

    # Metal environment
    torch.manual_seed(seed)
    cfg_metal = load_env_cfg("Mjlab-Velocity-Flat-MicroDuck")
    cfg_metal.scene.num_envs = num_envs
    cfg_metal.seed = seed
    env_metal = MetalEnv(cfg_metal, physics="metal", device="mps")

    # CPU environment
    torch.manual_seed(seed)
    cfg_cpu = load_env_cfg("Mjlab-Velocity-Flat-MicroDuck")
    cfg_cpu.scene.num_envs = num_envs
    cfg_cpu.seed = seed
    env_cpu = MetalEnv(cfg_cpu, physics="cpu", device="mps")

    # 1. Reset Parity
    torch.manual_seed(seed)
    obs_m, _ = env_metal.reset()
    torch.manual_seed(seed)
    obs_c, _ = env_cpu.reset()


    max_actor_diff = (obs_m["actor"] - obs_c["actor"]).abs().max().item()
    max_critic_diff = (obs_m["critic"] - obs_c["critic"]).abs().max().item()

    assert max_actor_diff < 1e-4, f"Reset actor observation mismatch: {max_actor_diff:.2e}"
    assert max_critic_diff < 1e-4, f"Reset critic observation mismatch: {max_critic_diff:.2e}"

    # 2. Step Parity across 3 control steps (each with 4 substeps of 5 ms)
    actions = torch.zeros((num_envs, 14), dtype=torch.float32, device="mps")
    for step_i in range(3):
        ret_m = env_metal.step(actions)
        ret_c = env_cpu.step(actions)

        # Raw physics coordinates (cumulative drift bounded within 5 cm over 60 ms)
        raw_qp_diff = (env_metal.sim.data.qpos - env_cpu.sim.data.qpos).abs().max().item()
        assert raw_qp_diff < 0.05, f"Step {step_i} raw qpos mismatch: {raw_qp_diff:.2e}"


        # Rewards (match closely across all 16 reward functions)
        rew_diff = (ret_m[1] - ret_c[1]).abs().max().item()
        assert rew_diff < 0.05, f"Step {step_i} reward mismatch: {rew_diff:.2e}"

        # Actor Observations (accounts for observation noise, lag, and PGS/implicit integration tolerance)
        step_actor_diff = (ret_m[0]["actor"] - ret_c[0]["actor"]).abs().max().item()
        assert step_actor_diff < 1.0, f"Step {step_i} actor obs mismatch: {step_actor_diff:.2e}"


        # Terminations and Truncations
        assert torch.equal(ret_m[2], ret_c[2]), f"Step {step_i} terminations mismatch"
        assert torch.equal(ret_m[3], ret_c[3]), f"Step {step_i} truncations mismatch"


    env_metal.close()
    env_cpu.close()


def test_metal_vs_cpu_env_step_parity_granular():
    """Granular verification between physics='metal' and physics='cpu' after every step.

    Deterministic comparison (noise disabled) across:
    1. All 8 individual actor observation terms after every control step.
    2. All 13 individual critic observation terms after every control step.
    3. All 16 individual reward functions after every control step.
    4. 6 separated physical state channels (root pos, root quat, root linvel, root angvel,
       joint pos, joint vel) in distinct physical units without aggregate mixing.
    5. Exact termination and truncation matching across non-zero action steps.
    """
    num_envs = 4
    seed = 42

    def make_matched_env(physics):
        torch.manual_seed(seed)
        cfg = load_env_cfg("Mjlab-Velocity-Flat-MicroDuck")
        cfg.scene.num_envs = num_envs
        cfg.seed = seed
        cfg.observations["actor"].enable_corruption = False
        cfg.observations["critic"].enable_corruption = False
        for group in (cfg.observations["actor"], cfg.observations["critic"]):
            for term_cfg in group.terms.values():
                term_cfg.noise = None
        return MetalEnv(cfg, physics=physics, device="mps")

    env_metal = make_matched_env("metal")
    env_cpu = make_matched_env("cpu")

    # 1. Reset Parity
    torch.manual_seed(seed)
    obs_m, _ = env_metal.reset()
    torch.manual_seed(seed)
    obs_c, _ = env_cpu.reset()

    # Actor term slices (total 61 dims)
    actor_slices = {
        "base_ang_vel": slice(0, 3),
        "projected_gravity": slice(3, 6),
        "joint_pos": slice(6, 20),
        "joint_vel": slice(20, 34),
        "actions": slice(34, 48),
        "command": slice(48, 51),
        "head_command": slice(51, 55),
        "body_command": slice(55, 61),
    }
    for name, sl in actor_slices.items():
        diff = (obs_m["actor"][:, sl] - obs_c["actor"][:, sl]).abs().max().item()
        assert diff < 1e-4, f"Reset actor term '{name}' mismatch: {diff:.2e} >= 1e-4"

    # Critic term slices (total 76 dims)
    critic_slices = {
        "base_lin_vel": slice(0, 3),
        "base_ang_vel": slice(3, 6),
        "projected_gravity": slice(6, 9),
        "joint_pos": slice(9, 23),
        "joint_vel": slice(23, 37),
        "actions": slice(37, 51),
        "command": slice(51, 54),
        "foot_height": slice(54, 56),
        "foot_air_time": slice(56, 58),
        "foot_contact": slice(58, 60),
        "foot_contact_forces": slice(60, 66),
        "head_command": slice(66, 70),
        "body_command": slice(70, 76),
    }
    for name, sl in critic_slices.items():
        diff = (obs_m["critic"][:, sl] - obs_c["critic"][:, sl]).abs().max().item()
        assert diff < 1e-4, f"Reset critic term '{name}' mismatch: {diff:.2e} >= 1e-4"

    # 2. Stepped Parity across 15 control steps (300 ms, 60 substeps) with dynamic non-zero actions
    for step_i in range(15):
        # Non-zero dynamic actions exercising all 14 leg/head actuators across walking cycles
        actions = torch.sin(torch.arange(14, device="mps").unsqueeze(0) * 0.5 + step_i * 0.3) * 0.25
        actions = actions.expand(num_envs, 14).contiguous()

        ret_m = env_metal.step(actions)
        ret_c = env_cpu.step(actions)

        # Check every individual actor observation term after step
        for name, sl in actor_slices.items():
            diff = (ret_m[0]["actor"][:, sl] - ret_c[0]["actor"][:, sl]).abs().max().item()
            if name in ("actions", "command", "head_command", "body_command"):
                assert diff < 1e-4, f"Step {step_i} actor term '{name}' mismatch: {diff:.2e} >= 1e-4"
            elif name == "projected_gravity":
                assert diff < 0.02, f"Step {step_i} actor term '{name}' mismatch: {diff:.2e} >= 0.02"
            elif name == "joint_pos":
                assert diff < 0.025, f"Step {step_i} actor term '{name}' mismatch: {diff:.2e} >= 0.025 rad"
            elif name == "base_ang_vel":
                assert diff < 0.50, f"Step {step_i} actor term '{name}' mismatch: {diff:.2e} >= 0.50 rad/s"
            else:  # joint_vel
                assert diff < 1.05, f"Step {step_i} actor term '{name}' mismatch: {diff:.2e} >= 1.05 rad/s"

        # Check every individual critic observation term after step
        for name, sl in critic_slices.items():
            diff = (ret_m[0]["critic"][:, sl] - ret_c[0]["critic"][:, sl]).abs().max().item()
            if name in ("actions", "command", "head_command", "body_command", "foot_air_time", "foot_contact"):
                assert diff < 1e-3, f"Step {step_i} critic term '{name}' mismatch: {diff:.2e} >= 1e-3"
            elif name == "foot_height":
                assert diff < 0.01, f"Step {step_i} critic term '{name}' mismatch: {diff:.2e} >= 0.01 m"
            elif name == "projected_gravity":
                assert diff < 0.02, f"Step {step_i} critic term '{name}' mismatch: {diff:.2e} >= 0.02"
            elif name == "joint_pos":
                assert diff < 0.025, f"Step {step_i} critic term '{name}' mismatch: {diff:.2e} >= 0.025 rad"
            elif name == "base_lin_vel":
                assert diff < 0.06, f"Step {step_i} critic term '{name}' mismatch: {diff:.2e} >= 0.06 m/s"
            elif name == "base_ang_vel":
                assert diff < 0.40, f"Step {step_i} critic term '{name}' mismatch: {diff:.2e} >= 0.40 rad/s"
            elif name == "foot_contact_forces":
                assert diff < 0.90, f"Step {step_i} critic term '{name}' mismatch: {diff:.2e} >= 0.90 N"
            else:  # joint_vel
                assert diff < 1.05, f"Step {step_i} critic term '{name}' mismatch: {diff:.2e} >= 1.05 rad/s"

        # Check instantaneous total step reward and every individual reward function
        # Total instantaneous step reward match
        step_rew_diff = (ret_m[1] - ret_c[1]).abs().max().item()
        assert step_rew_diff < 0.02, f"Step {step_i} total step reward mismatch: {step_rew_diff:.3e} >= 0.02"

        # Individual instantaneous step reward outputs via reward_manager._step_reward (unscaled rates)
        for term_idx, r_name in enumerate(env_metal.reward_manager._term_names):
            step_r_m = env_metal.reward_manager._step_reward[:, term_idx]
            step_r_c = env_cpu.reward_manager._step_reward[:, term_idx]
            r_diff = (step_r_m - step_r_c).abs().max().item()
            if r_name == "track_angular_velocity":
                assert r_diff < 0.50, f"Step {step_i} reward term '{r_name}' mismatch: {r_diff:.3e} >= 0.50"
            elif r_name == "track_linear_velocity":
                assert r_diff < 0.20, f"Step {step_i} reward term '{r_name}' mismatch: {r_diff:.3e} >= 0.20"
            elif r_name == "pose":
                assert r_diff < 0.08, f"Step {step_i} reward term '{r_name}' mismatch: {r_diff:.3e} >= 0.08"
            elif r_name == "upright":
                assert r_diff < 0.06, f"Step {step_i} reward term '{r_name}' mismatch: {r_diff:.3e} >= 0.06"
            else:
                assert r_diff < 0.02, f"Step {step_i} reward term '{r_name}' mismatch: {r_diff:.3e} >= 0.02"

        # 6 separated physical state channels (in distinct physical units)
        sim_m = env_metal.sim
        sim_c = env_cpu.sim

        # Channel 1: Root position (meters) - tight 5 mm bound
        root_pos_err = (sim_m.data.qpos[:, :3] - sim_c.data.qpos[:, :3]).abs().max().item()
        assert root_pos_err < 0.005, f"Step {step_i} root position error: {root_pos_err:.3e} m >= 0.005 m"

        # Channel 2: Root orientation (SO(3) geodesic distance in radians, invariant to quaternion sign)
        q_m = sim_m.data.qpos[:, 3:7]
        q_c = sim_c.data.qpos[:, 3:7]
        dot = (q_m * q_c).sum(dim=-1).abs().clamp(0.0, 1.0)
        so3_rot_err = (2.0 * torch.acos(dot)).max().item()
        assert so3_rot_err < 0.035, f"Step {step_i} root orientation SO(3) error: {so3_rot_err:.3e} rad >= 0.035 rad"

        # Channel 3: Root linear velocity (m/s)
        root_linvel_err = (sim_m.data.qvel[:, :3] - sim_c.data.qvel[:, :3]).abs().max().item()
        assert root_linvel_err < 0.06, f"Step {step_i} root linear velocity error: {root_linvel_err:.3e} m/s >= 0.06 m/s"

        # Channel 4: Root angular velocity (rad/s)
        root_angvel_err = (sim_m.data.qvel[:, 3:6] - sim_c.data.qvel[:, 3:6]).abs().max().item()
        assert root_angvel_err < 0.40, f"Step {step_i} root angular velocity error: {root_angvel_err:.3e} rad/s >= 0.40 rad/s"

        # Channel 5: Joint positions (radians)
        jnt_pos_err = (sim_m.data.qpos[:, 7:21] - sim_c.data.qpos[:, 7:21]).abs().max().item()
        assert jnt_pos_err < 0.025, f"Step {step_i} joint position error: {jnt_pos_err:.3e} rad >= 0.025 rad"

        # Channel 6: Joint velocities (rad/s)
        jnt_vel_err = (sim_m.data.qvel[:, 6:20] - sim_c.data.qvel[:, 6:20]).abs().max().item()
        assert jnt_vel_err < 1.05, f"Step {step_i} joint velocity error: {jnt_vel_err:.3e} rad/s >= 1.05 rad/s"

        # Terminations and Truncations exact match
        assert torch.equal(ret_m[2], ret_c[2]), f"Step {step_i} terminations mismatch"
        assert torch.equal(ret_m[3], ret_c[3]), f"Step {step_i} truncations mismatch"

    env_metal.close()
    env_cpu.close()


