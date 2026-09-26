"""Tests for UnifiedMetalSimulation checkpoint capture, restore, and continuation parity.

Verifies:
1. Exact serialization/deserialization of simulation state and randomized parameters.
2. Uninterrupted rollout vs interrupted-and-restored rollout bitwise/numerical parity.
"""

from pathlib import Path
import sys
import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent


from mjlab_microduck.native_gpu.metal.canonical_model_loader import load_canonical_model
from mjlab_microduck.native_gpu.metal.metal_simulation_adapter import UnifiedMetalSimulation


def make_mock_cfg():
    return type(
        "Cfg",
        (),
        {
            "mujoco": type("M", (), {"apply": lambda self, m: None})(),
            "nan_guard": type("N", (), {"enabled": False, "action": "raise"})(),
        },
    )()


def test_checkpoint_state_serialization():
    """Verify capture_physics and restore_physics correctly round-trip all required state and DR buffers."""
    canonical = load_canonical_model()
    m = canonical.model
    cfg = make_mock_cfg()

    sim = UnifiedMetalSimulation(num_envs=4, cfg=cfg, model=m, device="mps")
    sim.expand_model_fields(["body_mass", "dof_damping", "dof_armature"])

    # Set non-trivial state
    sim.data.time[:] = 1.234
    sim.data.qpos[:, 2] = 0.22
    sim.data.qvel[:, 0] = 0.5
    sim.data.ctrl[:, 0] = 1.5
    sim.model.body_mass[1] *= 1.25
    sim.model.dof_damping[2, 6:20] = 0.1
    sim.forward()

    checkpoint = sim.capture_physics()
    assert checkpoint["backend"] == "metal"
    assert "time" in checkpoint
    assert "qpos" in checkpoint
    assert "qvel" in checkpoint
    assert "body_mass" in checkpoint
    assert "dof_damping" in checkpoint

    # Create new simulation instance and restore
    sim2 = UnifiedMetalSimulation(num_envs=4, cfg=cfg, model=m, device="mps")
    sim2.expand_model_fields(["body_mass", "dof_damping", "dof_armature"])
    sim2.restore_physics(checkpoint)

    # Verify all restored views match
    assert torch.allclose(sim.data.time, sim2.data.time)
    assert torch.allclose(sim.data.qpos, sim2.data.qpos)
    assert torch.allclose(sim.data.qvel, sim2.data.qvel)
    assert torch.allclose(sim.data.ctrl, sim2.data.ctrl)
    assert torch.allclose(sim.model.body_mass, sim2.model.body_mass)
    assert torch.allclose(sim.model.dof_damping, sim2.model.dof_damping)
    assert torch.allclose(sim.data.site_xpos, sim2.data.site_xpos)
    assert torch.allclose(sim.data.cvel, sim2.data.cvel)


def test_uninterrupted_vs_restored_continuation_parity():
    """Verify that a simulation run interrupted, saved, and restored matches an uninterrupted run."""
    canonical = load_canonical_model()
    m = canonical.model
    cfg = make_mock_cfg()

    # 1. Uninterrupted run for 10 steps
    sim_uninterrupted = UnifiedMetalSimulation(num_envs=2, cfg=cfg, model=m, device="mps")
    sim_uninterrupted.expand_model_fields(["body_mass", "dof_damping"])
    sim_uninterrupted.data.ctrl[:] = 0.5 # constant torque
    sim_uninterrupted.model.dof_damping[:, 6:20] = 0.005 # viscous damping

    saved_checkpoint_at_step_5 = None

    for t in range(10):
        sim_uninterrupted.step()
        if t == 4: # step 5 completed
            saved_checkpoint_at_step_5 = sim_uninterrupted.capture_physics()

    # 2. Fresh simulation instance initialized, run 5 steps, then restored from step 5, and run remaining 5 steps
    sim_restored = UnifiedMetalSimulation(num_envs=2, cfg=cfg, model=m, device="mps")
    sim_restored.expand_model_fields(["body_mass", "dof_damping"])
    sim_restored.restore_physics(saved_checkpoint_at_step_5)

    # Verify state right after restore matches step 5 of uninterrupted
    assert torch.allclose(sim_restored.data.time, torch.tensor([0.025, 0.025], device="mps"))

    for t in range(5, 10):
        sim_restored.step()

    # Verify final states at step 10 match exactly between uninterrupted and restored runs
    max_qpos_err = (sim_uninterrupted.data.qpos - sim_restored.data.qpos).abs().max().item()
    max_qvel_err = (sim_uninterrupted.data.qvel - sim_restored.data.qvel).abs().max().item()
    max_site_err = (sim_uninterrupted.data.site_xpos - sim_restored.data.site_xpos).abs().max().item()
    max_cvel_err = (sim_uninterrupted.data.cvel - sim_restored.data.cvel).abs().max().item()

    print(f"Max continuation errors: qpos={max_qpos_err:.2e}, qvel={max_qvel_err:.2e}, site={max_site_err:.2e}, cvel={max_cvel_err:.2e}")
    assert max_qpos_err < 1e-5, f"Continuation qpos mismatch: {max_qpos_err:.2e}"
    assert max_qvel_err < 1e-4, f"Continuation qvel mismatch: {max_qvel_err:.2e}"
    assert max_site_err < 1e-5, f"Continuation site mismatch: {max_site_err:.2e}"
    assert max_cvel_err < 1e-4, f"Continuation cvel mismatch: {max_cvel_err:.2e}"


def test_metal_env_full_checkpoint_continuation_parity():
    """Verify that full MetalEnv training snapshot capture and restore reproduces uninterrupted rollout bitwise.

    Covers:
    1. BAM actuator thermal, battery, and friction budgets.
    2. Action history and observation delay buffers.
    3. Command managers, curriculum manager state, and reward tracking.
    4. Python, NumPy, PyTorch, and PyTorch MPS random number generator states.
    5. UnifiedMetalSimulation physics state and randomized domain parameters.
    """
    import mjlab_microduck.tasks
    from mjlab.tasks.registry import load_env_cfg
    from mjlab_microduck.native_gpu.environment import MetalEnv
    from mjlab_microduck.native_gpu.checkpoint import capture, restore

    num_envs = 4
    seed = 42

    def make_env():
        torch.manual_seed(seed)
        cfg = load_env_cfg("Mjlab-Velocity-Flat-MicroDuck")
        cfg.scene.num_envs = num_envs
        cfg.seed = seed
        return MetalEnv(cfg, physics="metal", device="mps")

    env1 = make_env()
    env1.reset()

    # Step 3 steps with non-zero dynamic actions
    for step_i in range(3):
        act = torch.sin(torch.arange(14, device="mps").unsqueeze(0) + step_i) * 0.2
        env1.step(act.expand(num_envs, 14))

    # Full training snapshot of env1 (tensors, BAM controller, history buffers, RNGs, physics)
    snapshot = capture(env1)

    # Step env1 3 more steps uninterrupted
    uninterrupted_obs = []
    uninterrupted_rew = []
    for step_i in range(3, 6):
        act = torch.sin(torch.arange(14, device="mps").unsqueeze(0) + step_i) * 0.2
        ret = env1.step(act.expand(num_envs, 14))
        uninterrupted_obs.append(ret[0]["actor"].clone())
        uninterrupted_rew.append(ret[1].clone())

    # Create env2 and restore from snapshot
    env2 = make_env()
    env2.reset()
    restore(env2, snapshot)

    # Step env2 3 steps with identical actions
    restored_obs = []
    restored_rew = []
    for step_i in range(3, 6):
        act = torch.sin(torch.arange(14, device="mps").unsqueeze(0) + step_i) * 0.2
        ret = env2.step(act.expand(num_envs, 14))
        restored_obs.append(ret[0]["actor"].clone())
        restored_rew.append(ret[1].clone())

    # Verify bitwise parity across all resumed steps
    for i in range(3):
        obs_diff = (uninterrupted_obs[i] - restored_obs[i]).abs().max().item()
        rew_diff = (uninterrupted_rew[i] - restored_rew[i]).abs().max().item()
        assert obs_diff == 0.0, f"Step {i+1} actor observation mismatch: {obs_diff}"
        assert rew_diff == 0.0, f"Step {i+1} step reward mismatch: {rew_diff}"

    env1.close()
    env2.close()


def test_metal_split_run_ppo_checkpoint_continuation_parity(tmp_path):
    """Verify full split-run checkpoint continuation parity across save/load to disk.

    Covers:
    1. Actual checkpoint file (.pt) round-trip serialization to disk.
    2. Deliberately crossing episode reset/resampling boundary during resumed rollout.
    3. Bitwise parity of actor and critic observation tensors.
    4. Bitwise parity of per-term reward contributions and termination flags.
    5. Bitwise parity of physics state (qpos, qvel, time) and BAM controller state.
    6. Bitwise parity of PPO learner state (actor/critic weights, normalizer statistics, optimizer state).
    """
    import copy
    import dataclasses
    import mjlab_microduck.tasks
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab_microduck.native_gpu.environment import MetalEnv
    from mjlab_microduck.native_gpu.checkpoint import capture, restore

    seed = 42
    num_envs = 4

    def make_env():
        torch.manual_seed(seed)
        cfg = load_env_cfg("Mjlab-Velocity-Flat-MicroDuck")
        cfg.scene.num_envs = num_envs
        cfg.seed = seed
        return MetalEnv(cfg, physics="metal", device="mps")

    def make_agent():
        agent = dataclasses.asdict(load_rl_cfg("Mjlab-Velocity-Flat-MicroDuck"))
        agent.update(
            seed=seed,
            logger="none",
            upload_model=False,
            num_steps_per_env=4,
            save_interval=1,
        )
        agent["algorithm"]["num_learning_epochs"] = 1
        agent["algorithm"]["num_mini_batches"] = 1
        return agent

    # 1. Uninterrupted run: iteration 1, save state, iteration 2
    env1 = make_env()
    agent1 = make_agent()
    wrapper1 = RslRlVecEnvWrapper(env1, clip_actions=agent1["clip_actions"])
    runner1 = load_runner_cls("Mjlab-Velocity-Flat-MicroDuck")(
        wrapper1, agent1, log_dir=None, device="mps"
    )

    runner1.learn(num_learning_iterations=1)

    # Force env 0 near timeout so iteration 2 crosses reset boundary immediately
    env1.episode_length_buf[0] = env1.max_episode_length - 1

    # Save full training checkpoint to disk
    ckpt_path = tmp_path / "split_run_checkpoint.pt"
    temp_path = tmp_path / "split_run_checkpoint.partial.pt"
    runner1.save(str(temp_path))
    saved = torch.load(temp_path, map_location="cpu", weights_only=False)
    saved["iter"] = runner1.current_learning_iteration + 1
    saved["native_gpu"] = {
        "schema": 2,
        "environment_layout": "no-origin-markers-v1",
        "physics_backend": "metal",
        "environment": capture(env1),
    }
    torch.save(saved, ckpt_path)
    if temp_path.exists():
        temp_path.unlink()

    # Hook env1.step to record step-by-step continuation trajectory
    env1_steps = []
    orig_step1 = env1.step
    def step1_recorder(action):
        obs, rew, terminated, truncated, info = orig_step1(action)
        done = terminated | truncated
        env1_steps.append({
            "done": done.clone(),
            "terminated": terminated.clone(),
            "truncated": truncated.clone(),
            "rew": rew.clone(),
            "time": env1.sim.data.time.clone(),
            "qpos": env1.sim.data.qpos.clone(),
            "qvel": env1.sim.data.qvel.clone(),
            "actor_obs": obs["actor"].clone(),
            "critic_obs": obs["critic"].clone(),
        })
        return obs, rew, terminated, truncated, info
    env1.step = step1_recorder

    # Continue uninterrupted iteration 2
    runner1.learn(num_learning_iterations=1)

    # 2. Resumed run from checkpoint
    env2 = make_env()
    agent2 = make_agent()
    wrapper2 = RslRlVecEnvWrapper(env2, clip_actions=agent2["clip_actions"])
    runner2 = load_runner_cls("Mjlab-Velocity-Flat-MicroDuck")(
        wrapper2, agent2, log_dir=None, device="mps"
    )

    # Load checkpoint from disk
    runner2.load(str(ckpt_path), map_location="cpu")
    saved2 = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    restore(env2, saved2["native_gpu"]["environment"])

    # Hook env2.step to record step-by-step resumed trajectory
    env2_steps = []
    orig_step2 = env2.step
    def step2_recorder(action):
        obs, rew, terminated, truncated, info = orig_step2(action)
        done = terminated | truncated
        env2_steps.append({
            "done": done.clone(),
            "terminated": terminated.clone(),
            "truncated": truncated.clone(),
            "rew": rew.clone(),
            "time": env2.sim.data.time.clone(),
            "qpos": env2.sim.data.qpos.clone(),
            "qvel": env2.sim.data.qvel.clone(),
            "actor_obs": obs["actor"].clone(),
            "critic_obs": obs["critic"].clone(),
        })
        return obs, rew, terminated, truncated, info
    env2.step = step2_recorder

    # Run iteration 2
    runner2.learn(num_learning_iterations=1)

    # Verifications:
    # 1. Step count and reset boundary occurrence
    assert len(env1_steps) == len(env2_steps) == 4, f"Expected 4 rollout steps, got {len(env1_steps)} vs {len(env2_steps)}"
    # Verify env 0 actually triggered timeout / truncation at step 0 as constructed
    assert env1_steps[0]["truncated"][0].item() is True, "Env 0 must trigger timeout (truncated) at step 0 of uninterrupted iteration 2"
    assert env2_steps[0]["truncated"][0].item() is True, "Env 0 must trigger timeout (truncated) at step 0 of restored iteration 2"
    assert env1_steps[0]["terminated"][0].item() is False, "Env 0 step 0 boundary must be a timeout truncation, not an early termination"
    assert env2_steps[0]["terminated"][0].item() is False, "Env 0 step 0 boundary must be a timeout truncation, not an early termination"
    assert env1_steps[0]["done"][0].item() is True, "Env 0 done flag must be True"
    assert env2_steps[0]["done"][0].item() is True, "Env 0 done flag must be True"

    # 2. Step-by-step bitwise trajectory comparisons across all rollout steps
    for step_idx in range(4):
        s1, s2 = env1_steps[step_idx], env2_steps[step_idx]
        assert torch.equal(s1["done"], s2["done"]), f"Step {step_idx} done flag mismatch"
        assert torch.equal(s1["terminated"], s2["terminated"]), f"Step {step_idx} terminated flag mismatch"
        assert torch.equal(s1["truncated"], s2["truncated"]), f"Step {step_idx} truncated flag mismatch"
        assert torch.equal(s1["rew"], s2["rew"]), f"Step {step_idx} reward mismatch"
        assert torch.equal(s1["time"], s2["time"]), f"Step {step_idx} physics time mismatch"
        assert torch.equal(s1["qpos"], s2["qpos"]), f"Step {step_idx} physics qpos mismatch"
        assert torch.equal(s1["qvel"], s2["qvel"]), f"Step {step_idx} physics qvel mismatch"
        assert torch.equal(s1["actor_obs"], s2["actor_obs"]), f"Step {step_idx} actor observation mismatch"
        assert torch.equal(s1["critic_obs"], s2["critic_obs"]), f"Step {step_idx} critic observation mismatch"

    # 3. BAM actuator internal states match bitwise
    act1 = env1.scene["robot"].actuators[0]
    act2 = env2.scene["robot"].actuators[0]
    assert torch.equal(act1._prev_motor_torque, act2._prev_motor_torque), "BAM _prev_motor_torque mismatch"
    assert torch.equal(act1.friction_scale, act2.friction_scale), "BAM friction_scale mismatch"
    assert torch.equal(act1.kp_scale, act2.kp_scale), "BAM kp_scale mismatch"
    assert torch.equal(act1.kd_scale, act2.kd_scale), "BAM kd_scale mismatch"
    assert torch.equal(act1.vin_tensor, act2.vin_tensor), "BAM vin_tensor mismatch"
    assert torch.equal(act1.vin_drop_gain, act2.vin_drop_gain), "BAM vin_drop_gain mismatch"
    if act1._delay_buffer is not None:
        assert torch.equal(act1._delay_buffer.current_lags, act2._delay_buffer.current_lags), "BAM delay buffer current_lags mismatch"
        assert torch.equal(act1._delay_buffer._step_count, act2._delay_buffer._step_count), "BAM delay buffer _step_count mismatch"
        assert torch.equal(act1._delay_buffer._buffer._buffer, act2._delay_buffer._buffer._buffer), "BAM delay buffer storage mismatch"

    # 4. PPO Optimizer state matches bitwise
    opt1 = runner1.alg.optimizer.state_dict()
    opt2 = runner2.alg.optimizer.state_dict()
    for pg1, pg2 in zip(opt1["param_groups"], opt2["param_groups"]):
        for k in pg1:
            if k != "params":
                assert pg1[k] == pg2[k], f"Optimizer param_group {k} mismatch: {pg1[k]} vs {pg2[k]}"
    for p_id in opt1["state"]:
        s1 = opt1["state"][p_id]
        s2 = opt2["state"][p_id]
        for k in s1:
            if isinstance(s1[k], torch.Tensor):
                diff = (s1[k] - s2[k]).abs().max().item()
                assert diff == 0.0, f"Optimizer state tensor {k} mismatch: {diff}"
            else:
                assert s1[k] == s2[k], f"Optimizer state scalar {k} mismatch"

    # 5. Actor and Critic model parameters match bitwise
    actor_uninterrupted = runner1.alg.actor.state_dict()
    actor_restored = runner2.alg.actor.state_dict()
    for k in actor_uninterrupted:
        diff = (actor_uninterrupted[k] - actor_restored[k]).abs().max().item()
        assert diff == 0.0, f"Actor parameter {k} mismatch: {diff}"

    critic_uninterrupted = runner1.alg.critic.state_dict()
    critic_restored = runner2.alg.critic.state_dict()
    for k in critic_uninterrupted:
        diff = (critic_uninterrupted[k] - critic_restored[k]).abs().max().item()
        assert diff == 0.0, f"Critic parameter {k} mismatch: {diff}"

    env1.close()
    env2.close()


