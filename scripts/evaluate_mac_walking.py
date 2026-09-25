#!/usr/bin/env python3
"""Evaluate an exported walking policy using the existing CPU playback/BAM path.

This is a bounded nominal-flat playback assessment, not a sim2real qualification.
Optional GIF rendering uses the same rollout, without resets or altered actions.
"""

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
from pathlib import Path

import mujoco
import numpy as np

import infer_policy as playback


CASES = {
    "idle": (0.0, 0.0, 0.0),
    "forward_slow": (0.05, 0.0, 0.0),
    "forward": (0.15, 0.0, 0.0),
    "forward_fast": (0.30, 0.0, 0.0),
    "backward": (-0.10, 0.0, 0.0),
    "left": (0.0, 0.10, 0.0),
    "right": (0.0, -0.10, 0.0),
    "turn_left": (0.0, 0.0, 0.50),
    "turn_right": (0.0, 0.0, -0.50),
    "walk_stop": (0.30, 0.0, 0.0),
    "forward_turn_left": (0.30, 0.0, 0.50),
    "forward_turn_right": (0.30, 0.0, -0.50),
}


def rollout(policy_path, case, lag, seed, seconds, gif=None):
    np.random.seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        bam_model = playback.load_bam_model(200, 7.4, None)
        model, data, controller, _ = playback.load_mujoco_with_bam(
            playback.MICRODUCK_XML, bam_model, 0.005, 0.1, 6.0
        )
        policy = playback.PolicyInference(
            model, data, walking_onnx_path=str(policy_path), bam_ctrl=controller,
            new_cmd_obs=True, use_projected_gravity=True,
            delay_min_lag=lag, delay_max_lag=lag,
        )
        policy.set_vel_cmd(*CASES[case])
    data.qpos[:3] = [0, 0, 0.125]
    # Small reproducible perturbations; seed zero is the nominal playback pose.
    rng = np.random.default_rng(seed)
    roll, pitch = rng.uniform(-math.radians(2), math.radians(2), 2) if seed else (0, 0)
    data.qpos[3:7] = [math.cos(roll/2)*math.cos(pitch/2),
                      math.sin(roll/2)*math.cos(pitch/2),
                      math.cos(roll/2)*math.sin(pitch/2),
                      -math.sin(roll/2)*math.sin(pitch/2)]
    noise = rng.uniform(-0.01, 0.01, 14) if seed else np.zeros(14)
    data.qpos[policy.joint_qpos_indices] = policy.default_pose + noise
    controller.reset(data.qpos)
    mujoco.mj_forward(model, data)
    renderer = None
    frames = []
    if gif:
        from PIL import Image, ImageDraw
        renderer = mujoco.Renderer(model, height=400, width=640)
        camera = mujoco.MjvCamera()
        camera.distance = 0.85
        camera.azimuth = 125
        camera.elevation = -16
    samples, fall_time = [], None
    min_height, max_tilt = float("inf"), 0.0
    origin = data.qpos[:2].copy()
    previous_xy = origin.copy()
    distance = 0.0
    command = np.array(CASES[case])
    try:
        for step in range(round(seconds / 0.02)):
            if case == "walk_stop" and step == round(seconds / 0.04):
                command = np.zeros(3)
                with contextlib.redirect_stdout(io.StringIO()):
                    policy.set_vel_cmd(*command)
            action = policy.infer()
            playback.step_policy_physics(model, data, policy, action, controller)
            # Measure the integrated root pose without refreshing simulation caches:
            # playback's observation timing must remain unchanged.
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, data.qpos[3:7])
            rotation = rotation.reshape(3, 3)
            local_v = rotation.T @ data.qvel[:3]
            velocity = np.array([local_v[0], local_v[1], data.qvel[5]])
            tilt = math.degrees(math.acos(np.clip(rotation[2, 2], -1, 1)))
            height = float(data.qpos[2])
            min_height, max_tilt = min(min_height, height), max(max_tilt, tilt)
            distance += float(np.linalg.norm(data.qpos[:2] - previous_xy))
            previous_xy = data.qpos[:2].copy()
            if data.time >= 2.0:
                samples.append((float(data.time), *velocity, *command))
            if renderer and 2.0 <= data.time < 10.0 and step % 2 == 1:
                camera.lookat[:] = [data.qpos[0], data.qpos[1], 0.13]
                renderer.update_scene(data, camera=camera)
                frame = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 640, 43), fill=(18, 26, 37))
                draw.text((12, 7), "Microduck | Mac-trained walking policy | CPU playback", fill="white")
                draw.text((12, 24), f"Command {command[0]:.2f} m/s | delay {lag*5} ms | real time", fill=(185, 212, 230))
                frames.append(frame)
            if (not np.isfinite(data.qpos).all() or not np.isfinite(action).all()
                    or height < 0.065 or tilt > 60):
                fall_time = float(data.time)
                break
    finally:
        if renderer:
            renderer.close()
    values = np.asarray(samples).reshape(-1, 7)
    measured = values[:, 1:4]
    result = {
        "case": case, "seed": seed, "actuator_delay_steps": lag,
        "initial_command": list(CASES[case]),
        "duration_s": round(float(data.time), 3), "fall_time_s": fall_time,
        "min_root_height_m": min_height, "max_tilt_deg": max_tilt,
        "displacement_xy_m": (data.qpos[:2] - origin).tolist(),
        "path_length_m": distance,
        "mean_body_velocity": measured.mean(axis=0).tolist() if len(values) else None,
        "tracking_rmse": np.sqrt(((measured-values[:, 4:7])**2).mean(axis=0)).tolist() if len(values) else None,
    }
    if case == "walk_stop" and len(values):
        result["stopped_mean_body_velocity"] = measured[values[:, 0] > seconds/2+2].mean(axis=0).tolist()
    if frames:
        gif.parent.mkdir(parents=True, exist_ok=True)
        # Keep README media small; preserve playback speed when reducing frame rate.
        compact = [frame.resize((512, 320)).quantize(colors=96) for frame in frames[::2]]
        compact[0].save(gif, save_all=True, append_images=compact[1:], duration=80, loop=0, optimize=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--lags", nargs="+", type=int, default=[3, 4, 5, 6])
    parser.add_argument("--gif", type=Path, help="Render nominal forward_fast, seconds 2–10 at 12.5 fps")
    args = parser.parse_args()
    args.policy = args.policy.resolve()
    args.output = args.output.resolve()
    args.gif = args.gif.resolve() if args.gif else None
    os.chdir(Path(__file__).resolve().parents[1])
    results = []
    for case in args.cases:
        for seed, lag in enumerate(args.lags):
            gif = args.gif if case == "forward_fast" and seed == 0 else None
            result = rollout(args.policy, case, lag, seed, args.seconds, gif)
            results.append(result)
            print(json.dumps(result), flush=True)
    report = {
        "policy_sha256": hashlib.sha256(args.policy.read_bytes()).hexdigest(),
        "mujoco_version": mujoco.__version__,
        "policy_rate_hz": 50, "physics_rate_hz": 200,
        "bam": {"model": "m6", "vin": 7.4, "voltage_drop_gain": 0.1, "minimum_vin": 6.0},
        "scope": "Nominal flat CPU playback with ONNX inference and BAM M6; no training-distribution randomization",
        "fall_rule": "root height < 0.065 m, trunk tilt > 60 degrees, or nonfinite state/action; no autoreset",
        "measurement": "body-frame vx, vy (m/s), local-root wz (rad/s); first 2 s excluded from averages",
        "initialization": "seed 0 nominal; others roll/pitch +/-2 deg and joints +/-0.01 rad; seed and lag paired",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
