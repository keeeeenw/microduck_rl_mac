"""Train the canonical flat task on Mac with MPS learning and selectable physics."""

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import sys
import time
import types


def atomic_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-interval", type=int, default=25)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--verify-backend-only", action="store_true", help="CPU-only source path preflight")
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--physics",
        choices=("mps", "cpu", "metal", "unified_metal"),
        default="cpu",
        help="Physics backend (default: cpu; metal is experimental). Policy inference and PPO always use MPS.",
    )
    args = parser.parse_args()
    if args.verify_backend_only:
        from .backend_selection import resolve_metal_backend
        _, paths = resolve_metal_backend()
        print(json.dumps(paths, indent=2, sort_keys=True))
        return
    if args.log_dir is None:
        parser.error("--log-dir is required for training")
    if args.physics in ("metal", "unified_metal"):
        from .backend_selection import resolve_metal_backend
        _, backend_source_paths = resolve_metal_backend()
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            parser.error("Experimental Metal physics requires an Apple Silicon Mac")
        print("Experimental Metal physics: flat walking only; broader validation remains open.", file=sys.stderr)
    else:
        backend_source_paths = None
    if min(args.num_envs, args.iterations, args.save_interval) < 1:
        parser.error("Environment count, iterations and save interval must be positive")
    os.environ["JAX_PLATFORMS"] = "mps" if args.physics == "mps" else "cpu"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    os.environ.setdefault("WANDB_MODE", "disabled")
    directory = args.log_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "status.json").exists():
        parser.error("Use a new log directory; an existing run must not be overwritten")
    status_path = directory / "status.json"
    status = {
        "status": "initializing",
        "pid": os.getpid(),
        "started_at": time.time(),
        "platform": platform.platform(),
        "num_envs": args.num_envs,
        "seed": args.seed,
        "requested_iterations": args.iterations,
        "completed_iterations": 0,
        "task": "Mjlab-Velocity-Flat-MicroDuck",
        "device": "mps",
        "physics_backend": args.physics,
        "transfer_mode": (
            "CPU physics / MPS managers, BAM and PPO"
            if args.physics == "cpu"
            else (
                "Device-resident dynamics with CPU narrowphase fallback and reset-time constant synchronization"
                if args.physics in ("metal", "unified_metal")
                else "host-staged copies between GPU frameworks; no CPU numerics"
            )
        ),
    }
    atomic_json(status_path, status)
    source_root = Path(__file__).resolve().parents[1]
    repository = source_root.parents[1]
    source_sha256 = {
        str(p.relative_to(repository)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(source_root.rglob("*.py"))
    }
    if backend_source_paths:
        # Hash package resources as well as Python sources. No sibling repository
        # is needed, including when this package is installed from a wheel.
        for p in sorted((source_root / "native_gpu" / "metal").rglob("*")):
            if p.is_file() and p.suffix in (".metal", ".xml"):
                source_sha256[str(p.relative_to(repository))] = hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = {
        "command": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "source_sha256": source_sha256,
        "backend_source_paths": backend_source_paths,
        "lock_sha256": hashlib.sha256(
            (repository / "uv.lock").read_bytes()
        ).hexdigest() if (repository / "uv.lock").is_file() else None,
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "jax",
                "jaxlib",
                "jax-mps",
                "mujoco",
                "mujoco-mjx",
                "mjlab",
                "rsl-rl-lib",
            )
        },
        "resume": str(args.resume.resolve()) if args.resume else None,
        "resume_sha256": hashlib.sha256(args.resume.read_bytes()).hexdigest()
        if args.resume
        else None,
    }
    atomic_json(directory / "manifest.json", manifest)
    try:
        import torch
        import warp
        import mjlab_microduck.tasks  # noqa: F401 - populate task registry.
        from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
        from mjlab.rl import RslRlVecEnvWrapper
        from mjlab_microduck.export import export_runner_policy
        from .environment import MetalEnv

        def no_warp_kernels(*_args, **_kwargs):
            raise RuntimeError(
                "Warp kernel execution is disabled in the native GPU profile"
            )

        warp.launch = no_warp_kernels
        env_cfg = load_env_cfg(status["task"])
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.seed = args.seed
        agent = dataclasses.asdict(load_rl_cfg(status["task"]))
        agent.update(
            seed=args.seed,
            logger="tensorboard",
            upload_model=False,
            save_interval=args.save_interval,
        )
        atomic_json(directory / "agent.json", agent)
        status["stage"] = "constructing_environment"
        atomic_json(status_path, status)
        env = MetalEnv(env_cfg, physics=args.physics)
        status["stage"] = "resetting_environment"
        atomic_json(status_path, status)
        wrapper = RslRlVecEnvWrapper(env, clip_actions=agent["clip_actions"])
        status["stage"] = "constructing_runner"
        atomic_json(status_path, status)
        runner = load_runner_cls(status["task"])(
            wrapper, agent, log_dir=str(directory), device="mps"
        )
        if args.resume:
            runner.load(str(args.resume.resolve()), map_location="cpu")
            saved = torch.load(
                args.resume.resolve(), map_location="cpu", weights_only=False
            )
            native = saved.get("native_gpu", {})
            if (
                native.get("schema") == 2
                and native.get("physics_backend", "mps") == args.physics
                and native.get("environment_layout") == "no-origin-markers-v1"
            ):
                from .checkpoint import restore

                restore(env, native["environment"])
                status["resume_kind"] = "full_environment_and_learner"
            else:
                status["resume_kind"] = "learner_and_progress_with_fresh_episodes"

        obs = wrapper.get_observations()
        assert obs["actor"].shape == (args.num_envs, 61)
        assert wrapper.num_actions == 14
        for name, value in obs.items():
            if value.device.type != "mps" or not torch.isfinite(value).all():
                raise RuntimeError(f"Invalid initial GPU observation {name}")
        status.update(
            status="training",
            actor_width=61,
            critic_width=obs["critic"].shape[-1],
            completed_iterations=runner.current_learning_iteration,
        )
        atomic_json(status_path, status)

        original_optimizer_step = runner.alg.optimizer.step

        def checked_optimizer_step(*values, **kwargs):
            gradients = [
                p.grad
                for group in runner.alg.optimizer.param_groups
                for p in group["params"]
                if p.grad is not None
            ]
            if (
                not gradients
                or not torch.stack([torch.isfinite(g).all() for g in gradients]).all()
            ):
                raise RuntimeError("Missing or nonfinite PPO gradients")
            return original_optimizer_step(*values, **kwargs)

        runner.alg.optimizer.step = checked_optimizer_step

        original_step = env.step

        def monitored_step(action):
            start = time.perf_counter()
            if not torch.isfinite(action).all():
                raise RuntimeError("Nonfinite rollout action")
            result = original_step(action)
            if not torch.isfinite(result[1]).all():
                raise RuntimeError("Nonfinite rollout reward")
            status.update(
                stage="rollout",
                control_steps_completed=env.common_step_counter,
                last_step_seconds=time.perf_counter() - start,
                updated_at=time.time(),
            )
            atomic_json(status_path, status)
            return result

        env.step = monitored_step
        original_save = runner.save

        def save(_runner, path, infos=None):
            path = Path(path)
            temporary = path.with_suffix(".partial.pt")
            original_save(str(temporary), infos)
            saved = torch.load(temporary, map_location="cpu", weights_only=False)
            # The base runner stores the index just completed. Resume at the next
            # iteration so iteration counters and curricula do not repeat a step.
            saved["iter"] = _runner.current_learning_iteration + 1
            from .checkpoint import capture

            saved["native_gpu"] = {
                "schema": 2,
                "environment_layout": "no-origin-markers-v1",
                "physics_backend": args.physics,
                "environment": capture(env),
            }
            torch.save(saved, temporary)
            temporary.replace(path)
            status["checkpoint"] = str(path)
            atomic_json(status_path, status)

        runner.save = types.MethodType(save, runner)

        original_log = runner.logger.log

        def log(*values, **metrics):
            original_log(*values, **metrics)
            loss = metrics["loss_dict"]
            if not all(torch.isfinite(torch.tensor(v)) for v in loss.values()):
                raise RuntimeError(f"Nonfinite PPO loss: {loss}")
            elapsed = metrics["collect_time"] + metrics["learn_time"]
            record = {
                "iteration": metrics["it"] + 1,
                "time": time.time(),
                "collection_seconds": metrics["collect_time"],
                "learning_seconds": metrics["learn_time"],
                "transitions_per_second": args.num_envs
                * agent["num_steps_per_env"]
                / elapsed,
                "loss": loss,
                "learning_rate": metrics["learning_rate"],
                "mean_reward": statistics.mean(runner.logger.rewbuffer)
                if runner.logger.rewbuffer
                else None,
                "mean_episode_length": statistics.mean(runner.logger.lenbuffer)
                if runner.logger.lenbuffer
                else None,
                "transfer_bytes_total": env.sim.transfer.bytes,
                "transfer_seconds_total": env.sim.transfer.seconds,
                "reset_transfer_bytes_total": getattr(getattr(env.sim, "reset_transfer", None), "bytes", 0),
                "reset_transfer_seconds_total": getattr(getattr(env.sim, "reset_transfer", None), "seconds", 0.0),
                "collision_qpos_staging_bytes_total": getattr(env.sim, "collision_qpos_staging_bytes", 0),
                "collision_fric_staging_bytes_total": getattr(env.sim, "collision_fric_staging_bytes", 0),
                "collision_transfer_bytes_total": getattr(getattr(env.sim, "collision_transfer", None), "bytes", 0),
                "collision_transfer_seconds_total": getattr(getattr(env.sim, "collision_transfer", None), "seconds", 0.0),
                "collision_narrowphase_seconds_total": getattr(env.sim, "collision_narrowphase_seconds", 0.0),
                "collision_fallback_branch_seconds_total": getattr(env.sim, "collision_fallback_branch_seconds", 0.0),
                "collision_evaluations_count": getattr(env.sim, "collision_evaluations_count", 0),
                "constants_recomputed_envs_total": getattr(env.sim, "constants_recomputed_envs", 0),
                "constants_cache_skips_total": getattr(env.sim, "constants_cache_skips", 0),
                "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            }
            with (directory / "progress.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
                stream.flush()
            status.update(
                status="training",
                completed_iterations=record["iteration"],
                latest=record,
            )
            atomic_json(status_path, status)
            # Establish a recovery point after the first update in a new run,
            # even when the regular checkpoint interval has not been reached.
            if "checkpoint" not in status and metrics["it"] % args.save_interval:
                runner.save(str(directory / f"model_{metrics['it']}.pt"))

        runner.logger.log = log
        runner.learn(args.iterations, init_at_random_ep_len=not bool(args.resume))
        checkpoint = status["checkpoint"]
        onnx = export_runner_policy(runner, str(directory / "policy.onnx"), checkpoint)
        import onnxruntime as ort

        session = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
        actual_obs = wrapper.get_observations()
        with torch.inference_mode():
            expected = runner.alg.actor(actual_obs).cpu().numpy()
        actual = session.run(
            None, {session.get_inputs()[0].name: actual_obs["actor"][:1].cpu().numpy()}
        )[0]
        import numpy as np

        np.testing.assert_allclose(actual, expected[:1], rtol=2e-4, atol=2e-5)
        status.update(
            status="completed",
            finished_at=time.time(),
            onnx=str(onnx),
            onnx_sha256=hashlib.sha256(onnx.read_bytes()).hexdigest(),
            export_parity=True,
        )
        atomic_json(status_path, status)
        env.close()
    except BaseException as exc:
        status.update(
            status="failed", error=f"{type(exc).__name__}: {exc}", failed_at=time.time()
        )
        atomic_json(status_path, status)
        raise


if __name__ == "__main__":
    main()
