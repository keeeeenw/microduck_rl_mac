"""Bounded, auditable native-GPU physics qualification (not an RL trainer)."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time


def _write(path, report):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)


def audit_hlo(hlo):
    forbidden = [
        op for op in ("stablehlo.cholesky", "stablehlo.triangular_solve") if op in hlo
    ]
    calls = re.findall(r'stablehlo.custom_call\s+@(?:"([^"]+)"|([\w.]+))', hlo)
    if hlo.count("stablehlo.custom_call") != len(calls):
        raise RuntimeError("Unrecognized custom-call syntax; cannot audit backend")
    targets = sorted({quoted or plain for quoted, plain in calls})
    # This plugin target implements Threefry in a fused Metal kernel.
    unknown = sorted(set(targets) - {"mps.threefry2x32"})
    if forbidden or unknown:
        raise RuntimeError(f"Unqualified backend operations: {forbidden + unknown}")
    return {
        "bytes": len(hlo.encode()),
        "custom_calls": targets,
        "forbidden_operations": forbidden,
    }


def _worker(args):
    report = json.loads(args.output.read_text())

    def stage(name, **fields):
        report.update(stage=name, **fields)
        _write(args.output, report)

    try:
        stage("device_preflight")
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise RuntimeError("This probe requires native Apple Silicon macOS")
        import jax
        import jax.numpy as jnp
        import numpy as np
        from .linalg import register_mps_solver_lowerings
        from .physics import FlatPhysics

        versions = register_mps_solver_lowerings()
        versions.update(
            {
                name: importlib.metadata.version(name)
                for name in (
                    "mujoco",
                    "mujoco-mjx",
                    "mjlab",
                    "better-actuator-models",
                    "torch",
                )
            }
        )
        stage(
            "model_preparation",
            versions=versions,
            devices=[str(d) for d in jax.devices()],
        )
        start = time.perf_counter()
        physics = FlatPhysics()
        import mujoco

        model_path = args.output.with_suffix(".mjb")
        mujoco.mj_saveModel(physics.host_model, str(model_path))
        model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
        bam_hash = hashlib.sha256(
            Path(physics.actuator_cfg._resolved_json_path).read_bytes()
        ).hexdigest()
        seeds = jnp.arange(args.batch, dtype=jnp.uint32)
        state = jax.vmap(physics.initial_state)(seeds)
        params = (
            jnp.broadcast_to(physics.home, (args.batch, 14)),
            jnp.full((args.batch, 1), 7.35, dtype=jnp.float32),
            jnp.full((args.batch, 1), 0.1, dtype=jnp.float32),
            jnp.ones((args.batch, 1), dtype=jnp.float32),
        )
        function = jax.vmap(physics.step)
        step = jax.jit(function)
        if args.execution == "staged":
            from .staged import StagedJit

            step = StagedJit(function)
        stage(
            "lowering",
            compiled_model_sha256=model_hash,
            bam_model_sha256=bam_hash,
            preparation_seconds=time.perf_counter() - start,
            model_dimensions={
                "nq": physics.host_model.nq,
                "nv": physics.host_model.nv,
                "nu": physics.host_model.nu,
                "ngeom": physics.host_model.ngeom,
            },
        )
        hlo = str(jax.jit(function).lower(state, *params).compiler_ir())
        args.output.with_suffix(".mlir").write_text(hlo)
        audit = audit_hlo(hlo)
        stage("first_execution", hlo_audit=audit)
        start = time.perf_counter()
        state = step(state, *params)
        jax.block_until_ready(state)
        stage("steady_execution", first_execution_seconds=time.perf_counter() - start)
        times = []
        for _ in range(args.steps):
            start = time.perf_counter()
            state = step(state, *params)
            jax.block_until_ready(state)
            times.append(time.perf_counter() - start)
            if not np.isfinite(np.asarray(state.data.qpos)).all():
                raise RuntimeError("Nonfinite qpos after physics step")
            stage("steady_execution", steady_step_seconds=times)
        # Four physics steps per control step in the canonical task.
        stage(
            "complete",
            status="completed",
            finite_qpos=True,
            physics_transitions_per_second=args.batch * len(times) / sum(times),
            estimated_control_transitions_per_second=args.batch
            * len(times)
            / sum(times)
            / 4,
            training_qualified=False,
        )
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        _write(args.output, report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/native-gpu/physics.json")
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--execution", choices=("fused", "staged"), default="fused")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.batch < 1 or args.steps < 1 or args.timeout <= 0:
        parser.error("batch, steps and timeout must be positive")
    args.output = args.output.resolve()
    if args.worker:
        _worker(args)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[3]

    def git(*command):
        return subprocess.check_output(
            ["git", "-C", str(root), *command], text=True
        ).strip()

    report = {
        "status": "running",
        "stage": "starting",
        "scope": "physics qualification only",
        "batch": args.batch,
        "steps": args.steps,
        "execution": args.execution,
        "timeout_seconds": args.timeout,
        "platform": platform.platform(),
        "python": sys.version,
        "commit": git("rev-parse", "HEAD"),
        "git_status": git("status", "--short"),
        "lock_sha256": hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest(),
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py")
        },
        "training_qualified": False,
    }
    _write(args.output, report)
    env = dict(os.environ, JAX_PLATFORMS="mps", PYTORCH_ENABLE_MPS_FALLBACK="0")
    log = args.output.with_suffix(".log")
    with log.open("w") as stream:
        child = subprocess.Popen(
            [
                sys.executable,
                "-m",
                __package__ + ".probe",
                "--worker",
                "--output",
                str(args.output),
                "--batch",
                str(args.batch),
                "--steps",
                str(args.steps),
                "--execution",
                args.execution,
            ],
            env=env,
            stdout=stream,
            stderr=stream,
        )
        try:
            code = child.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
            report = json.loads(args.output.read_text())
            report.update(
                status="timed_out", error="Qualification exceeded the wall-clock limit"
            )
            _write(args.output, report)
            code = 124
        except KeyboardInterrupt:
            child.kill()
            child.wait()
            report = json.loads(args.output.read_text())
            report.update(status="interrupted")
            _write(args.output, report)
            raise
    report = json.loads(args.output.read_text())
    if code and report["status"] == "running":
        report.update(
            status="failed", error=f"Worker exited with status {code}; see {log}"
        )
        _write(args.output, report)
    print(json.dumps(report, indent=2))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
