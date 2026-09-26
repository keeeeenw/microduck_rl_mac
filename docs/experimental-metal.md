# Experimental Unified Metal physics

Unified Metal is available for community testing and improvement on Apple Silicon.
It runs rigid-body dynamics, ground contact generation, contact constraint solving,
policy inference and PPO on the Apple GPU. Self-contact narrowphase and reset-time
constant recomputation still use CPU MuJoCo and explicit data staging.

**This is an experimental feature that still requires validation.** CPU MuJoCo
physics with MPS learning remains the default and recommended walking baseline.
Rough terrain, other tasks, other Mac configurations, Linux execution and physical
robot deployment are not validated for this backend. Feedback and pull requests
are welcome; please report reproducible results and distinguish successful training
from demonstrated walking and command tracking.

## Try it

Use an Apple Silicon Mac and native Python 3.12. The qualified implementation was
tested on an **M1 Max with 32 GB unified memory and a 24-core GPU**, using Torch
2.9.1 and MuJoCo 3.10.0. The native entry point currently runs only
`Mjlab-Velocity-Flat-MicroDuck`.

From the repository root:

```bash
uv sync --locked --extra mac-gpu --python 3.12
uv run --locked --extra mac-gpu python -m mjlab_microduck.native_gpu.train \
  --verify-backend-only
uv run --locked --extra mac-gpu python -m mjlab_microduck.native_gpu.train \
  --physics metal --num-envs 64 --iterations 5 --seed 42 --save-interval 5 \
  --log-dir logs/native-gpu/metal-smoke
```

The first command installs dependencies into this checkout's environment. Do not
sync an environment belonging to an active training job. Preflight checks module,
shader and model paths without compiling a shader or running GPU physics. The
shader and canonical model are bundled in the package; a sibling repository and
`PYTHONPATH` customization are not required. If you previously configured
`MICRODUCK_METAL_BACKEND_ROOT`, unset it: external experiment overrides are rejected.

Use a fresh log directory for every invocation. Inspect `status.json`,
`progress.jsonl`, `console` output and `manifest.json`. A successful smoke run
finishes with `status: completed` and `export_parity: true`, after finite rollout
and optimizer checks and a comparison between the normalized actor and ONNX export.
This establishes a working short run, not a useful walking policy.

After the smoke succeeds, an example longer experiment is:

```bash
uv run --locked --extra mac-gpu python -m mjlab_microduck.native_gpu.train \
  --physics metal --num-envs 2048 --iterations 12000 --seed 42 --save-interval 500 \
  --log-dir logs/native-gpu/metal-walk
```

2,048 environments is the tested pilot configuration, not a universal optimum.
Monitor available unified memory and disk space. Set the log directory to external
storage if needed; checkpoint size varies with state and environment count.
Do not run GPU benchmarks or another trainer concurrently when measuring performance.

## Resume and playback

Read the checkpoint path from the previous run's `status.json`; replace the example
path below with that actual path. Preserve the environment count and use a new log
directory. `--iterations` specifies **additional updates**, not a final target.

```bash
uv run --locked --extra mac-gpu python -m mjlab_microduck.native_gpu.train \
  --physics metal --num-envs 2048 --iterations 1000 --seed 42 --save-interval 500 \
  --resume /absolute/path/to/model_CHECKPOINT.pt \
  --log-dir logs/native-gpu/metal-resumed
```

For compatible native Metal checkpoints, check that `resume_kind` is
`full_environment_and_learner`. A weights/progress-only resume with fresh episodes
is different and must not be presented as full simulation continuation. Do not
assume a full environment checkpoint transfers between different physics backends.

The logger's reward and episode-length accumulators restart on resume, so early
summaries may reflect only the post-resume portions of episodes. A rising log
curve immediately after resume does not establish newly learned walking.

Completed runs export `policy.onnx` with observation normalization included. Use
the existing [policy playback instructions](mac-training.md) to evaluate the actual
export in CPU MuJoCo. Check commanded forward, lateral, turning and stop behavior;
reward and survival alone are insufficient. Export support does not yet establish
Linux or real-robot deployment parity.

## Evidence and limitations

The qualified implementation before packaging passed 254 physics/task/continuation
cases plus 11 CPU cache/payload cases. Both fresh and resumed 64-environment,
five-update trainer checks passed, including normalized ONNX parity. Three matched
2,048-environment experiments produced equal logged learning/contact metrics over
15 updates per variant. This is bounded evidence, not exhaustive state equality or
long-run policy quality.

Packaging changes relocate imports and resources without changing the qualified
shader or physics calculations. CPU regressions and wheel/resource loading checks
cover this integration. **A fresh GPU smoke and continuation check on the packaged
entry point remain pending**; they were deferred to avoid interfering with the
active pilot. Please start with the smoke test on your machine and report results.

### Observed training speed

The longer training logs show a larger gain than the small, recent Metal-to-Metal
optimization. On the same M1 Max (32 GB unified memory, 24-core GPU), the observed
median collection-plus-PPO throughput was approximately **1.31× the CPU-physics
run for the original Unified Metal implementation**, and **1.46× for the updated
Metal candidate**:

| Training run | Environments | Updates measured | Median collection + PPO / update | Median transitions/s |
| --- | ---: | --- | ---: | ---: |
| Completed CPU physics + MPS PPO | 4,096 | 16–6,000 (5,985 updates) | 51.42 s | 1,912 |
| Original Unified Metal + MPS PPO, subsequently stopped | 2,048 | 11–3,931 (3,921 updates) | 19.61 s | 2,507 |
| Updated Unified Metal + MPS PPO, ongoing pilot | 2,048 | 3,512–3,622 (111 updates) | 17.55 s | 2,801 |

Snapshot: September 26, 2026, with fixed cutoffs shown above; the first ten records
of each run are excluded. These are measurements from the actual training jobs,
not extrapolations from a short smoke test. The current candidate has a shorter
observation window than the original Metal run. Median consecutive logged wall
intervals, including other update overhead, were 52.58 s, 20.57 s and 18.81 s,
respectively.

**Compare transitions/second, not updates/second:** Metal used half as many
environments, so an update contained 49,152 samples versus 98,304 for the CPU run.
The candidate's updates completed roughly three times as quickly, but the observed
sample-throughput gain was about **46%**, not threefold. These historical runs
have different batch sizes, training stages and runtime conditions; the ratios
describe observed training throughput, not a controlled backend-only speedup or
faster convergence to a walking policy. Broader hardware and matched CPU/Metal
comparisons remain open.

Separately, the recent optimization over the earlier Metal implementation measured
approximately **1.076×** using the ratio of median trial wall times (individual
trial ratios: 0.990×, 1.086× and 1.161×). That small comparison isolates an
incremental Metal optimization; it does not represent the whole transition from
CPU physics to Unified Metal.

The backend has separate collision and reset transfer counters. A zero general
`transfer_bytes_total` counter does not mean zero staging. Timers can overlap and
must not be summed as independent costs. Each 2,048-environment update contains
49,152 transitions (24 control steps per environment). Logged transitions/second
uses collection plus PPO time; full-update wall time also includes other overhead.
Peak process RSS is not a complete GPU or unified-memory measurement.

## Contribute and validate

Report the commit, Mac chip/GPU cores, memory, macOS, dependency versions, exact
command, environment count, seed and relevant logs. Include all matched timing
trials and their windows. Remove personal paths or credentials before sharing.
Useful contributions include installation fixes, trajectory/observation parity,
contact and reset regressions, checkpoint compatibility, command-following
assessments, broader hardware/task validation and measured performance improvements.
Keep correctness and timing evidence separate. CUDA and CPU behavior must remain
unchanged when Metal is not selected.

CPU checks (no Metal physics allocation):

```bash
uv run --locked --extra mac-gpu --with pytest python -m pytest -q \
  tests/test_metal_packaging.py tests/test_metal_constant_cache.py \
  tests/test_metal_fk_cache.py tests/test_metal_friction_payload.py \
  tests/test_analyze_native_gpu_profile.py tests/test_aarch64_cuda_torch.py
```

The GPU qualification suite is excluded from normal test collection. Run it only
on an idle Apple Silicon machine; it compiles shaders and performs GPU physics
and short continuation/learner checks:

```bash
MICRODUCK_RUN_METAL_TESTS=1 uv run --locked --extra mac-gpu --with pytest \
  python -m pytest -q tests/metal
```

The 25 small reference fixtures are included in `tests/metal/corpus`; tests resolve
fixtures independently of the working directory. The runtime package is under
`mjlab_microduck.native_gpu.metal`. It uses the original Microduck assets, task
recipe and RSL-RL PPO; those upstream methods are not this project's contribution.
The project retains Apache License 2.0 and upstream attribution.
