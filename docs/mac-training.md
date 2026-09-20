# Experimental Mac training

Requires a native Apple Silicon Python 3.12 environment and access to Metal.
The current entry point runs `Mjlab-Velocity-Flat-MicroDuck`; rough terrain is
not yet supported by this adapter. Run from the repository root with a recent uv.

**Validation hardware: Apple M1 Max with 32 GB of unified memory (24-core GPU).**
The recommended mode is native CPU MuJoCo physics with Torch MPS policy inference,
BAM actuator calculations, task managers and PPO learning. It preserves the original
flat task's rewards, commands, randomization and learner configuration.

```bash
uv sync --locked --extra mac-gpu --python 3.12
uv run --locked --extra mac-gpu python -m mjlab_microduck.native_gpu.train \
  --physics cpu --num-envs 64 --iterations 5 --save-interval 1 \
  --log-dir logs/native-gpu/smoke
```

`--physics cpu` is the default. The five-update smoke test and a five-update full
checkpoint continuation passed with 64 environments, 24 rollout steps per update,
finite losses/gradients, no logged NaN terminations, and normalized ONNX export
comparison. This does not establish useful walking or performance on other Macs.

The `mac-gpu` dependency extra supports both physics modes. Torch MPS fallback is
disabled in both: choosing CPU physics is explicit, not a silent GPU fallback.

### Physics backend measurements

Initial measurements on the M1 Max, using the same canonical flat-task settings,
64 environments and 1,536 transitions per PPO update:

| Mode | Mean seconds/update | Transitions/second |
| --- | ---: | ---: |
| MJX/Metal physics + MPS learner, earlier five-update smoke | 174.46 | 8.80 |
| CPU MuJoCo physics + MPS learner, five-update smoke | 5.45 | 282.00 |

These are full rollout-plus-PPO timings, excluding initialization, checkpoint/export
I/O and validation. They show about a 32-fold improvement for the hybrid mode in
these runs. They are not a trajectory-matched physics comparison or a learning
quality benchmark: native MuJoCo and MJX use different collision implementations
and numerical precision, and the runs did not start from identical physics state.
Repeat broader benchmarks before extrapolating to other tasks or hardware.

### Environment-count tuning on the M1 Max

Before the transfer optimizations below, a sequential CPU-physics/MPS-PPO sweep
on the same 32 GB M1 Max found:

| Environments | Seconds/update | Transitions/second | Peak process RSS (GiB) |
| ---: | ---: | ---: | ---: |
| 512 | 14.09 | 872 | 2.95 |
| 1,024 | 21.58 | 1,139 | 4.35 |
| 2,048 | 40.86 | 1,203 | 7.32 |
| 4,096 | 109.60 | 897 | 12.36 |

Timings use updates 2–3 of fresh seed-42 runs, excluding startup and checkpoint
I/O. These are short throughput measurements, not convergence comparisons. RSS
is process resident memory, not total system or GPU memory usage. All four runs
completed and passed normalized ONNX export comparison. An 8,192-environment
trial initialized but took 69 and 76 seconds for its first two control steps,
with increased system swap usage. It was stopped before PPO; it did not establish
OOM-free training at that size.

**2,048 environments was the fastest setting before these optimizations.** Keep 24
rollout steps, five PPO epochs and four minibatches (12,288 samples/minibatch).
PPO took only about 0.4 seconds/update; most time was rollout. CPU physics still
steps worlds sequentially, so increasing environments does not use more CPU cores.
CPU/MPS data transfers also contribute overhead. Parallel physics and reduced
transfers were investigated next; parallel physics remains future work.

The optimized bridge omits visualization-only environment-origin sites, eliminating
quadratic site-pose storage and copies across worlds. It also sizes constraint
buffers to active demand and copies directly into persistent MPS tensors. It still
uses separate NumPy and MPS buffers; unified memory does not make it zero-copy.
MuJoCo dynamics, robot sites and sensor values matched in regression tests. The
original and optimized five-update 4,096-environment exports produced identical
actions on 20 fixed synthetic observations, in addition to the normal export check.

Post-optimization five-update trials (means over updates 2–5):

| Environments | Seconds/update | Transitions/second | Peak process RSS (GiB) | Final checkpoint (MiB) |
| ---: | ---: | ---: | ---: | ---: |
| 2,048 | 28.76 | 1,709 | 5.68 | 41 |
| 4,096 | 48.24 | 2,038 | 8.89 | 77 |

The optimized 8,192 trial completed two PPO updates at 2,037 and 1,992
transitions/second, with 12.15 GiB peak RSS. It was stopped because it offered no
throughput benefit; it did not complete the five-update export-validation test.
**Use 4,096 environments on this tested M1 Max configuration:** it has the best
validated throughput here and preserves the original PPO batch size. These short
measurements do not establish an optimum for other hardware or learning quality.

```bash
uv run --locked --extra mac-gpu python -m mjlab_microduck.native_gpu.train \
  --physics cpu --num-envs 4096 --iterations 5 --save-interval 250 \
  --log-dir logs/native-gpu/smoke-4096
```

Use sparse checkpoints: every 250 updates plus initial/final saves. Full native
environment checkpoints grow with environment count (about 1 GiB at 4,096 before
optimization, 77 MiB in the optimized test). They remain larger than portable ONNX
policies. Use an external drive through `--log-dir` when desired, and keep it
mounted throughout training and checkpoint/export access.

At 2,048 environments, 8,000–12,000 updates collect as many transitions as the
upstream 4,096-environment, 4,000–6,000-update gait budget. This is only sample-count
equivalence: smaller PPO batches and the inherited curriculum change learning.
The curriculum currently advances by environment steps, not aggregate transitions;
review its schedule before a long campaign. The sweep does not validate walking.

To investigate the experimental Apple GPU physics path, select `--physics mps`
and use a separate log directory. It works for flat-task smoke tests but remains
slower; it is retained for phase 3 comparisons, not recommended for the first
learning campaign. Host memory transfers are used between JAX and Torch MPS.

Use a **new log directory** for each invocation. To continue a trusted checkpoint
with the same environment count, physics backend and compatible pinned dependencies/source:

```bash
uv run --locked --extra mac-gpu python -m mjlab_microduck.native_gpu.train \
  --physics cpu --num-envs 64 --iterations 1000 --save-interval 5 \
  --resume logs/native-gpu/smoke/model_4.pt \
  --log-dir logs/native-gpu/continued
```

The initial 64-environment hybrid campaign and earlier GPU-physics campaign have
been stopped with their checkpoints retained. The selected campaign continues the
validated 4,096-environment checkpoint toward 6,000 total updates, with saves every
250 updates on an external drive. Full-campaign completion and walking quality
still require validation. For example, after the five-update smoke test:

```bash
uv run --locked --extra mac-gpu python -m mjlab_microduck.native_gpu.train \
  --physics cpu --num-envs 4096 --iterations 5995 --save-interval 250 \
  --resume logs/native-gpu/smoke-4096/model_4.pt \
  --log-dir /Volumes/T7/microduck-rl/flat-4096
```

Replace the example external-drive directory with your mounted output location.

`--iterations` counts additional PPO updates. The run saves checkpoints after its
first update, at the requested interval, and on normal completion. Checkpoints
include the learner, native environment and RNG state. Full environment continuation
requires the same physics backend and native scene layout. Loading a checkpoint
from another backend or from before the origin-marker removal retains the learner
and progress but starts fresh episodes, reported in
`status.json` as `learner_and_progress_with_fresh_episodes`. Checkpoints are trusted Python
serialization files, not portable deployment artifacts.

Run outputs:

- `status.json`: current progress, latest checkpoint and completion/error status.
- `progress.jsonl` and TensorBoard events: learning metrics and throughput.
- `manifest.json`: source, dependency and input checkpoint provenance.
- `model_*.pt`: resumable native training checkpoints.
- `policy.onnx`: normalized policy exported on normal completion, checked against
  the trained actor using ONNX Runtime. Its SHA-256 appears in the final status.

The ONNX file is the intended cross-platform deployment artifact. Linux inference,
robot runtime integration and useful walking behavior still require validation.

## Interactive policy playback on Mac

Playback runs one robot in the native MuJoCo 3D viewer using CPU physics and ONNX
Runtime inference. This is a separate validation process: it does not show the
live training environments or interrupt the trainer. Early checkpoints may
stand still, stumble or fall; a successful replay is not evidence of learned walking.

From the repository root, after installing the environment above:

```bash
mkdir -p artifacts/playback
cp logs/native-gpu/smoke/policy.onnx artifacts/playback/policy.onnx
.venv/bin/python scripts/play_mac.py \
  --walking artifacts/playback/policy.onnx \
  --new-cmd-obs --delay 3 6 --lin-vel-x 0.05
```

For a run that is still training, the existing runner also refreshes a rolling
ONNX export at checkpoint saves: `logs/native-gpu/<run>/<run>.onnx`. Copy that file
to the playback snapshot instead of the completed run's `policy.onnx`. For example:

```bash
cp logs/native-gpu/continued/continued.onnx artifacts/playback/policy.onnx
```

Playback loads the snapshot once. To inspect a newer policy, close playback, copy
the newer export, and relaunch. The trainer's final `policy.onnx` additionally has
the explicit actor-versus-ONNX comparison recorded in `status.json`.

The launcher uses MuJoCo's `mjpython`, adds the base Python library directory for
uv-managed environments, and clears Linux-specific `MUJOCO_GL` selection for this
process. This avoids the macOS `Library not loaded: ... libpython3.12.dylib` error
that can occur when launching `mjpython` directly from a uv virtual environment.
It does not change your shell configuration or the running trainer.

Keep `--new-cmd-obs`: it selects the trained 61-input observation contract with
13 command values. BAM M6 actuation is enabled by default; do not use `--no-bam`
for this comparison. `--delay 3 6` enables actuator delay in physics steps, and
`--lin-vel-x 0.05` starts with a small forward command. Playback uses nominal
battery settings and is not a replica of the randomized training distribution.

The delay buffer advances at 200 Hz: 3–6 physics steps means 15–30 ms,
while policy inference runs at 50 Hz. Earlier playback incorrectly advanced
that buffer at policy rate (60–120 ms), which could make a standing policy fall.
Restart playback after updating to pick up the corrected timing and latest export.

A 1,000-update run with 64 environments is an initial learning baseline, not a
walking qualification. It collects 1,536,000 transitions; assess sustained upright
motion and commanded velocity tracking, not the percentage of updates completed.

Type commands in the **terminal that launched playback**, not the viewer window.
The terminal must be interactive (TTY); redirecting stdin disables keyboard input.

| Key | Effect in default velocity mode |
| --- | --- |
| Up / Down | Set forward velocity to +0.30 / −0.30 m/s (plain walking mode) |
| Left / Right | Change lateral velocity command |
| A / E | Change turning command |
| Space | Zero movement commands; does not reset the robot pose |
| T | Toggle policy inference; physics continues and motors hold the last target |
| P | Apply a random push |
| B / H | Toggle body-pose / head-command controls; terminal prints their bindings |
| Q | Quit playback (closing the viewer also exits) |

Restart playback to restore its initial robot state. Ground-pick, sitting, kicking
and rolling shortcuts require separately supplied policies; they are not skills
provided by the flat walking checkpoint. `--record` records observation data, not
video. The terminal prints achieved versus commanded velocity and trunk height
so behavior can be checked alongside the visualization.
