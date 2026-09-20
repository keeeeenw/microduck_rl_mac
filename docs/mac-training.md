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

The longer campaign uses the recommended hybrid mode on the same M1 Max.
Full-campaign completion and policy quality still require validation. The earlier
GPU-physics campaign is paused with its checkpoints retained.

`--iterations` counts additional PPO updates. The run saves checkpoints after its
first update, at the requested interval, and on normal completion. Checkpoints
include the learner, native environment and RNG state. Full environment continuation
requires the same physics backend. Loading a checkpoint from another backend
retains the learner and progress but starts fresh episodes, reported in
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
