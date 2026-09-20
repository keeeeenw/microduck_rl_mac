# Experimental Mac training

Requires a native Apple Silicon Python 3.12 environment and access to Metal.
The current entry point runs `Mjlab-Velocity-Flat-MicroDuck`; rough terrain is
not yet supported by this adapter. Run from the repository root with a recent uv.

```bash
uv sync --locked --extra mac-gpu --python 3.12
uv run --locked --extra mac-gpu python -m mjlab_microduck.native_gpu.train \
  --num-envs 64 --iterations 5 --save-interval 1 \
  --log-dir logs/native-gpu/smoke
```

The command selects JAX MPS and Torch MPS and disables CPU numerical fallback.
JAX/Torch transfers currently pass through host memory. Compilation and detailed
mesh collisions can make the first run slow.

Use a **new log directory** for each invocation. To continue a trusted checkpoint
with the same environment count and compatible pinned dependencies/source:

```bash
uv run --locked --extra mac-gpu python -m mjlab_microduck.native_gpu.train \
  --num-envs 64 --iterations 1000 --save-interval 5 \
  --resume logs/native-gpu/smoke/model_4.pt \
  --log-dir logs/native-gpu/continued
```

`--iterations` counts additional PPO updates. The run saves checkpoints after its
first update, at the requested interval, and on normal completion. Checkpoints
include the learner, native environment and RNG state; they are trusted Python
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
live GPU training environments or interrupt the trainer. Early checkpoints may
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

Type commands in the **terminal that launched playback**, not the viewer window.
The terminal must be interactive (TTY); redirecting stdin disables keyboard input.

| Key | Effect in default velocity mode |
| --- | --- |
| Up / Down | Increase / decrease forward velocity command |
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
