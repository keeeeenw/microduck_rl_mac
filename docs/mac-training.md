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
