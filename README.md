# Microduck RL on Mac

**Microduck reinforcement learning on Apple Silicon, with native Apple GPU learning.**
Built on the original [Microduck RL](https://github.com/pollen-robotics/microduck_rl)
environments and [mjlab](https://github.com/mujocolab/mjlab) for the
[Microduck robot](https://github.com/pollen-robotics/microduck).

The recommended development path combines **CPU MuJoCo physics with Apple GPU
policy inference and PPO training**. The original task managers, rewards,
randomization and BAM actuator calculations are retained. The goal is to train
locally on a Mac and deploy the same exported policy on Linux.

Native Apple GPU physics also runs, but our current experimental implementation
is substantially slower than CPU physics. Improving it is a later phase, using
the working task configurations as baselines. Initial validation used an
**M1 Max with 32 GB of unified memory**.

Community support and pull requests are welcome. Reproducible bug reports,
Apple Silicon test results, task validation and performance improvements help
move the project forward. See the [training and playback guide](docs/mac-training.md).

## Mac-specific improvements

- **Native Apple GPU learning:** Torch MPS runs policy inference and PPO alongside
  native CPU MuJoCo physics, preserving the original task and actuator models.
- **Less copying and memory overhead:** the native training scene omits unused
  environment visualization markers, and constraint buffers grow with actual
  demand. This removes unnecessary work as environment counts increase.
- **Playback aligned with training:** actuator delays advance at physics rate,
  fixing excessive delay that could make a standing policy fall in playback.
- **Portable policies and resumable training:** normalized ONNX exports target
  the existing deployment interface; full checkpoints preserve native training
  state, with configurable save intervals and external-drive output paths.

See the training guide for measured results and validation limits. Apple Silicon's
unified memory does not make this Python/Torch bridge zero-copy.

## Phases

1. **Establish the best practical Mac setup for walking.** Get
   `Mjlab-Velocity-Flat-MicroDuck` and `Mjlab-Velocity-Rough-MicroDuck` working
   with a useful combination of simulation throughput and GPU learning. Validate
   training, checkpoint resume, playback and export. CPU physics with MPS learning
   is the current choice; flat-task smoke tests pass, while rough terrain and
   reliable learned walking remain to be validated.
2. **Validate the remaining tasks.** Extend and qualify the other Microduck
   environments and policy families. Establish repeatable behavior, evaluation
   and performance baselines before claiming support for each task.
3. **Improve native Apple GPU physics.** Use the validated walking and other tasks
   from phases 1–2 as baselines. Improve physics throughput while checking learning
   quality and simulation fidelity, and adopt GPU physics where it demonstrates
   a practical advantage.

## Limitations

- Experimental and currently focused on Apple Silicon. Intel Macs are not supported.
- The current Mac adapter is qualified only for flat-task smoke tests. Rough terrain
  and other tasks still need backend support and validation.
- **These Mac changes and Mac-trained models have not been validated on Linux yet.**
- CPU physics with GPU learning is currently faster than the experimental native
  GPU physics path. Data transfers between CPU and GPU remain part of the workflow.
- A successful smoke test does not establish walking quality, simulation equivalence
  across backends or readiness for physical robot deployment.
- Full Mac runtime, visualization and peripheral support is still in progress.

Licensed under [Apache License 2.0](LICENSE), the same license as the original
[Microduck RL project](https://github.com/pollen-robotics/microduck_rl/blob/develop/LICENSE).
Upstream license and attribution notices are retained.
