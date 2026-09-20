# Microduck RL — Native Apple GPU Training

**Native Apple GPU support for physics simulation and RL training** on Apple Silicon Macs.
For [Microduck](https://github.com/pollen-robotics/microduck), using the
[Microduck RL](https://github.com/pollen-robotics/microduck_rl) environments and
[mjlab](https://github.com/mujocolab/mjlab).

The goal is to train locally on a Mac and deploy the same exported policy on a
Linux machine. Native GPU acceleration is the priority; CPU execution is used for
reference checks and validation where needed. The experimental flat-terrain path
runs physics, BAM actuation, policy rollouts and PPO updates on the Apple GPU.
The initial 64-environment smoke test was validated on an **M1 Max with 32 GB
of unified memory**. Learning quality and throughput remain under evaluation.

Community support and pull requests are welcome. Reproducible bug reports,
Apple Silicon test results, performance improvements and Linux validation all
help move the project forward.

## Phases

1. **Mac locomotion training.** Get `Mjlab-Velocity-Flat-MicroDuck` and
   `Mjlab-Velocity-Rough-MicroDuck` simulation and training working on Apple GPUs,
   with checkpoint resume and policy export. The initial work targets flat terrain;
   rough terrain remains part of this phase. See the [current training commands](docs/mac-training.md).
2. **Training quality and Linux portability.** Improve throughput and evaluate
   learned behaviors, then validate deployment of the exact Mac-exported model
   on Linux without retraining or re-exporting.
3. **More tasks and behaviors.** Extend Mac support to additional Microduck
   environments, policy families and evaluation workflows.
4. **End-to-end Mac experience.** Bring simulation, visualization, runtime
   integration and supported peripherals together into a practical local workflow.

## Limitations

- Experimental and currently focused on Apple Silicon. Intel Macs are not supported.
- Flat-terrain training is the first supported development path. Rough terrain and
  other environments are not yet qualified on the native GPU backend.
- **These Mac changes and Mac-trained models have not been validated on Linux yet.**
- The GPU backend uses pinned dependencies and still needs performance and
  long-running learning validation. JAX and PyTorch currently exchange arrays
  through host memory; physics and learning numerics run on the GPU.
- A successful training smoke test does not establish walking quality, simulation
  equivalence across backends or readiness for physical robot deployment.
- Full Mac runtime, visualization and peripheral support is still in progress.

Licensed under [Apache License 2.0](LICENSE), the same license as the original
[Microduck RL project](https://github.com/pollen-robotics/microduck_rl/blob/develop/LICENSE).
Upstream license and attribution notices are retained.
