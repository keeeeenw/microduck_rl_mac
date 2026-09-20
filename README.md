# Unified Metal Physics & Training (Phase U0)

Shared-buffer Metal physics and training exploration for MicroDuck locomotion on Apple Silicon.

## Overview

This isolated workspace implements Phase U0 of the Unified Metal Training Plan (`private-notes/native-mac/unified-metal-training-plan.md`). It explores a shared-buffer architecture where **PyTorch MPS owns persistent environment state** and **custom Metal physics operators consume and update that state directly**, eliminating host staging between simulation, actuator dynamics, and policy learning.

## Key Reports

- [Task Contract & Physical Inventory](reports/task_contract_inventory.md) (`Deliverable 1`): Live extraction of canonical coordinates, options, contact pairs, BAM parameters, DR curricula, and state views.
- [Metal Dispatch Overhead & Memory Stability Report](reports/metal_probe_results.md) (`Deliverables 2 & 3`): Synchronized benchmark measuring CPU submission latency, GPU completion events, and Chrome trace evidence of zero host staging.
- [Upstream Reusable Kernel & Algorithm Audit](reports/reusable_kernel_audit.md) (`Deliverable 4`): Systematic evaluation of MuJoCo Warp and MuJoCo-MLX-Cpp against canonical physics stages.
- [Phase U0 Gate Decision & Route Roadmap](reports/phase_u0_gate_decision.md) (`Deliverable 5`): Architectural comparison and recommendation to proceed with Torch MPS + Metal Physics.

## Directory Structure

```
unified-metal/
├── .gitignore
├── README.md
├── provenance.json                        # Git hashes, dependency versions, hardware config
├── configs/
│   ├── canonical_flat_task.json           # Extracted JSON configuration
│   └── microduck_live_flat.xml            # Live compiled XML model
├── shaders/
│   ├── probe_ops.metal                    # Bidirectional mutation, slicing, and strided ops
│   └── benchmark_ops.metal                # 5-stage representative physics sequence
├── src/
│   ├── __init__.py
│   ├── metal_kernel_manager.py            # MSL compiler & layout assertion helpers
│   ├── task_inventory.py                  # Live task extraction script
│   ├── metal_probe.py                     # Bidirectional shared-buffer probe & memory test
│   ├── benchmark_dispatch.py              # Synchronized multi-kernel dispatch benchmark
│   └── audit_reusable_kernels.py          # Upstream kernel & license audit script
├── tests/
│   ├── test_task_inventory.py             # Pytest for task inventory verification
│   └── test_shared_buffer.py              # Pytest for two-way ordering, slicing, memory
└── reports/
    ├── task_contract_inventory.md
    ├── metal_probe_results.md
    ├── reusable_kernel_audit.md
    ├── phase_u0_gate_decision.md
    └── two_way_ordering_trace.json        # Chrome trace proving zero host copies
```

## Running Tests & Benchmarks

Using the pinned Python environment:

```bash
# 1. Run full test suite
/Users/zixiao/workspace/microduck/microduck_rl/.venv/bin/python -m pytest tests/ -v

# 2. Run two-way buffer probe and 10k step memory test
/Users/zixiao/workspace/microduck/microduck_rl/.venv/bin/python src/metal_probe.py

# 3. Run dispatch overhead benchmark across batch sizes 1 to 4096
/Users/zixiao/workspace/microduck/microduck_rl/.venv/bin/python src/benchmark_dispatch.py
```
