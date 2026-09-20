# Metal Dispatch Overhead, Memory Stability, and Trace Evidence Report

Generated on 2026-09-20.

> [!NOTE]
> **Workload Context**: 5-stage representative physics sequence (kinematics, broadphase, actuation, integration, sensors), 4 substeps (20 kernel launches per control step).
> **Machine Contention**: `concurrent-load` (Production hybrid training PID 4632 actively running on CPU/MPS).
> **Scope Clarification**: This benchmark measures kernel submission latency, dispatch overhead, and stream ordering on PyTorch MPS. It does not measure iterative constraint solver convergence or full CAD narrow phase arithmetic.

---

## 1. Dispatch Overhead & Throughput Table

| Batch Size (`num_envs`) | CPU Submission / Kernel ($\mu$s) | GPU Event / Step (ms) | Wall Time / Step (ms) | Control SPS (50 Hz equiv) | Physics SPS (200 Hz equiv) |
| --- | --- | --- | --- | --- | --- |
| 1 | 2.92 | 0.106 | 0.111 | 9,044.1 | 36,176.3 |
| 8 | 2.72 | 0.112 | 0.114 | 8,810.2 | 35,240.9 |
| 32 | 2.69 | 0.124 | 0.109 | 9,173.1 | 36,692.5 |
| 64 | 2.66 | 0.130 | 0.107 | 9,318.6 | 37,274.5 |
| 128 | 2.70 | 0.141 | 0.107 | 9,359.4 | 37,437.4 |
| 256 | 2.75 | 0.108 | 0.112 | 8,933.8 | 35,735.0 |
| 512 | 2.72 | 0.160 | 0.127 | 7,868.7 | 31,475.0 |
| 1,024 | 2.68 | 0.151 | 0.161 | 6,197.2 | 24,788.7 |
| 2,048 | 2.65 | 0.190 | 0.199 | 5,015.0 | 20,059.8 |
| 4,096 | 2.61 | 0.275 | 0.259 | 3,866.2 | 15,464.8 |

---

## 2. Memory Residency & Stability (10,000 Steps)

| Metric | PyTorch MPS Allocator | Apple Metal Driver Allocation |
| --- | --- | --- |
| **Baseline after Warmup** | 15.00 KB | 26.40 MB |
| **Peak during 10k Steps** | 15.00 KB | 26.40 MB |
| **Final Retained Memory** | 15.00 KB | 26.40 MB |
| **Net Growth ($\Delta$)** | **0 Bytes** | **0 Bytes** |

- **Plateau Stability**: The MPS allocator and Apple driver memory exhibit a completely flat plateau over 10,000 consecutive batched control steps.
- **Leak Verification**: No unreferenced command buffers or leaked intermediate allocations occur across repeated step invocations.

---

## 3. Two-Way Ordering & Trace Evidence (Absence of Host Staging)

A Chrome Trace capture was executed for the full bidirectional sequence:
$$\text{Torch Producer (MPS)} \longrightarrow \text{Metal Kernel 1 (MPS)} \longrightarrow \text{Torch Transform (MPS)} \longrightarrow \text{Metal Kernel 2 (MPS)} \longrightarrow \text{Torch MLP Consumer (MPS)}$$

- **Trace Artifact Location**:
  - Local: `unified-metal/reports/two_way_ordering_trace.json`
  - Persistent T7 Storage: `/Volumes/T7/ChatGPOExtension/unified-metal/traces/two_way_ordering_trace.json`
- **Trace Analysis Findings**:
  1. **Zero Host Copy Operators**: The trace records 23 in-stream MPS operator events (`aten::sin`, `aten::cos`, `aten::add`). Exactly **0** `aten::copy_`, `aten::_to_copy`, or CPU transfer operators occurred.
  2. **Zero Host Memory Allocations**: Host memory allocation events during the sequence = **0 Bytes**.
  3. **Direct Mutation Visibility**: Output tensors from Metal kernels were directly read by subsequent PyTorch native MPS operators without intermediate synchronization or host round-trips.

---

## 4. Layout & Contiguity Enforcement

1. **Contiguous Offset Views (`x[10:20]`)**: Verified that a 1D sliced view with non-zero storage offset (`storage_offset() == 10`) passes the correct device pointer offset into Metal kernels, mutating only the targeted slice while leaving prefix and suffix memory intact.
2. **Rejection of Non-Contiguous Layouts (`x[:, :14]`)**: Slices across non-contiguous dimensions and transposed tensors are detected and rejected with explicit `ValueError` before kernel submission, protecting against silent memory corruption.

---

## 5. Architectural Implications for Phase U0 Gate

- **Dispatch Overhead is Negligible**: CPU submission cost is $\sim 2.6$ to $2.9\ \mu\text{s}$ per kernel. For a 20-kernel control step (4 substeps $\times$ 5 physics stages), total submission overhead is $\sim 53\ \mu\text{s}$ ($0.053\text{ ms}$). This leaves $>99\%$ of the 20 ms control interval for GPU physics computation.
- **Hardware Residency Confirmed**: Tensors remain 100% resident on Apple Silicon unified GPU memory across both PyTorch and custom Metal shaders.
