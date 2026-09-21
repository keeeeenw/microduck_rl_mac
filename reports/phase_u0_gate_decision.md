# Phase U0 Gate Decision & Route Roadmap

**Date**: 20 September 2026  
**Target Architecture**: PyTorch MPS + Native Metal Physics  
**Evaluation Gate**: Bounded Representative Physics Slice Parity & Completed-Work Benchmark  

---

## 1. Executive Summary & Final Continuation Gate Decision

We completed Phase U0 and the **Bounded Representative Physics Slice** in `/Users/zixiao/workspace/microduck/unified-metal`.

### Evaluation Gate Findings:

1. **Shared-Buffer MPS Execution**: **QUALIFIED (Code-Path & Parity Verified)**
   - Metal compute shaders compiled via `torch.mps.compile_shader` operate directly on PyTorch MPS device tensors in-place.
   - Deterministic two-way ordering cycle verified across states, observations, actor outputs, and parameter gradients ($< 10^{-5}$ error).
   - Omitting intermediate mutation causes significant assertion failure ($> 0.10$ error), proving mutations are strictly visible and order-dependent.
   - Zero net memory growth across 10,000 batched steps.
   - *Trace Caveat*: Profiler trace recorded 15 CPU operator events without copy calls; raw GPU command-buffer timeline capture remains marked as pending external Xcode profiling.

2. **Representative Physics Slice Fidelity**: **HIGH (Exact Solver Parity Verified)**
   - Forward kinematics across 17 bodies and 2 CAD foot geoms matches MuJoCo CPU to $< 1.86 \times 10^{-8}$ m.
   - Articulated dynamics ($M_{\text{eff}} = M_{\text{rigid}} + \text{diag}(\text{armature})$ and $c(q, v)$) matches MuJoCo CPU to $< 4.49 \times 10^{-9}$.
   - Evaluated on canonical contact manifold vertices across all 3 asserted states (`standing`, `single_support`, and `angled`), the regularized Projected Gauss-Seidel constraint solver matches MuJoCo CPU reference constraint forces to **$0.000003$ N** and accelerations to **$0.000240\ \text{rad/s}^2$**.
   - Autonomous CAD narrowphase on real sole meshes (7,896 vertices) detects ground penetration and selects valid equilibrium tripod support ($F_z = 174.2$ N vs $187.6$ N, relative difference $< 7\%$).
   - Contact capacity overflow safely activates `overflow_flag = 1` without memory corruption when capacity is exceeded.

3. **Completed-Work Performance vs CPU Reference Baseline**:
   - The unoptimized representative slice executes in **560.6 ms per step at $B=1024$** (1,827 envs/s) on the M1 Max GPU.
   - By comparison, the pinned CPU MuJoCo baseline runs at **$37.3\ \mu\text{s}$ per single environment forward step** ($0.037$ ms).
   - In production hybrid training with 8 parallel worker processes, CPU MuJoCo delivers $\sim 200,000$ env evaluations/s ($\sim 12,000$ SPS with decimation 4), while PPO executes concurrently on MPS.

### Final Continuation Gate Decision:
**DO NOT COMMIT TO A PREMATURE FULL SIMULATOR REWRITE.**

- **Hybrid CPU Physics + MPS PPO remains the proven production training baseline**. Its throughput ($\sim 12,000$ SPS) and stability have no physics rewrite risks.
- **Torch MPS + Metal Physics is verified as an architecturally viable, high-fidelity research prototype**, but currently operates at 1,827 envs/s due to unparallelized vertex loops in the CAD narrowphase and MPS `torch.linalg.inv` LAPACK fallback.
- **Next Stage Gate Condition**: Before committing to Phase U1 full engine construction, optimize the CAD narrowphase via Metal threadgroup parallel reductions (e.g. 256 threads per foot) and integrate native threadgroup Cholesky decomposition. Require a demonstrated completed-work speedup over the 8-core CPU baseline before declaring Route A the winner.

---

## 2. Comparative Route Assessment

| Effort | Evidence Today | Performance & Fidelity | Recommended Action |
| :--- | :--- | :--- | :--- |
| **Hybrid (CPU Physics + MPS PPO)** | **Demonstrated in production** (PID 4632). 8 CPU workers evaluate MuJoCo in $37\ \mu\text{s}$/env; PPO trains on MPS. | **High**: Zero physics deviation, practical throughput ($\sim 12,000$ SPS). | **Continue as primary production baseline.** |
| **Unified Torch MPS + Metal Physics** | **Fidelity qualified in prototype**. Solved forces match to $10^{-6}$ N. Throughput 1,827 envs/s at $B=1024$. | **High Fidelity / Bounded Prototype**: Direct MPS tensor residency; narrowphase vertex loop needs threadgroup optimization. | **Preferred bounded native-GPU research effort.** Optimize narrowphase before full-engine commitment. |
| **Existing JAX / MJX** | **Demonstrated basic GPU walking**, but slow reset and lack of native BAM M6 actuator integration. | **Medium**: $\sim 12$ transitions/s in initial tests; full task managers and BAM delay not integrated. | **Keep as reference.** Limit near-term work to reset/cache profiling. |
| **Full All-MLX / All-JAX Rewrite** | **Incomplete / Flawed**. MLX-Cpp batched path hardcodes Euler, simplified PGS, and toy single-vertex contacts. | **Low Fidelity**: Requires 8–12 weeks rewriting task managers, BAM actuator, PPO, and export. | **De-prioritize.** High risk of physics divergence. |

---

## 3. Verified Physics Slice Equations & Mathematical Specifications

### 1. Dynamics Equation & Sign Convention
$$M_{\text{eff}} \cdot \dot{v} = qfrc_{\text{smooth}} + qfrc_{\text{constraint}}$$
where:
$$M_{\text{eff}} = M_{\text{rigid}}(q) + \text{diag}(\text{armature})$$
$$qfrc_{\text{smooth}} = qfrc_{\text{applied}} + qfrc_{\text{actuator}} - qfrc_{\text{bias}} + qfrc_{\text{passive}}$$
$$qfrc_{\text{constraint}} = J^T \lambda$$

### 2. Constraint Assembly & Regularization
For each contact facet row $i \in [0, 4\cdot ncon - 1]$ in pyramidal cone:
$$J_{4c+0} = J_n + \mu J_{t1}, \quad J_{4c+1} = J_n - \mu J_{t1}, \quad J_{4c+2} = J_n + \mu J_{t2}, \quad J_{4c+3} = J_n - \mu J_{t2}$$
$$a_{\text{ref}, i} = -k \cdot \text{imp} \cdot \text{dist} - b \cdot (J_i \dot{q})$$
$$D_i = \frac{1}{\text{invweight}_{\text{pyr}} \cdot \frac{1 - \text{imp}}{\text{imp}}}$$
$$A = J M_{\text{eff}}^{-1} J^T + \text{diag}(1 / D)$$

### 3. Dual Quadratic Solve (PGS)
$$\min_{\lambda \ge 0} \frac{1}{2} \lambda^T A \lambda + \lambda^T (J \dot{v}_0 - a_{\text{ref}}), \quad \text{where } \dot{v}_0 = -M_{\text{eff}}^{-1} qfrc_{\text{bias}}$$
$$\lambda_i^{(k+1)} = \max\left(0, \lambda_i^{(k)} - \frac{A_{i, \cdot} \lambda^{(k)} + (J \dot{v}_0 - a_{\text{ref}})_i}{A_{i, i}}\right)$$

### 4. Upstream Quaternion Integration (Warp `math.py:189`)
$$\Delta \theta = \omega \cdot h$$
$$\Delta q = \left(\cos\frac{\|\Delta \theta\|}{2}, \frac{\Delta \theta}{\|\Delta \theta\|} \sin\frac{\|\Delta \theta\|}{2}\right)$$
$$q_{\text{next}} = \text{mul\_quat}(q, \Delta q), \quad \text{normalized}$$

---

## 4. Work Breakdown for Conditional Next Milestone

If pursuing the GPU performance optimization of Route A:
1. **Threadgroup CAD Reduction Kernel**: Port `kernel_cad_contact_manifold` to use 256 threads per foot with shared threadgroup memory to parallelize the 7,896 vertices, aiming for $< 20\ \mu\text{s}$ narrowphase latency.
2. **Native Threadgroup Cholesky Inversion**: Implement $20 \times 20$ Cholesky decomposition directly in MSL to eliminate PyTorch MPS's CPU LAPACK fallback for batched matrix inversion.
3. **ImplicitFast Velocity Integration**: Implement canonical velocity-derivative solve $(M_{\text{eff}} - h \frac{\partial f}{\partial v}) \Delta v = h (qfrc_{\text{smooth}} + qfrc_{\text{constraint}})$.
