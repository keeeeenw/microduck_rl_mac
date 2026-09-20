# Phase U0 Gate Decision and Route Roadmap

20 September 2026. Private decision document for the MicroDuck GPU physics optimization workstream.

---

## 1. Executive Summary & Gate Verdict

### Phase U0 Gate Criteria Evaluation:
1. **Shared-Buffer MPS Execution**: **PASSED**
   - Metal shaders compiled via `torch.mps.compile_shader` directly read and mutate PyTorch MPS tensors in-place.
   - Bidirectional two-way ordering cycle verified: $\text{Torch} \rightarrow \text{Metal} \rightarrow \text{Torch} \rightarrow \text{Metal} \rightarrow \text{Torch MLP}$.
   - Memory stability verified: 0 bytes net growth in both PyTorch allocator and Apple driver memory across 10,000 steps.
   - Zero host staging confirmed via Chrome trace (`reports/two_way_ordering_trace.json`): zero copy operators and zero host staging events.
2. **Dispatch Overhead Budget**: **PASSED**
   - Measured CPU submission latency: $\mathbf{2.6\text{ to }2.9\ \mu s}$ per kernel launch.
   - For a 20-kernel control step (4 substeps $\times$ 5 representative physics stages), total submission overhead is $\sim 53\ \mu\text{s}$ ($0.053\text{ ms}$), well under the $1.0\text{ ms}$ target budget.
3. **Upstream Kernel & Algorithm Reuse**: **PASSED**
   - MuJoCo Warp (`mujoco_warp._src`) provides high-fidelity, Apache-2.0 reference implementations for `ImplicitFast` with velocity derivatives (`forward.py:577`), `Newton` solver with line search (`solver.py`), pyramidal friction cones (`constraint.py`), and robust GJK convex hull distance (`collision_gjk.py`).
4. **Architectural Simplicity**: **HIGH**
   - Because `mjlab`'s decimation loop (`manager_based_rl_env.py:414–421`) executes `action_manager.apply_action()` before each `sim.step()`, PyTorch's native `FrictionDRBamActuator` continues to run on MPS without needing a complex BAM Metal port.
   - PPO learner, rollouts, advantage estimation, and ONNX policy export remain 100% untouched in PyTorch.

### Final Phase U0 Decision:
**RECOMMEND ROUTE A: Proceed with Torch MPS + Native Metal Physics (Conditional Phase U1).**

---

## 2. Comparative Assessment of Architecture Routes

| Evaluation Dimension | Route A: Torch MPS + Metal Physics | Route B: All JAX / MJX | Route C: All MLX |
| --- | --- | --- | --- |
| **Physics Fidelity & Integrator** | **High Potential**: Port MuJoCo Warp's canonical `ImplicitFast` + Newton line-search solver directly to MSL | **Demonstrated**: Functioning MJX GPU physics | **Low/Flawed**: MLX-Cpp batched path hardcodes Euler, simplified PGS, single-vertex contacts |
| **Data Residency** | **Zero Staging Potential**: PyTorch MPS owns all tensors; Metal shaders mutate device buffers directly | **Zero Staging**: All JAX arrays on MPS | **Zero Staging**: All MLX arrays on MPS |
| **BAM Actuation & DR** | **Zero Porting**: Retain PyTorch `FrictionDRBamActuator` and DR managers unchanged | **Substantial**: Must port BAM M6 equations, voltage control, and delay ring buffers to JAX | **Substantial**: Must port BAM M6 and DR managers to MLX |
| **Task Managers & Sensors** | **Zero Porting**: Retain all `mjlab` managers, terms, and curricula in PyTorch | **Substantial**: Must rewrite all rewards, terminations, and observations in JAX | **Substantial**: Must rewrite all rewards, terminations, and observations in MLX |
| **PPO Learner & Rollouts** | **Zero Porting**: Existing PyTorch PPO learner, symmetry, and ONNX export preserved | **Substantial**: Must implement or port full PPO learner, GAE, and rollout storage to JAX | **Substantial**: Must implement or port full PPO learner, GAE, and rollout storage to MLX |
| **Dispatch Latency (Synthetic)** | $\sim 2.6\text{ to }2.9\ \mu\text{s}$ per kernel launch | PJRT dispatch latency | MLX stream dispatch latency |
| **Engineering Scope & Uncertainty** | **Bounded Physics Engine**: Build kinematics, dynamics, contact narrowphase & Newton solver. Uncertainty: solver iteration speed and contact robustness. | **Complete Rollout Rewrite**: Retain physics; rewrite BAM, task managers, rollouts, learner, and ONNX export. | **Full Stack Rewrite**: Fix flawed physics engine + port BAM + rewrite task managers + port learner. |

---

## 3. Explicit Missing-Feature List for Route A (Torch MPS + Metal)

To build the qualified physics engine in Phase U1, the following concrete stages must be implemented in Metal Shading Language (MSL):

### 1. Kinematics & Spatial Transforms (`kinematics.metal`)
- **Required**: Batched forward kinematics evaluating floating base ($p_{xyz}, q_{wxyz}$) and 14 joint angles.
- **Outputs**: 17 body frames ($xpos, xquat$) and 8 site positions ($site\_xpos$).
- **Upstream Reference**: MuJoCo Warp `smooth.py:kinematics`.

### 2. Articulated Dynamics: Inertia & Bias (`dynamics.metal`)
- **Required**:
  - Composite Rigid Body Algorithm (CRBA) to compute dense $20 \times 20$ joint inertia matrix $M(q)$.
  - Recursive Newton-Euler (RNE) to compute Coriolis, centrifugal, and gravity bias forces $c(q, v)$.
  - Rotor armature diagonal addition: $M_{ii} \mathrel{+}= I_{armature, i}$.
- **Upstream Reference**: MuJoCo Warp `smooth.py:crba`, `smooth.py:rne`, `smooth.py:make_m`.

### 3. Collision Detection & Contact Manifolds (`collision.metal`)
- **Required**:
  - Conservative Broadphase: Bounding sphere rejection to discard definitely separated pairs.
  - Narrowphase: Convex hull support functions and GJK/plane narrowphase for surviving candidate pairs.
  - Foot Contact Manifolds: Multi-point contact manifold generation for sole/ground plane interaction (ensuring stable contact normals and support polygons).
  - Bounded Capacity & Overflow Policy: Enforce `nconmax=35` with atomic contact counting. If contact count exceeds `nconmax`, set an overflow sentinel and fail/retry the step; NEVER silently truncate active contacts.
- **Upstream Reference**: MuJoCo Warp `collision_convex.py`, `collision_gjk.py`, `collision_driver.py`.

### 4. Constraint Jacobian & Solver Equations (`solver.metal`)
- **Required**:
  - Compute contact Jacobians $J \in \mathbb{R}^{n_c \times 20}$ and pyramidal friction cone constraints (4 facets per contact with randomized task friction $\mu$).
  - Include joint limit constraints and BAM frictionloss constraints explicitly in the solver equation.
  - Newton solver with line search (10 iterations, 20 line-search iterations, tolerance $10^{-8}$) solving:
    $$(M - h \frac{\partial f}{\partial v}) \dot{v} + J^T \lambda + J_{limit}^T \lambda_{limit} + \tau_{friction} = \tau - c$$
  - $20 \times 20$ symmetric positive definite system factorization (Cholesky $L L^T$).
  - Warm start acceleration integration from previous step ($qacc\_warmstart$).
- **Upstream Reference**: MuJoCo Warp `solver.py:solve_newton`, `solve_linesearch`, `constraint.py`.

### 5. Integrator: Canonical ImplicitFast & Quaternion Integration (`integrator.metal`)
- **Required**:
  - Velocity derivative correction $\frac{\partial f}{\partial v}$ evaluated via `derivative.deriv_smooth_vel` before factorization.
  - Velocity update: $v_{t+h} = v_t + h \dot{v}$.
  - Quaternion integration matching MuJoCo Warp `math.py:189`:
    $$\Delta \theta = \omega h, \quad \Delta q = \left(\cos\frac{\|\Delta \theta\|}{2}, \frac{\Delta \theta}{\|\Delta \theta\|} \sin\frac{\|\Delta \theta\|}{2}\right)$$
    $$q_{t+h} = \text{normalize}\left(\text{mul\_quat}(q_t, \Delta q)\right) \quad (\text{exact right-multiplication})$$
  - Generalized position update: $q_{joint, t+h} = q_{joint, t} + h v_{joint}$.
- **Upstream Reference**: MuJoCo Warp `forward.py:577`, `math.py:189`.

---

## 4. Phase U1 Implementation Roadmap

If authorized to proceed to Phase U1, work will proceed in five bounded milestones:

```
Milestone U1.1: Kinematics & Articulated Dynamics (CRBA/RNE)
  ├── Implement kinematics.metal & dynamics.metal
  └── Verify M(q) and c(q,v) against MuJoCo CPU across standing & walking poses
Milestone U1.2: Bounded Collision Pipeline & Contact Manifolds
  ├── Implement broadphase sphere rejection + GJK/plane narrowphase
  └── Validate sole/ground and sole/sole contact points and normals vs MuJoCo CPU
Milestone U1.3: Newton Constraint Solver & Pyramidal Friction
  ├── Implement solver.metal with line search and Cholesky 20x20
  └── Verify contact forces and joint accelerations against reference
Milestone U1.4: Canonical ImplicitFast Integrator
  ├── Implement integrator.metal with velocity derivative correction
  └── Verify stability and trajectory matching on multi-step rollouts
Milestone U1.5: Simulation Adapter & Conformance Tests
  ├── Wrap engine in UnifiedMetalSimulation adapter (1 substep per sim.step())
  └── Run deterministic action replays (standing, walking, turning, crossed feet)
```

---

## 5. Summary Table of Phase U0 Artifacts

| Deliverable | Output Path | Status |
| --- | --- | --- |
| **Deliverable 1: Task Inventory** | `reports/task_contract_inventory.md`<br>`configs/canonical_flat_task.json` | Complete & Verified by Pytest |
| **Deliverable 2: Two-Way Buffer Probe** | `src/metal_probe.py`<br>`shaders/probe_ops.metal`<br>`tests/test_shared_buffer.py` | Complete & Verified by Pytest |
| **Deliverable 3: Dispatch Benchmark & Trace** | `reports/metal_probe_results.md`<br>`reports/two_way_ordering_trace.json`<br>`/Volumes/T7/ChatGPOExtension/unified-metal/traces/` | Complete (2.6 $\mu$s dispatch, 0 bytes leak, 0 host copies) |
| **Deliverable 4: Upstream Kernel Audit** | `reports/reusable_kernel_audit.md`<br>`src/audit_reusable_kernels.py` | Complete (Warp & MLX-Cpp audited) |
| **Deliverable 5: Route Decision & Roadmap** | `reports/phase_u0_gate_decision.md` | Complete (Route A Recommended) |
