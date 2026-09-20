# Upstream Reusable Kernel & Algorithm Audit (Verified)

Systematic evaluation of **MuJoCo Warp** and **MuJoCo-MLX-Cpp** against canonical physics requirements.

## 1. Upstream Source Verification

- **MuJoCo Warp** (`/Users/zixiao/workspace/microduck/microduck_rl/.venv/lib/python3.12/site-packages/mujoco_warp/_src`):
  - `math.py:quat_integrate` present: `True` (uses right-multiplication: `True`)
  - `forward.py:deriv_smooth_vel` present: `True`
  - `solver.py:solve_newton` present: `True`
  - `solver.py:solve_linesearch` present: `True`
  - `smooth.py:crba` & `rne` present: `True` / `True`
- **MuJoCo-MLX-Cpp** (`/private/tmp/microduck-mujoco-mlx-cpp-review-20260920/src`):
  - `collision.cpp:plane_mesh_multi` present: `True`
  - Batched path Euler hardcoding: `True`

## 2. Licensing & Attribution

- **MuJoCo Warp**: Apache-2.0 License (Copyright 2024 DeepMind Technologies Limited).
- **MuJoCo-MLX-Cpp**: Apache-2.0 License (Copyright 2024 Genesis Interactive).
- **Obligation**: Any ported algorithms or source fragments must preserve license headers, copyright notices, and original attribution in the adapted Metal files.

## 3. Comprehensive Pipeline Stage Mapping

| Pipeline Stage | Canonical Task Specification | MuJoCo Warp (`_src`) | MuJoCo-MLX-Cpp (`src`) | MSL / Metal Adaptation Path |
| --- | --- | --- | --- | --- |
| **Kinematics (FK)** | Base transform + 14 joints -> body xpos, xquat, site_xpos (8 sites) | Reused (`smooth.py:kinematics`) | Reused (`smooth_vmap.cpp`, `batched.cpp:2180`) | Direct port to MSL: 17 bodies evaluated hierarchically in body tree order. |
| **Articulated Inertia (CRBA)** | Composite Rigid Body Algorithm for dense M(q) (20x20 matrix) | Reused (`smooth.py:crba`) | Reused (`smooth.cpp`, `batched.cpp:2210`) | Direct port to MSL: for nv=20, 20x20 M(q) is small (400 floats = 1.6 KB per env). |
| **Bias Forces (RNE)** | Recursive Newton-Euler for Coriolis, centrifugal, gravity c(q, v) (20 DOF) | Reused (`smooth.py:rne`) | Reused (`smooth.cpp`, `batched.cpp:2240`) | Direct port to MSL: Spatial vector arithmetic (6D lin/ang) maps cleanly to float3x2. |
| **Armature & Joint Damping** | Reflected rotor armature (0.0018 kg m^2) added to M diagonal; joint damping added to velocity derivative | Supported (`smooth.py:make_m`) | Partial | Trivial MSL diagonal addition: M[i, i] += armature[i]. |
| **Conservative Collision Rejection** | Broadphase bounding spheres to discard definitely separated geometry before narrow phase | Supported (`bvh.py`, `collision_driver.py`) | Missing in batched path | Custom MSL broadphase kernel: compute world-space bounding spheres; discard if ||c1 - c2|| > r1 + r2. |
| **CAD Convex Hull Narrow Phase & Manifolds** | Exact CAD mesh convex hull distance, contact points, penetration depths, and normals for sole/ground and self-contacts | Supported (`collision_convex.py`, `collision_gjk.py`) | Scalar only (`collision.cpp:plane_mesh_multi`) | Port MuJoCo's verified plane-mesh manifold algorithm (transform CAD vertices to plane frame, collect penetrating vertices, cluster to supporting polygon). |
| **Capacity & Overflow Handling** | Bounded capacity nconmax=35 with explicit overflow detection and safe retry/fail policy | Supported (`types.py`, `collision_driver.py`) | Silent truncation | Atomic contact count per env in MSL. If count >= nconmax, set overflow flag; env marks step invalid rather than silently truncating physics. |
| **Constraint Solver** | Newton solver with line search (10 iterations, 20 line-search iterations, tol=1e-8), pyramidal friction cone, BAM frictionloss constraints | Full Support (`solver.py:solve_newton`, `solve_linesearch`) | Incompatible | Port MuJoCo Warp's `solver.py` Newton algorithm to MSL. Matrix factorisation for nv=20 is 20x20 Cholesky. |
| **Integrator (ImplicitFast)** | ImplicitFast: velocity derivative correction before solve: (M - h*df/dv)*v_dot = tau + tau_c - c; then v_{t+h} = v_t + h*v_dot | Full Support (`forward.py:577`, `derivative.py`) | Incompatible (`batched.cpp:2453`) | Must implement velocity derivative correction df/dv to achieve ImplicitFast numerical stability. |
| **Quaternion Integration** | Canonical axis-angle increment: delta_theta = omega * h; delta_q = (cos(|theta|/2), theta/|theta| * sin(|theta|/2)); right multiplication: q_{next} = mul_quat(q, delta_q) followed by normalize | Full Support (`math.py:189 quat_integrate`) | Supported (`batched.cpp`) | Direct MSL port of Warp's `quat_integrate` (right multiplication with axis-angle increment, not simplified first-order left multiplication). |
| **Actuator Dynamics (BAM M6)** | FrictionDRBamActuator: Stribeck + load-dependent friction, motor delay buffer, voltage control, per-env friction scale | Upstream does not contain BAM | Upstream does not contain BAM | Key Architectural Benefit: BAM already runs in PyTorch on MPS in `action_manager.apply_action()`! Retain in PyTorch; no Metal port required. |

## 4. Key Upstream Takeaways for Metal Engine Design

1. **MuJoCo Warp is the Superior Mathematical Reference**:
   - Warp implements the exact canonical algorithms: `ImplicitFast` with velocity derivatives (`forward.py:577`), `Newton` solver with line search (`solver.py`), pyramidal friction cones (`constraint.py`), and robust GJK convex hull distance (`collision_gjk.py`).
   - MLX-Cpp's batched path took major shortcuts: hardcoded Euler integrator, simplified PGS solver, single-vertex foot contact, and lack of overflow detection.

2. **Quaternion Integration Mathematical Exactness**:
   - Must follow `math.py:189` `quat_integrate`: compute axis-angle increment $\Delta \theta = \omega h$, quaternion $\Delta q$, and right-multiply $q_{next} = \text{mul\_quat}(q, \Delta q)$ followed by normalization.

3. **BAM Actuator Stays in PyTorch**:
   - Because `mjlab`'s decimation loop (`manager_based_rl_env.py:414–421`) executes `action_manager.apply_action()` before each `sim.step()`, PyTorch's native MPS implementation of `FrictionDRBamActuator` runs directly on the shared MPS tensors.
   - No porting of BAM to Metal is necessary, eliminating a major source of potential motor model discrepancy.

4. **Small Model Dimension Advantage ($nv=20$)**:
   - For MicroDuck ($nv=20$), the mass matrix $M$ is only $20 \times 20$ (400 floats = 1.6 KB per environment).
   - Cholesky factorization and forward/back-substitution for $20 \times 20$ matrices can easily fit entirely within threadgroup memory on Apple Silicon GPUs without memory bandwidth bottlenecks.
