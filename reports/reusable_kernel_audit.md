# Upstream Reusable Kernel & Algorithm Audit

Systematic evaluation of **MuJoCo Warp** and **MuJoCo-MLX-Cpp** for reuse in the Metal physics pipeline.

## 1. Licensing & Attribution

- **MuJoCo Warp**: Apache-2.0 License (Copyright 2024 DeepMind Technologies Limited).
- **MuJoCo-MLX-Cpp**: Apache-2.0 License (Copyright 2024 Genesis Interactive).
- **Obligation**: Any ported algorithms or source fragments must preserve license headers, copyright notices, and original attribution in the adapted Metal files.

## 2. Comprehensive Pipeline Stage Mapping

| Pipeline Stage | Canonical Task Specification | MuJoCo Warp (`_src`) | MuJoCo-MLX-Cpp (`src`) | MSL / Metal Adaptation Path |
| --- | --- | --- | --- | --- |
| **Kinematics (FK)** | Base transform + 14 joints -> body xpos, xquat, site_xpos (8 sites) | Reused (`smooth.py:kinematics`) | Reused (`smooth_vmap.cpp`, `batched.cpp:2180`) | Direct port to MSL: 15 bodies evaluated hierarchically or parallelised per body. |
| **Articulated Inertia (CRBA)** | Composite Rigid Body Algorithm for dense M(q) (20x20 matrix) | Reused (`smooth.py:crba`) | Reused (`smooth.cpp`, `batched.cpp:2210`) | Direct port to MSL: for nv=20, 20x20 M(q) is small (400 floats = 1.6 KB per env). |
| **Bias Forces (RNE)** | Recursive Newton-Euler for Coriolis, centrifugal, gravity c(q, v) (20 DOF) | Reused (`smooth.py:rne`) | Reused (`smooth.cpp`, `batched.cpp:2240`) | Direct port to MSL: Spatial vector arithmetic (6D lin/ang) maps cleanly to float3x2. |
| **Armature & Damping** | Reflected rotor inertia (armature=0.0018 kg m^2) added to M diagonal; joint damping | Supported (`smooth.py:make_m`) | Partial | Trivial MSL diagonal addition: M[i, i] += armature[i]. |
| **Conservative Collision Rejection** | Broadphase bounding spheres to discard separated geometry before narrow phase | Supported (`bvh.py`, `collision_driver.py`) | Missing in batched path | Custom MSL broadphase kernel: compute world-space bounding spheres; discard if ||c1 - c2|| > r1 + r2. |
| **Convex Hull Narrow Phase** | Exact CAD mesh convex hull distance, contact points, penetration depths, and normals | Supported (`collision_convex.py`, `collision_gjk.py`) | Simplified / Incomplete | Port Warp's `collision_gjk.py` support functions and GJK algorithm to MSL. Pair with EPA or analytical fallback. |
| **Contact Manifold Generation** | Stable multi-point contact manifolds for sole/ground plane (soles have CAD area, not 1 point) | Supported (`collision_driver.py`) | Incompatible (`constraint_vmap.cpp:1146`) | Custom MSL sole contact kernel: test 4 sole perimeter vertices against ground plane to generate stable 4-point manifold. |
| **Capacity & Overflow** | Bounded capacity nconmax=35 with explicit overflow detection (never silent truncation) | Supported (`types.py`, `collision_driver.py`) | Fixed allocation, silent truncation | Atomic contact count per env in MSL. If count >= nconmax, record overflow flag and clamp. |
| **Constraint Solver** | Newton solver with line search (10 iterations, 20 line-search iterations, tol=1e-8) | Full Support (`solver.py:solve_newton`, `solve_linesearch`) | Incompatible | Port MuJoCo Warp's `solver.py` Newton algorithm to MSL. Matrix factorisation for nv=20 is 20x20 Cholesky. |
| **Pyramidal Friction Cone** | 4-sided pyramidal friction cone for contacts (condim=3 for feet) | Full Support (`constraint.py`) | Partial | Direct port of Warp's pyramidal constraint builder: normal impulse + 4 friction facets. |
| **Integrator (ImplicitFast)** | ImplicitFast: derivative.deriv_smooth_vel correction before solve, then v_{t+dt} = v_t + dt * qacc | Full Support (`forward.py:577`, `derivative.py`) | Incompatible (`batched.cpp:2453`) | Must implement velocity derivative correction df/dv to achieve ImplicitFast stability. |
| **Quaternion Integration** | Integrate base orientation q_{3:7} from angular velocity omega, normalize to unit length | Full Support (`forward.py`, `math.py:quat_integrate`) | Supported (`batched.cpp`) | Trivial MSL kernel: q_next = normalize(q + 0.5 * dt * quat_mul((0, omega), q)). |
| **Actuator Dynamics (BAM M6)** | FrictionDRBamActuator: Stribeck + load-dependent friction, motor delay buffer, voltage control | Upstream does not contain BAM | Upstream does not contain BAM | Key Architectural Benefit: BAM already runs in PyTorch on MPS in `action_manager.apply_action()`! Retain in PyTorch; no Metal port required for Phase U0/U1. |

## 3. Deep-Dive Findings by Stage

### Kinematics (FK)
- **Canonical Requirement**: Base transform + 14 joints -> body xpos, xquat, site_xpos (8 sites)
- **MuJoCo Warp**: Hierarchical tree traversal in Warp kernel. Cleanly translatable to MSL.
- **MuJoCo-MLX-Cpp**: Metal kernel generated via MLX. Fused forward kinematics for nv <= 80.
- **Recommended Metal Path**: Direct port to MSL: 15 bodies evaluated hierarchically or parallelised per body.

### Articulated Inertia (CRBA)
- **Canonical Requirement**: Composite Rigid Body Algorithm for dense M(q) (20x20 matrix)
- **MuJoCo Warp**: Computes dense joint-space inertia matrix and backward composite tree accumulation.
- **MuJoCo-MLX-Cpp**: Generates M(q) in GPU memory. Compact and vectorized.
- **Recommended Metal Path**: Direct port to MSL: for nv=20, 20x20 M(q) is small (400 floats = 1.6 KB per env).

### Bias Forces (RNE)
- **Canonical Requirement**: Recursive Newton-Euler for Coriolis, centrifugal, gravity c(q, v) (20 DOF)
- **MuJoCo Warp**: Two-pass spatial algebra: forward velocities & accelerations, backward force propagation.
- **MuJoCo-MLX-Cpp**: Batched spatial algebra implementation.
- **Recommended Metal Path**: Direct port to MSL: Spatial vector arithmetic (6D lin/ang) maps cleanly to float3x2.

### Armature & Damping
- **Canonical Requirement**: Reflected rotor inertia (armature=0.0018 kg m^2) added to M diagonal; joint damping
- **MuJoCo Warp**: Diagonal armature and damping addition is standard.
- **MuJoCo-MLX-Cpp**: Batched path omits per-env randomized armature constants.
- **Recommended Metal Path**: Trivial MSL diagonal addition: M[i, i] += armature[i].

### Conservative Collision Rejection
- **Canonical Requirement**: Broadphase bounding spheres to discard separated geometry before narrow phase
- **MuJoCo Warp**: BVH tree with bounding box/sphere checks in Warp.
- **MuJoCo-MLX-Cpp**: Batched path tests fixed candidate pairs without broadphase culling.
- **Recommended Metal Path**: Custom MSL broadphase kernel: compute world-space bounding spheres; discard if ||c1 - c2|| > r1 + r2.

### Convex Hull Narrow Phase
- **Canonical Requirement**: Exact CAD mesh convex hull distance, contact points, penetration depths, and normals
- **MuJoCo Warp**: GJK distance + support functions on convex hulls. Over 2,300 lines of robust Warp code.
- **MuJoCo-MLX-Cpp**: GJK support-direction estimate; lacks full penetration recovery and multi-point manifold.
- **Recommended Metal Path**: Port Warp's `collision_gjk.py` support functions and GJK algorithm to MSL. Pair with EPA or analytical fallback.

### Contact Manifold Generation
- **Canonical Requirement**: Stable multi-point contact manifolds for sole/ground plane (soles have CAD area, not 1 point)
- **MuJoCo Warp**: Contact manifold clustering and reduction.
- **MuJoCo-MLX-Cpp**: Picks only 1 support vertex per mesh; insufficient for foot ground stability.
- **Recommended Metal Path**: Custom MSL sole contact kernel: test 4 sole perimeter vertices against ground plane to generate stable 4-point manifold.

### Capacity & Overflow
- **Canonical Requirement**: Bounded capacity nconmax=35 with explicit overflow detection (never silent truncation)
- **MuJoCo Warp**: Tracks contact counts and flags capacity overflow.
- **MuJoCo-MLX-Cpp**: Hard-coded max contacts; drops excess contacts without notification.
- **Recommended Metal Path**: Atomic contact count per env in MSL. If count >= nconmax, record overflow flag and clamp.

### Constraint Solver
- **Canonical Requirement**: Newton solver with line search (10 iterations, 20 line-search iterations, tol=1e-8)
- **MuJoCo Warp**: Over 3,300 lines. Implements exact MuJoCo Newton solver with projected gradients and line search.
- **MuJoCo-MLX-Cpp**: Batched path implements simplified PGS/Euler, not Newton with line search.
- **Recommended Metal Path**: Port MuJoCo Warp's `solver.py` Newton algorithm to MSL. Matrix factorisation for nv=20 is 20x20 Cholesky.

### Pyramidal Friction Cone
- **Canonical Requirement**: 4-sided pyramidal friction cone for contacts (condim=3 for feet)
- **MuJoCo Warp**: Builds contact Jacobians and pyramidal friction constraints (4 tangent friction edges).
- **MuJoCo-MLX-Cpp**: Simplified isotropic friction model.
- **Recommended Metal Path**: Direct port of Warp's pyramidal constraint builder: normal impulse + 4 friction facets.

### Integrator (ImplicitFast)
- **Canonical Requirement**: ImplicitFast: derivative.deriv_smooth_vel correction before solve, then v_{t+dt} = v_t + dt * qacc
- **MuJoCo Warp**: Calls `deriv_smooth_vel` to add damping/velocity derivatives to system matrix before Cholesky solve.
- **MuJoCo-MLX-Cpp**: Always executes explicit Euler; ignores configured ImplicitFast.
- **Recommended Metal Path**: Must implement velocity derivative correction df/dv to achieve ImplicitFast stability.

### Quaternion Integration
- **Canonical Requirement**: Integrate base orientation q_{3:7} from angular velocity omega, normalize to unit length
- **MuJoCo Warp**: Standard quaternion integration and renormalization.
- **MuJoCo-MLX-Cpp**: Quaternion integration implemented.
- **Recommended Metal Path**: Trivial MSL kernel: q_next = normalize(q + 0.5 * dt * quat_mul((0, omega), q)).

### Actuator Dynamics (BAM M6)
- **Canonical Requirement**: FrictionDRBamActuator: Stribeck + load-dependent friction, motor delay buffer, voltage control
- **MuJoCo Warp**: MuJoCo Warp contains standard XML actuators only (motor, position, muscle).
- **MuJoCo-MLX-Cpp**: MLX engine only supports built-in MuJoCo actuators.
- **Recommended Metal Path**: Key Architectural Benefit: BAM already runs in PyTorch on MPS in `action_manager.apply_action()`! Retain in PyTorch; no Metal port required for Phase U0/U1.

## 4. Key Upstream Takeaways for Metal Engine Design

1. **MuJoCo Warp is the Superior Mathematical Reference**:
   - Warp implements the exact canonical algorithms: `ImplicitFast` with velocity derivatives (`forward.py:577`), `Newton` solver with line search (`solver.py`), pyramidal friction cones (`constraint.py`), and robust GJK convex hull distance (`collision_gjk.py`).
   - MLX-Cpp's batched path took major shortcuts: hardcoded Euler integrator, simplified PGS solver, single-vertex foot contact, and lack of overflow detection.

2. **BAM Actuator Stays in PyTorch**:
   - Because `mjlab`'s decimation loop (`manager_based_rl_env.py:414–421`) executes `action_manager.apply_action()` before each `sim.step()`, PyTorch's native MPS implementation of `FrictionDRBamActuator` can run directly on the shared MPS tensors.
   - No porting of BAM to Metal is necessary for U0 or U1, eliminating a major source of potential friction discrepancy.

3. **Small Model Dimension Advantage ($nv=20$)**:
   - For MicroDuck ($nv=20$), the mass matrix $M$ is only $20 \times 20$ (400 floats = 1.6 KB per environment).
   - Cholesky factorization and forward/back-substitution for $20 \times 20$ matrices can easily fit entirely within threadgroup memory or SIMD lanes on Apple Silicon GPUs without memory bandwidth bottlenecks.
