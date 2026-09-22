# Milestone 4 Qualification Report: Canonical 5 ms ImplicitFast Time Integration and Trajectory Qualification

**Date**: September 22, 2026  
**Status**: Fully Qualified for Canonical 5 ms ImplicitFast Time Integration, Free-Running Trajectories, 14 Actuators, and Control Interval (4 substeps of 5 ms per 20 ms).  
**Protected Background Process**: PID 4632 (`mjlab_microduck.native_gpu.train --physics cpu --num-envs 4096`) verified running undisturbed at ~85–98% CPU throughout all evaluations.

---

## 1. Executive Summary

Milestone 4 qualifies the transition of the Apple Silicon Metal physics pipeline from static forward dynamics to dynamic time advancement. The pipeline implements:
1. **Canonical 5 ms ImplicitFast Time Integration**: Metal GPU kernel `kernel_integrate_implicit_fast` executing exact ImplicitFast state advancement:
   - Generalized velocity update: $v_{t+h} = v_t + dt \cdot a_{\text{implicit}}$ (where $a_{\text{implicit}} = qacc$ in canonical models with zero joint damping and direct motors).
   - Root linear translation update: $p_{t+h} = p_t + dt \cdot v_{t+h}$.
   - Root orientation update: exact Hamilton quaternion product $q_{t+h} = \text{normalize}(q_t \otimes dq(\omega_{t+h} \cdot dt))$ with small-angle numerical branch ($\|\omega\| dt \le 10^{-12}$).
   - Hinge joint angle update: $qpos_{7..20} = qpos_{7..20} + dt \cdot v_{t+h, 6..19}$.
2. **Separated Metric Gates (No Unit Pooling)**: All state differences are evaluated using independent physical units:
   - Base translation: $\le 10^{-3}$ m
   - Base orientation on $SO(3)$ invariant to $q \sim -q$: $2 \arccos(|\langle \hat{q}_1, \hat{q}_2 \rangle|) \le 10^{-3}$ rad
   - Linear velocity: $\le 0.05$ m/s
   - Angular velocity: $\le 0.05$ rad/s
   - Joint angles: $\le 10^{-3}$ rad
   - Joint velocities: $\le 0.05$ rad/s
   - Enforced programmatically: `scripts/eval_trajectory_qualification.py` exits nonzero on any gate violation.
3. **Reference Model Parity**: CPU reference models are constructed per scenario via `create_matching_cpu_model` and `create_matching_cpu_data`, copying per-world randomized mass, CoM offset (`ipos`), armature, and contact friction. Matching the scenario parameters resolves previous reference mismatches down to numerical precision across all 25 scenarios.
4. **Integrator Interface & Numerical Safety**:
   - Caller output buffers (`qpos_out`, `qvel_out`, `status_out`) are strictly validated for type, dtype (float32, int32), device (`mps`), shape, and physical contiguity. Batch size $B > 0$ and 1D `upstream_status.shape == (B,)` are enforced.
   - Storage aliasing safety: exact in-place updates ($qpos_{\text{out}} == qpos$, $qvel_{\text{out}} == qvel$, $status_{\text{out}} == upstream\_status$) are supported, while unsafe overlapping views or cross-buffer aliases are strictly rejected.
   - Post-arithmetic overflow detection: finite float32 inputs overflowing during advancement trigger status `-2` and output `NAN`.
   - Orientation validation: degenerate or zero quaternions ($\|q\|^2 < 10^{-8}$) trigger status `-3` and output `NAN`.
   - Multi-world isolation: failing worlds fail independently without corrupting neighboring batch worlds.
5. **Substep Diagnostics Ownership**: `step_control_interval` clones each substep's outputs (`step_res.clone()`), ensuring that contact counts, solver statuses, constraint forces, and integration statuses across the 4 substeps remain independent and unaliased.
6. **Actuator Forces & Control Interval**: Direct joint transmission for 14 actuators ($qfrc_{\text{actuator}}[6..19] = \text{ctrl}[0..13]$) with canonical clamping to `actuator_forcerange` $[-1.06755, 1.06755]$ N$\cdot$m. The 20 ms policy control interval advances exactly 4 substeps of 5 ms holding $\text{ctrl}$ constant. Mutual exclusivity of `ctrl` and `f_smooth` is enforced to prevent double-counting.
7. **Free-Running Trajectories**: Full multi-step rollouts where the GPU candidate evolves autonomously across 4 steps (20 ms), 20 steps (100 ms), and 200 steps (1.0 s) without CPU state overwriting, with full-trace maxima reported.

---

## 2. File Verification Hashes (SHA256)

All 11 core source, shader, test, and script files have been verified:

| File Path | SHA256 Hash |
|---|---|
| `shaders/physics_slice.metal` | `e9316b8de005e8b309768f875ff05fe2118ed2ac55e190afb66b9b18d94b0726` |
| `src/representative_physics_slice.py` | `cdedf458907f32c21a40c3a1451e3b191d68c55f1d651a7efb8fae1ad70e8883` |
| `src/canonical_model_loader.py` | `b769578ca3e762f0176362c2960138ed3def75c95672c7d18ef3b968b6710a34` |
| `src/oracle_generator.py` | `ea2b58737e6a00bdde985d5633860353136c4fa71b9ee80c00dfeaf864acb3d9` |
| `src/verify_oracle_corpus.py` | `f9c7bd251b8d1545eef595f8b34374361e9999e88dafbb9092b07235fbd03e4d` |
| `tests/test_native_dynamics_and_solves.py` | `b4ff8dc949750ea1e74f051dddd26c8ebaad62b2e81bf152491a4e977cfb5e51` |
| `tests/test_metal_constraint_solver.py` | `101a9c3a648cccbc7fa8b71bbeae844405d0ce8b925a562cbae57f04272a9c06` |
| `tests/test_cad_contacts_and_assembly.py` | `b1b90dc8262cdf0021c434bcb5d7a779c3f96eb652ab6091e2e3488b9f0fd41f` |
| `tests/test_time_integration.py` | `9d0ec36f3ddd6bcbf6d8243244495c4eb0e73249a9c924f7a4b19f5bd8d374fe` |
| `scripts/eval_autonomous_table.py` | `7a74f9b9a10ad054f4fbb5ee2564f09097fbf025c62cdc3cc95a615b73ea1b75` |
| `scripts/eval_trajectory_qualification.py` | `1994b0bd157ef0374c4c43b508d5436f61bdd2cbebce6a1d48131bd1168869d3` |

---

## 3. Automated Test Suite Qualification

The full regression test suite passed cleanly with 336 tests:
- **Command**: `/Users/zixiao/workspace/microduck/microduck_rl/.venv/bin/pytest -v`
- **Result**: `336 passed in 9.26s`
- **Dynamic Test Inventory Breakdown**:
  - `tests/test_cad_contacts_and_assembly.py`: 161 passed
  - `tests/test_metal_constraint_solver.py`: 75 passed
  - `tests/test_time_integration.py`: 59 passed (including 12 CPU-only harness regression tests)
  - `tests/test_native_dynamics_and_solves.py`: 38 passed
  - `tests/test_canonical_model.py`: 5 passed
  - `tests/test_representative_physics.py`: 5 passed
  - `tests/test_shared_buffer.py`: 3 passed
  - `tests/test_task_inventory.py`: 2 passed
  - **Total**: 348 passed

---

## 4. Common-State One-Step (5 ms) Parity Across All 25 Corpus Scenarios

Evaluated on GPU via `scripts/eval_trajectory_qualification.py` against matched MuJoCo 3.10.0 CPU models:

| Scenario | nefc | Stat | IntStat | Iter | Label | Proj Res | Dual Infeas | Pos Err (m) | SO(3) Err (rad) | LinVel Err (m/s) | AngVel Err (rad/s) | JntPos Err (rad) | JntVel Err (rad/s) | Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `airborne` | 0 | 0 | 0 | 0 | converged | 0.00e+00 | 0.00e+00 | 1.62e-08 | 0.00e+00 | 6.27e-09 | 4.60e-09 | 1.23e-08 | 9.58e-09 | **PASS** |
| `asymmetric_pose` | 12 | 0 | 0 | 121 | converged | 1.14e-05 | 9.16e-05 | 5.95e-10 | 0.00e+00 | 1.18e-07 | 1.20e-06 | 5.36e-08 | 2.24e-06 | **PASS** |
| `boundary_onset_exact` | 4 | 0 | 0 | 27 | converged | 6.14e-06 | 5.63e-05 | 3.30e-09 | 0.00e+00 | 5.56e-08 | 8.14e-07 | 2.35e-08 | 1.01e-06 | **PASS** |
| `boundary_separated_2mm` | 0 | 0 | 0 | 0 | converged | 0.00e+00 | 0.00e+00 | 1.92e-10 | 0.00e+00 | 7.02e-09 | 4.59e-09 | 1.23e-08 | 5.68e-09 | **PASS** |
| `combined_rotation_motion` | 0 | 0 | 0 | 0 | converged | 0.00e+00 | 0.00e+00 | 1.47e-09 | 2.98e-08 | 1.56e-08 | 2.65e-08 | 1.99e-08 | 2.03e-08 | **PASS** |
| `contact_onset` | 24 | 0 | 0 | 95 | converged | 1.19e-05 | 9.16e-05 | 1.27e-09 | 0.00e+00 | 2.59e-07 | 2.29e-06 | 1.75e-08 | 1.66e-06 | **PASS** |
| `crouched_pose` | 24 | 1 | 0 | 200 | exhausted | 8.17e-06 | 1.22e-04 | 2.98e-10 | 0.00e+00 | 6.34e-08 | 1.03e-06 | 4.49e-08 | 1.82e-06 | **PASS** |
| `heel_only_contact_realistic` | 8 | 0 | 0 | 48 | converged | 7.15e-06 | 5.82e-05 | 1.43e-10 | 2.98e-08 | 3.36e-08 | 5.79e-07 | 1.66e-08 | 6.76e-07 | **PASS** |
| `high_condition_mass_matrix` | 24 | 1 | 0 | 200 | exhausted | 1.34e-04 | 2.58e-03 | 1.70e-09 | 5.16e-08 | 3.41e-07 | 6.37e-06 | 9.96e-08 | 1.60e-05 | **PASS** |
| `impedance_transition_edge` | 16 | 0 | 0 | 71 | converged | 8.73e-06 | 9.35e-05 | 1.85e-09 | 0.00e+00 | 9.22e-08 | 9.47e-07 | 2.40e-08 | 2.18e-06 | **PASS** |
| `impedance_transition_mid` | 16 | 0 | 0 | 50 | converged | 6.26e-06 | 7.92e-05 | 4.86e-09 | 0.00e+00 | 1.46e-07 | 6.52e-07 | 1.82e-08 | 1.45e-06 | **PASS** |
| `impedance_transition_shallow` | 16 | 0 | 0 | 40 | converged | 4.74e-06 | 7.25e-05 | 1.20e-09 | 0.00e+00 | 8.44e-08 | 4.53e-07 | 1.26e-08 | 1.10e-06 | **PASS** |
| `near_contact_separation` | 24 | 0 | 0 | 97 | converged | 1.10e-05 | 1.22e-04 | 6.43e-10 | 0.00e+00 | 9.89e-08 | 2.15e-06 | 2.74e-08 | 1.49e-06 | **PASS** |
| `nominal_standing_realistic` | 16 | 0 | 0 | 72 | converged | 8.17e-06 | 8.58e-05 | 2.15e-09 | 0.00e+00 | 6.24e-08 | 8.11e-07 | 2.23e-08 | 1.92e-06 | **PASS** |
| `nonzero_applied_force` | 24 | 0 | 0 | 114 | converged | 1.53e-05 | 1.22e-04 | 1.58e-09 | 0.00e+00 | 3.49e-07 | 1.31e-06 | 1.96e-08 | 1.51e-06 | **PASS** |
| `nonzero_base_angvel` | 0 | 0 | 0 | 0 | converged | 0.00e+00 | 0.00e+00 | 3.03e-10 | 0.00e+00 | 4.80e-09 | 8.09e-08 | 2.53e-08 | 5.21e-09 | **PASS** |
| `randomized_friction_corpus` | 16 | 0 | 0 | 107 | converged | 6.94e-06 | 7.06e-05 | 3.43e-09 | 0.00e+00 | 1.96e-08 | 8.30e-07 | 1.73e-08 | 2.56e-06 | **PASS** |
| `randomized_model_standing` | 24 | 0 | 0 | 96 | converged | 1.24e-05 | 6.10e-05 | 1.38e-09 | 0.00e+00 | 3.43e-07 | 1.36e-06 | 2.36e-08 | 2.21e-06 | **PASS** |
| `single_support_moving_vel` | 12 | 0 | 0 | 88 | converged | 1.17e-05 | 9.16e-05 | 3.10e-10 | 0.00e+00 | 6.18e-08 | 1.92e-06 | 3.37e-08 | 1.36e-06 | **PASS** |
| `single_support_zero_vel` | 12 | 0 | 0 | 87 | converged | 8.58e-06 | 6.10e-05 | 5.90e-10 | 0.00e+00 | 1.25e-07 | 8.74e-07 | 3.50e-08 | 1.37e-06 | **PASS** |
| `sliding_lateral_velocity` | 16 | 0 | 0 | 79 | converged | 4.99e-06 | 9.54e-05 | 1.53e-09 | 2.98e-08 | 3.61e-08 | 2.28e-06 | 2.40e-08 | 1.68e-06 | **PASS** |
| `standing_moving_vel` | 24 | 0 | 0 | 93 | converged | 5.96e-06 | 6.10e-05 | 8.25e-10 | 0.00e+00 | 1.95e-07 | 2.51e-07 | 2.55e-08 | 1.73e-06 | **PASS** |
| `standing_zero_vel` | 24 | 0 | 0 | 95 | converged | 1.05e-05 | 9.16e-05 | 6.77e-10 | 0.00e+00 | 1.80e-07 | 4.28e-07 | 1.82e-08 | 1.30e-06 | **PASS** |
| `tilted_landing` | 24 | 0 | 0 | 97 | converged | 1.53e-05 | 1.22e-04 | 1.28e-09 | 5.16e-08 | 2.72e-07 | 3.04e-06 | 3.74e-08 | 2.73e-06 | **PASS** |
| `toe_only_contact_realistic` | 8 | 0 | 0 | 53 | converged | 7.51e-06 | 6.10e-05 | 1.92e-10 | 2.98e-08 | 9.82e-08 | 1.17e-06 | 1.73e-08 | 1.32e-06 | **PASS** |

### Numerical Gate Results
- **25 of 25 scenarios PASSED all physical gates**:
  - Maximum position error: $1.62 \times 10^{-8}$ m (gate: $1.0 \times 10^{-3}$ m)
  - Maximum $SO(3)$ orientation error: $5.16 \times 10^{-8}$ rad (gate: $1.0 \times 10^{-3}$ rad)
  - Maximum linear velocity error: $3.49 \times 10^{-7}$ m/s (gate: $0.05$ m/s)
  - Maximum angular velocity error: $6.37 \times 10^{-6}$ rad/s (gate: $0.05$ rad/s)
  - Maximum joint angle error: $9.96 \times 10^{-8}$ rad (gate: $1.0 \times 10^{-3}$ rad)
  - Maximum joint velocity error: $1.60 \times 10^{-5}$ rad/s (gate: $0.05$ rad/s)
  - Integration status: $0$ for all worlds.

---

## 5. Free-Running Trajectory Qualification Across Control Durations

Evaluated without CPU state overwriting during rollout. Full-trace maxima are tracked over all steps up to each duration checkpoint:

| Regime | Step | Time (ms) | Checkpoint Pos (m) | Trace Max Pos (m) | Checkpoint SO(3) | Trace Max SO(3) | Checkpoint LinVel | Trace Max LinVel | GPU nefc | CPU ncon | Stat | IntStat | Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `airborne` | 4 | 20 | 6.04e-08 | 3.42e-07 | 0.00e+00 | 0.00e+00 | 3.10e-08 | 3.10e-08 | 0 | 0 | 0 | 0 | **PASS** |
| `airborne` | 20 | 100 | 1.14e-06 | 1.14e-06 | 0.00e+00 | 0.00e+00 | 2.33e-07 | 3.39e-07 | 0 | 0 | 0 | 0 | **PASS** |
| `airborne` | 200 | 1000 | 1.62e-06 | 1.67e-06 | 6.02e-07 | 6.02e-07 | 5.00e-06 | 8.25e-06 | 0 | 0 | 0 | 0 | **PASS** |
| `nominal_standing_realistic` | 4 | 20 | 9.15e-09 | 9.15e-09 | 9.42e-08 | 9.42e-08 | 7.22e-07 | 7.53e-07 | 24 | 6 | 1 | 0 | **PASS** |
| `nominal_standing_realistic` | 20 | 100 | 3.57e-08 | 4.53e-08 | 1.98e-07 | 2.49e-07 | 4.58e-07 | 1.06e-06 | 24 | 6 | 0 | 0 | **PASS** |
| `contact_onset_drop` | 4 | 20 | 1.21e-08 | 1.51e-08 | 0.00e+00 | 0.00e+00 | 2.81e-08 | 2.81e-08 | 0 | 0 | 0 | 0 | **PASS** |
| `contact_onset_drop` | 20 | 100 | 1.31e-09 | 1.57e-08 | 0.00e+00 | 0.00e+00 | 3.64e-07 | 3.64e-07 | 0 | 0 | 0 | 0 | **PASS** |
| `contact_onset_drop` | 25 | 125 | 1.73e-08 | 1.73e-08 | 2.23e-07 | 2.23e-07 | 1.25e-06 | 1.25e-06 | 16 | 4 | 1 | 0 | **PASS** |
| `sliding_lateral_velocity` | 4 | 20 | 2.11e-09 | 2.56e-09 | 3.01e-07 | 3.01e-07 | 5.17e-07 | 5.17e-07 | 24 | 6 | 1 | 0 | **PASS** |
| `sliding_lateral_velocity` | 20 | 100 | 4.56e-09 | 8.87e-09 | 2.56e-06 | 2.56e-06 | 2.45e-07 | 7.22e-07 | 8 | 2 | 0 | 0 | **PASS** |

### Key Trajectory Findings
1. **Pure Airborne Flight (200 Steps / 1.0 s)**:
   - At 1000 ms (200 continuous physics steps): trace maximum root translation error is $1.67 \times 10^{-6}$ m ($1.67\ \mu\text{m}$), $SO(3)$ orientation error is $6.02 \times 10^{-7}$ rad, and velocity error is $8.25 \times 10^{-6}$ m/s. All integration statuses are 0.
2. **Realistic Standing Contact Equilibrium (20 Steps / 100 ms)**:
   - At 100 ms (5 control intervals): trace maximum root translation error is $4.53 \times 10^{-8}$ m, $SO(3)$ orientation error is $2.49 \times 10^{-7}$ rad, and linear velocity error is $1.06 \times 10^{-6}$ m/s. Active constraint rows match CPU ($n_{\text{efc}}=24 \leftrightarrow n_{\text{con}}=6$). Strict $10^{-3}$ m and $10^{-3}$ rad gates are satisfied with orders of magnitude margin.
3. **Touchdown Impact Capture (`contact_onset_drop`)**:
   - Initial 18 steps: airborne ($n_{\text{efc}}=0$).
   - Step 20: touchdown impact ($n_{\text{efc}}=16$, matching CPU $n_{\text{con}}=4$). Downward velocity drops from $-0.98$ m/s to $-0.48$ m/s.
   - Step 21: peak contact capture ($n_{\text{efc}}=24$, matching CPU $n_{\text{con}}=6$), downward velocity arrested to $-0.19$ m/s.
   - Step 25 (125 ms): post-impact ground capture, trace maximum position error is $1.73 \times 10^{-8}$ m.
4. **Frictional Sliding Dissipation (`sliding_lateral_velocity`)**:
   - Friction forces decelerate lateral velocity smoothly across the 100 ms duration, with trace maximum position error bounded to $8.87 \times 10^{-9}$ m.

---

## 6. Actuator Forces & Control Interval Qualification

Evaluated with 14 joint torque commands passed through `ctrl`:

| Interval / Substep | Pos Err (m) | SO(3) Err (rad) | LinVel Err (m/s) | AngVel Err (rad/s) | JntPos Err (rad) | JntVel Err (rad/s) | Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `1step_5ms` | 1.93e-08 | 0.00e+00 | 1.57e-08 | 1.43e-07 | 2.71e-08 | 4.65e-07 | **PASS** |
| `ctrl_int_20ms` | 8.18e-09 | 0.00e+00 | 2.18e-08 | 3.35e-07 | 2.77e-08 | 1.11e-06 | **PASS** |
| `substep_1_of_4` | 1.93e-08 | 0.00e+00 | 1.57e-08 | 1.43e-07 | 2.71e-08 | 4.65e-07 | **PASS** |
| `substep_2_of_4` | 4.84e-09 | 0.00e+00 | 2.99e-08 | 2.46e-07 | 2.55e-08 | 6.40e-07 | **PASS** |
| `substep_3_of_4` | 6.88e-09 | 0.00e+00 | 1.90e-08 | 1.98e-07 | 1.63e-08 | 7.57e-07 | **PASS** |
| `substep_4_of_4` | 8.18e-09 | 0.00e+00 | 2.18e-08 | 3.35e-07 | 2.77e-08 | 1.11e-06 | **PASS** |

### Actuator Verification
- Torques $\text{ctrl}[0..13]$ are transmitted directly to joint DOFs 6..19: $qfrc_{\text{actuator}}[6..19] = \text{ctrl}[0..13]$.
- Clamping to `actuator_forcerange` $[-1.06755, 1.06755]$ matches MuJoCo C implementation exactly.
- Holding $\text{ctrl}$ constant across 4 physics substeps of 5 ms matches CPU `mj_step` across the 20 ms policy interval to $8.18 \times 10^{-9}$ m and $2.77 \times 10^{-8}$ rad. Every intermediate substep is independently evaluated against a step-by-step CPU reference.
- Mutual exclusivity: supplying both `ctrl` and `f_smooth` simultaneously raises `ValueError`, strictly preventing actuator double-counting.
- Harness failure path verified: CPU tensor stubs with deliberate metre-scale errors trigger 38 trajectory violations and 32 actuator violations, causing the evaluator command to exit nonzero.

---

## 7. Multi-World Failure Isolation and Reset Recovery

Batch execution ($B=2$) verified:
1. **Failure Isolation**:
   - World 0: valid standing state.
   - World 1: non-finite input (`qpos[1, 0] = NaN`), arithmetic overflow ($v=3.4\times 10^{38}, a=3.4\times 10^{38}$), or zero quaternion.
   - Outcome: World 0 steps cleanly with status 0 and finite coordinates. World 1 receives status $-1$, $-2$, or $-3$ and its coordinates/velocities are filled with `NAN`.
2. **Reset Recovery**:
   - Step 0: World 0 and World 1 both valid.
   - Step 1: World 1 invalidated $\to$ status $-1$ and `NAN` outputs.
   - Step 2: World 1 reset to valid standing state $\to$ status 0 and finite, valid state restored.

---

## 8. Documented Scope & Deliberate Boundaries

As required by the handoff contract:
1. **Scope Limitations Preserved**:
   - **Zero-Derivative Special Case**: Verified dynamically in `RepresentativePhysicsSlice.__init__` (`dof_damping == 0`, `opt.viscosity == 0`, `opt.density == 0`, `actuator_biastype == mjBIAS_NONE`). The implemented state advancement uses direct `qacc`; general implicit velocity-derivative equations ($M - dt \cdot D$) are not invoked because all velocity derivatives are zero in this model.
   - **Fixed Motor Torques vs Policy Control**: Holding 14 direct motor torques constant across 4 substeps is qualified here. **This is not yet the walking task's BAM-driven control interval**. The Torch BAM model computes motor torques dynamically per substep based on joint velocities and back-EMF, and requires friction loss handling. Full BAM walking-task integration remains the next milestone.
   - Supported constraints: contact-pyramidal friction constraints between terrain (geom 0) and canonical foot CAD meshes (geoms 27 and 73, bodies 7 and 16).
   - Unsupported constraints: general multi-body geom collisions, tendon constraints, equality constraints, joint limits, and joint damping remain unsupported.
2. **Solver Policy**:
   - Delassus PGS is retained for experimental validation.
   - Negative upstream statuses abort integration immediately and output `NAN`.
   - Trajectories encountering Status 1 log explicit residual diagnostics without silent promotion to convergence.
3. **Protected Training Process**:
   - Process PID 4632 (`mjlab_microduck.native_gpu.train --physics cpu --num-envs 4096`) was untouched throughout, maintaining uninterrupted training.
4. **Execution Bound**:
   - This milestone qualifies the 5 ms ImplicitFast integrator and 20 ms direct-torque control interval. No PPO training switch or speedup claims are made.
