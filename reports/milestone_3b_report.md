# Milestone 3B Qualification Report: Autonomous CAD Contact Manifold Assembly and Integration

**Date**: 2026-09-22  
**Workspace**: `/Users/zixiao/workspace/microduck/unified-metal`  
**Status**: QUALIFIED (Milestone 3B Focused Correction Pass v2 Fully Satisfied)  
**Deliverable**: Autonomous CAD contact manifold generation and constraint parameter assembly ($J, a_{\text{ref}}, R$) on Metal without host staging or oracle constraint inputs, coupled to the factor-and-solve Delassus solver for static forward-dynamics qualification against pinned CPU MuJoCo 3.10.0.

---

## 1. Scope, Safety & Review Directives

This milestone completes the autonomous contact and assembly pipeline in accordance with the review directives from `/Users/zixiao/workspace/microduck/private-notes/native-mac/unified-metal-3b-implementation-review.md` and `/Users/zixiao/workspace/microduck/private-notes/native-mac/unified-metal-3b-correction-review-v2.md`:

1. **Production Process Safety**:
   The background production training process (`PID 4632`: `mjlab_microduck.native_gpu.train --physics cpu --num-envs 4096`) was **not stopped, interrupted, or modified**, running continuously and undisturbed at ~87–95% CPU throughout all testing and development. All qualification workloads executed sequentially with small batch sizes ($B \le 4$, default $B=1$).
2. **Compact Contact Friction Stride & Allocation Safety**:
   - Fixed friction buffer preparation in `assemble_contact_constraints` to guarantee exact matching between the friction buffer layout and the incoming `stride_ncon` dimension passed to `kernel_assemble_contact_constraints`.
   - When consuming compact contact outputs (`stride_ncon < nconmax`), `f_buf` is prepared with exact contiguous shape `(B, stride_ncon, 2)`. This eliminates the stride mismatch where world 1 read into world 0's slot 4 default friction.
   - For scalar or default friction with arbitrary $N$, buffer preparation allocates matching dimension rather than overflowing persistent allocations.
3. **Active Contact Count Range Validation**:
   - In `kernel_assemble_contact_constraints`, added strict per-world bounds checks: `0 <= ncon[b] <= stride_nconmax`.
   - Counts $< 0$ or $> \text{stride\_nconmax}$ immediately flag `overflow_flag_out[b] = -4` (invalid contact count error), set `nefc = 0`, and write `NAN` to all output rows before any contact loop or memory reads.
4. **Zero Host-Synchronization in Friction Validation**:
   - Eliminated device `.item()` and `(friction > 0).all()` host synchronizations on MPS tensors during assembly.
   - Host checks perform structural shape and dtype validation. Positivity and finiteness are checked directly on GPU in `kernel_assemble_contact_constraints`: non-positive or non-finite coefficients set `overflow_flag_out[b] = -3`, zero `nefc`, and write `NAN`.
5. **Capacity Handling & GPU Memory Safety**:
   - Separated buffer allocation stride (`stride_nconmax`, `stride_capacity`) from active capacity limits (`active_nconmax`, `active_capacity`) in both `kernel_cad_contact_manifold_v2` and `kernel_assemble_contact_constraints`. Under $B \ge 2$, worlds index strictly by allocation stride, eliminating cross-world buffer corruption.
   - Enforced complete 4-row pyramidal contact groups: assembly caps rows to `4 * (min(stride_capacity, active_capacity) / 4)`.
   - Added pre-dispatch Python validation: rejects negative, oversized ($> 32$), and non-multiple-of-4 capacities before GPU dispatch.
   - Decoupled autonomous assembly allocation (`autonomous_capacity = 32`) from mutable oracle solver capacity.
6. **Pipeline Failure Propagation & Invalidation**:
   - Combined upstream stage validity on-device into `autonomous_upstream_status` without host synchronization:
     - Cholesky failure: status $< 0$ (-1, -2, -3)
     - Contact manifold overflow: status `-6`
     - Contact non-finite input: status `-8`
     - Constraint assembly overflow: status `-7`
     - Assembly error (non-finite state, invalid body ID, invalid friction, invalid count): status `-9`
   - When upstream failure occurs, `kernel_oracle_constrained_solve` propagates the failure code and fills all solve outputs (`qacc`, `qfrc_constraint`, `lambda_force`) with `NAN`.
   - Never accepts a truncated manifold as qualified physical output.
7. **Canonical Default Friction & General Friction Contracts**:
   - Derived default friction from model geoms: $\mu = \max(\mu_{\text{terrain}}, \mu_{\text{foot}}) = [1.0, 1.0]$.
   - Supports scalar, per-world `(B, 2)`, and per-contact `(B, N, 2)` friction tensors.
   - Qualified with nominal full-path standing with `friction=None`, independently specified randomized friction, and heterogeneous worlds ($B=2$).
8. **Consistent Iteration-Budget Comparison & KKT Diagnostics**:
   - Scenario inputs, model fields, external forces, and friction are prepared once and supplied identically to both 100-iteration and 200-iteration solver budgets.
   - For `randomized_friction_corpus`, $\mu = 0.6$ is passed consistently to both budgets.
   - Full KKT metric breakdown (primal infeasibility, dual infeasibility, complementarity, diagonally-scaled projected residual, convergence label) generated and verified.
9. **Non-Aliased Mixed-World Recovery**:
   - Outputs are cloned immediately after baseline execution to prevent in-place buffer aliasing.
   - Exercises real `forward_autonomous` orchestration under heterogeneous contact counts (8 rows vs 24 rows) with a shared active-capacity limit (`capacity=16`).
10. **Supported Domain & Operational Limits**:
   - **Supported Bodies / Geoms**: Two canonical foot meshes (`robot/left_foot_collision` on body 7, internal label 1; `robot/right_foot_collision` on body 16, internal label 2).
   - **Terrain**: Horizontal flat plane at $z=0$ with normal $[0, 0, 1]$.
   - **Contact Parameter Domain**: Margin must be $0.0$ (`params.margin == 0.0`); $condim=3$ (pyramidal friction cone with 4 facets per contact); positive impedance parameters (`timeconst > 0, dampratio > 0, dmin > 0, dmax > 0, width > 0`).
   - **Smooth Force Semantics**: Default smooth force is $-qfrc\_bias$ (zero applied/actuator/passive forces plus native bias). Complete smooth forces (including applied/actuator forces) can be supplied via `f_smooth`.
   - **Randomized-Model Regularization Semantics**: Dynamics accepts per-world mass/CoM/armature. Assembly retains nominal `body_invweight0` matching MuJoCo default constraint regularization unless constants are reloaded.
11. **Nominal Standing Height Clarification**:
   - The realistic nominal standing fixture uses base height **$z = 0.1151823622$ m**, producing measured 2.0 mm sole penetration. Contact parity is evaluated as measured numerical agreement against MuJoCo 3.10.0.

---

## 2. Source Code Manifest & Verification Hashes

| File | SHA256 Hash | Purpose |
|---|---|---|
| `shaders/physics_slice.metal` | `5c8be0d0351cb87bae576a1eb2d9aaab7e2d57160cfe4fc011bfa3e032ce722b` | Native Metal kernels (stride-aware contact v2, count-guarded assembly, device friction validation, Delassus PGS) |
| `src/representative_physics_slice.py` | `da63664cedbea7f21e09dee50ffb213bd8f82c10d5528268e64b6e68ffd15e09` | Driver with exact compact friction stride, decoupled capacity, zero-sync validation, failure propagation |
| `src/metal_kernel_manager.py` | `34f88232c3671ec5159d4e5d304e7548f9458d070c5e79b2ea3d8fcc17416961` | Metal kernel compilation and dispatch manager |
| `src/canonical_model_loader.py` | `38bb3a4f68cf4f22ff3d7d5638f4651a5ec4122cd39d1a167b2164809a404be4` | Model parameter loading, mesh hull graph extraction (`m.mesh_graph`), bounding radius |
| `src/oracle_generator.py` | `ea2b58737e6a00bdde985d5633860353136c4fa71b9ee80c00dfeaf864acb3d9` | Corpus generation for all 25 scenarios with full contact & constraint metadata |
| `src/verify_oracle_corpus.py` | `f9c7bd251b8d1545eef595f8b34374361e9999e88dafbb9092b07235fbd03e4d` | Verification script for corpus integrity |
| `scripts/eval_autonomous_table.py` | `7a74f9b9a10ad054f4fbb5ee2564f09097fbf025c62cdc3cc95a615b73ea1b75` | Evaluation script for 25-scenario budget comparison and complete candidate-QP KKT metrics |
| `tests/test_cad_contacts_and_assembly.py` | `b1b90dc8262cdf0021c434bcb5d7a779c3f96eb652ab6091e2e3488b9f0fd41f` | 161 automated tests covering equations, compact friction, count guards, non-aliased recovery, KKT |
| `tests/test_metal_constraint_solver.py` | `101a9c3a648cccbc7fa8b71bbeae844405d0ce8b925a562cbae57f04272a9c06` | Milestone 3A solver verification suite (75 tests) |
| `tests/test_native_dynamics_and_solves.py` | `b4ff8dc949750ea1e74f051dddd26c8ebaad62b2e81bf152491a4e977cfb5e51` | Milestones 0-2 CRBA/RNE and Cholesky solve suite (38 tests) |
| `tests/test_representative_physics.py` | `0e21f0e44264ccf4ced28ce925125bf650128172f0f59874db97a3039ac6f25a` | Milestone 0 representative physics suite (7 tests) |

---

## 3. Contact Manifold Parity vs MuJoCo 3.10.0

Comparing Metal `kernel_cad_contact_manifold_v2` against CPU MuJoCo 3.10.0 oracle contacts:

| Scenario | Metal/Ref $n_{\text{con}}$ | Max Pos Error (m) | Max Dist Error (m) | Manifold Selection Match |
|---|:---:|:---:|:---:|:---:|
| `airborne` | 0 / 0 | 0.00 | 0.00 | Exact (Zero contacts) |
| `asymmetric_pose` | 3 / 3 | $1.96 \times 10^{-8}$ | $1.20 \times 10^{-8}$ | Exact (100% vertex match) |
| `boundary_onset_exact` | 1 / 1 | $1.36 \times 10^{-8}$ | $2.33 \times 10^{-10}$ | Exact (Single touch vertex) |
| `boundary_separated_2mm` | 0 / 0 | 0.00 | 0.00 | Exact (Zero contacts) |
| `combined_rotation_motion` | 0 / 0 | 0.00 | 0.00 | Exact (Zero contacts) |
| `contact_onset` | 6 / 6 | $1.36 \times 10^{-8}$ | $1.47 \times 10^{-8}$ | Exact (100% vertex match) |
| `crouched_pose` | 6 / 6 | $1.36 \times 10^{-8}$ | $1.47 \times 10^{-8}$ | Exact (100% vertex match) |
| `heel_only_contact_realistic` | 2 / 2 | $1.96 \times 10^{-8}$ | $1.16 \times 10^{-8}$ | Exact (Heel vertices only) |
| `high_condition_mass_matrix` | 6 / 6 | $1.36 \times 10^{-8}$ | $1.47 \times 10^{-8}$ | Exact (100% vertex match) |
| `impedance_transition_edge` | 4 / 4 | $1.36 \times 10^{-8}$ | $6.74 \times 10^{-9}$ | Exact (100% vertex match) |
| `impedance_transition_mid` | 4 / 4 | $1.36 \times 10^{-8}$ | $6.74 \times 10^{-9}$ | Exact (100% vertex match) |
| `impedance_transition_shallow` | 4 / 4 | $1.36 \times 10^{-8}$ | $6.74 \times 10^{-9}$ | Exact (100% vertex match) |
| `near_contact_separation` | 6 / 6 | $1.36 \times 10^{-8}$ | $1.53 \times 10^{-8}$ | Exact (100% vertex match) |
| `nominal_standing_realistic` | 4 / 4 | $1.36 \times 10^{-8}$ | $6.74 \times 10^{-9}$ | Exact (100% vertex match) |
| `nonzero_applied_force` | 6 / 6 | $1.36 \times 10^{-8}$ | $1.47 \times 10^{-8}$ | Exact (100% vertex match) |
| `nonzero_base_angvel` | 0 / 0 | 0.00 | 0.00 | Exact (Zero contacts) |
| `randomized_friction_corpus` | 4 / 4 | $1.36 \times 10^{-8}$ | $6.74 \times 10^{-9}$ | Exact (100% vertex match) |
| `randomized_model_standing` | 6 / 6 | $1.36 \times 10^{-8}$ | $1.47 \times 10^{-8}$ | Exact (100% vertex match) |
| `single_support_moving_vel` | 3 / 3 | $8.12 \times 10^{-9}$ | $4.85 \times 10^{-9}$ | Exact (Left foot only) |
| `single_support_zero_vel` | 3 / 3 | $8.12 \times 10^{-9}$ | $4.85 \times 10^{-9}$ | Exact (Left foot only) |
| `sliding_lateral_velocity` | 4 / 4 | $1.36 \times 10^{-8}$ | $6.74 \times 10^{-9}$ | Exact (100% vertex match) |
| `standing_moving_vel` | 6 / 6 | $1.36 \times 10^{-8}$ | $1.47 \times 10^{-8}$ | Exact (100% vertex match) |
| `standing_zero_vel` | 6 / 6 | $1.36 \times 10^{-8}$ | $1.47 \times 10^{-8}$ | Exact (100% vertex match) |
| `tilted_landing` | 6 / 6 | $1.96 \times 10^{-8}$ | $1.20 \times 10^{-8}$ | Exact (100% vertex match) |
| `toe_only_contact_realistic` | 2 / 2 | $1.05 \times 10^{-8}$ | $1.61 \times 10^{-8}$ | Exact (Toe vertices only) |

---

## 4. Constraint Assembly Parity vs MuJoCo 3.10.0

Comparing Metal `kernel_assemble_contact_constraints` outputs ($J, a_{\text{ref}}, R$) against CPU MuJoCo 3.10.0 reference rows:

| Scenario | Metal/Ref $n_{\text{efc}}$ | Max $\|J - J_{\text{ref}}\|_\infty$ | Max $\|a_{\text{ref}} - a_{\text{ref, ref}}\|_\infty$ | Max $\|R - R_{\text{ref}}\|_\infty$ |
|---|:---:|:---:|:---:|:---:|
| `airborne` | 0 / 0 | 0.00 | 0.00 | 0.00 |
| `asymmetric_pose` | 12 / 12 | $2.54 \times 10^{-8}$ | $4.29 \times 10^{-5}$ | $4.16 \times 10^{-7}$ |
| `boundary_onset_exact` | 4 / 4 | $2.83 \times 10^{-8}$ | $1.86 \times 10^{-6}$ | $4.16 \times 10^{-7}$ |
| `boundary_separated_2mm` | 0 / 0 | 0.00 | 0.00 | 0.00 |
| `combined_rotation_motion` | 0 / 0 | 0.00 | 0.00 | 0.00 |
| `contact_onset` | 24 / 24 | $2.45 \times 10^{-8}$ | $4.14 \times 10^{-5}$ | $4.16 \times 10^{-7}$ |
| `crouched_pose` | 24 / 24 | $2.45 \times 10^{-8}$ | $4.14 \times 10^{-5}$ | $4.16 \times 10^{-7}$ |
| `heel_only_contact_realistic` | 8 / 8 | $2.75 \times 10^{-8}$ | $3.34 \times 10^{-5}$ | $4.16 \times 10^{-7}$ |
| `high_condition_mass_matrix` | 24 / 24 | $2.45 \times 10^{-8}$ | $4.14 \times 10^{-5}$ | $4.16 \times 10^{-7}$ |
| `impedance_transition_edge` | 16 / 16 | $2.83 \times 10^{-8}$ | $1.72 \times 10^{-5}$ | $4.16 \times 10^{-7}$ |
| `impedance_transition_mid` | 16 / 16 | $2.73 \times 10^{-8}$ | $2.29 \times 10^{-5}$ | $2.84 \times 10^{-5}$ |
| `impedance_transition_shallow` | 16 / 16 | $2.67 \times 10^{-8}$ | $2.27 \times 10^{-5}$ | $6.03 \times 10^{-6}$ |
| `near_contact_separation` | 24 / 24 | $2.58 \times 10^{-8}$ | $3.18 \times 10^{-5}$ | $4.16 \times 10^{-7}$ |
| `nominal_standing_realistic` | 16 / 16 | $2.83 \times 10^{-8}$ | $1.72 \times 10^{-5}$ | $4.16 \times 10^{-7}$ |
| `nonzero_applied_force` | 24 / 24 | $2.45 \times 10^{-8}$ | $4.14 \times 10^{-5}$ | $4.16 \times 10^{-7}$ |
| `nonzero_base_angvel` | 0 / 0 | 0.00 | 0.00 | 0.00 |
| `randomized_friction_corpus` | 16 / 16 | $2.47 \times 10^{-8}$ | $1.72 \times 10^{-5}$ | $1.28 \times 10^{-7}$ |
| `randomized_model_standing` | 24 / 24 | $2.45 \times 10^{-8}$ | $4.05 \times 10^{-5}$ | $4.16 \times 10^{-7}$ |
| `single_support_moving_vel` | 12 / 12 | $1.57 \times 10^{-8}$ | $5.32 \times 10^{-5}$ | $3.22 \times 10^{-7}$ |
| `single_support_zero_vel` | 12 / 12 | $1.57 \times 10^{-8}$ | $4.14 \times 10^{-5}$ | $3.22 \times 10^{-7}$ |
| `sliding_lateral_velocity` | 16 / 16 | $2.83 \times 10^{-8}$ | $1.79 \times 10^{-5}$ | $4.16 \times 10^{-7}$ |
| `standing_moving_vel` | 24 / 24 | $2.45 \times 10^{-8}$ | $4.05 \times 10^{-5}$ | $4.16 \times 10^{-7}$ |
| `standing_zero_vel` | 24 / 24 | $2.45 \times 10^{-8}$ | $4.14 \times 10^{-5}$ | $4.16 \times 10^{-7}$ |
| `tilted_landing` | 24 / 24 | $2.54 \times 10^{-8}$ | $7.69 \times 10^{-5}$ | $4.16 \times 10^{-7}$ |
| `toe_only_contact_realistic` | 8 / 8 | $2.37 \times 10^{-8}$ | $4.29 \times 10^{-5}$ | $4.16 \times 10^{-7}$ |

---

## 5. End-to-End Autonomous Forward Dynamics & QP KKT Diagnostics

Full forward pipeline: $\text{FK} \to \text{CRBA} / \text{RNE} \to \text{Cholesky} \to \text{CAD Contacts} \to \text{Assembly} \to \text{Delassus PGS Solve}$.  
Reproducible via `/Users/zixiao/workspace/microduck/microduck_rl/.venv/bin/python scripts/eval_autonomous_table.py`.

### Autonomous Forward Dynamics & Budget Comparison (100 vs 200 iters)

| Scenario | $n_{\text{efc}}$ | Stat (100) | Iter (100) | Label (100) | Stat (200) | Iter (200) | Label (200) | Dual Infeas (200) | Proj Res (200) | Max Acc Err ($\text{m/s}^2, \text{rad/s}^2$) | Max Frc Err ($\text{N}, \text{Nm}$) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `airborne` | 0 | 0 | 0 | converged | 0 | 0 | converged | 0.00e+00 | 0.00e+00 | $3.36 \times 10^{-6}$ | 0.00 |
| `asymmetric_pose` | 12 | 1 | 100 | exhausted | 0 | 121 | converged | $9.16 \times 10^{-5}$ | $1.14 \times 10^{-5}$ | $4.47 \times 10^{-4}$ | $1.95 \times 10^{-5}$ |
| `boundary_onset_exact` | 4 | 0 | 27 | converged | 0 | 27 | converged | $5.53 \times 10^{-5}$ | $6.14 \times 10^{-6}$ | $2.01 \times 10^{-4}$ | $6.18 \times 10^{-6}$ |
| `boundary_separated_2mm` | 0 | 0 | 0 | converged | 0 | 0 | converged | 0.00e+00 | 0.00e+00 | $3.36 \times 10^{-6}$ | 0.00 |
| `combined_rotation_motion` | 0 | 0 | 0 | converged | 0 | 0 | converged | 0.00e+00 | 0.00e+00 | $5.20 \times 10^{-6}$ | 0.00 |
| `contact_onset` | 24 | 0 | 95 | converged | 0 | 95 | converged | $9.16 \times 10^{-5}$ | $1.19 \times 10^{-5}$ | $4.61 \times 10^{-4}$ | $1.54 \times 10^{-5}$ |
| `crouched_pose` | 24 | 1 | 100 | exhausted | 1 | 200 | exhausted | $1.22 \times 10^{-4}$ | $8.17 \times 10^{-6}$ | $4.91 \times 10^{-4}$ | $5.89 \times 10^{-6}$ |
| `heel_only_contact_realistic` | 8 | 0 | 48 | converged | 0 | 48 | converged | $5.82 \times 10^{-5}$ | $6.91 \times 10^{-6}$ | $1.34 \times 10^{-4}$ | $5.37 \times 10^{-6}$ |
| `high_condition_mass_matrix` | 24 | 1 | 100 | exhausted | 1 | 200 | exhausted | $2.58 \times 10^{-3}$ | $1.34 \times 10^{-4}$ | $3.29 \times 10^{-3}$ | $6.23 \times 10^{-5}$ |
| `impedance_transition_edge` | 16 | 0 | 71 | converged | 0 | 71 | converged | $9.16 \times 10^{-5}$ | $8.67 \times 10^{-6}$ | $4.25 \times 10^{-4}$ | $1.36 \times 10^{-5}$ |
| `impedance_transition_mid` | 16 | 0 | 50 | converged | 0 | 50 | converged | $7.72 \times 10^{-5}$ | $6.23 \times 10^{-6}$ | $2.92 \times 10^{-4}$ | $2.05 \times 10^{-5}$ |
| `impedance_transition_shallow` | 16 | 0 | 40 | converged | 0 | 40 | converged | $6.77 \times 10^{-5}$ | $4.68 \times 10^{-6}$ | $2.21 \times 10^{-4}$ | $1.16 \times 10^{-5}$ |
| `near_contact_separation` | 24 | 0 | 97 | converged | 0 | 97 | converged | $1.22 \times 10^{-4}$ | $1.10 \times 10^{-5}$ | $4.29 \times 10^{-4}$ | $9.52 \times 10^{-6}$ |
| `nominal_standing_realistic` | 16 | 0 | 72 | converged | 0 | 72 | converged | $8.68 \times 10^{-5}$ | $8.17 \times 10^{-6}$ | $3.84 \times 10^{-4}$ | $9.97 \times 10^{-6}$ |
| `nonzero_applied_force` | 24 | 1 | 100 | exhausted | 0 | 114 | converged | $1.22 \times 10^{-4}$ | $1.53 \times 10^{-5}$ | $3.14 \times 10^{-4}$ | $3.11 \times 10^{-5}$ |
| `nonzero_base_angvel` | 0 | 0 | 0 | converged | 0 | 0 | converged | 0.00e+00 | 0.00e+00 | $3.29 \times 10^{-6}$ | 0.00 |
| `randomized_friction_corpus` | 16 | 1 | 100 | exhausted | 0 | 107 | converged | $6.96 \times 10^{-5}$ | $6.84 \times 10^{-6}$ | $5.11 \times 10^{-4}$ | $4.03 \times 10^{-6}$ |
| `randomized_model_standing` | 24 | 0 | 96 | converged | 0 | 96 | converged | $6.10 \times 10^{-5}$ | $1.24 \times 10^{-5}$ | $3.90 \times 10^{-4}$ | $1.20 \times 10^{-5}$ |
| `single_support_moving_vel` | 12 | 0 | 88 | converged | 0 | 88 | converged | $9.16 \times 10^{-5}$ | $1.17 \times 10^{-5}$ | $3.86 \times 10^{-4}$ | $2.40 \times 10^{-5}$ |
| `single_support_zero_vel` | 12 | 0 | 87 | converged | 0 | 87 | converged | $6.10 \times 10^{-5}$ | $8.58 \times 10^{-6}$ | $2.93 \times 10^{-4}$ | $1.28 \times 10^{-5}$ |
| `sliding_lateral_velocity` | 16 | 0 | 79 | converged | 0 | 79 | converged | $9.06 \times 10^{-5}$ | $4.74 \times 10^{-6}$ | $4.54 \times 10^{-4}$ | $6.40 \times 10^{-6}$ |
| `standing_moving_vel` | 24 | 0 | 94 | converged | 0 | 94 | converged | $1.22 \times 10^{-4}$ | $1.05 \times 10^{-5}$ | $2.34 \times 10^{-4}$ | $3.14 \times 10^{-5}$ |
| `standing_zero_vel` | 24 | 0 | 95 | converged | 0 | 95 | converged | $9.16 \times 10^{-5}$ | $1.05 \times 10^{-5}$ | $2.46 \times 10^{-4}$ | $1.62 \times 10^{-5}$ |
| `tilted_landing` | 24 | 0 | 97 | converged | 0 | 97 | converged | $1.22 \times 10^{-4}$ | $1.53 \times 10^{-5}$ | $6.14 \times 10^{-4}$ | $5.48 \times 10^{-5}$ |
| `toe_only_contact_realistic` | 8 | 0 | 53 | converged | 0 | 53 | converged | $6.39 \times 10^{-5}$ | $7.15 \times 10^{-6}$ | $2.64 \times 10^{-4}$ | $1.61 \times 10^{-5}$ |

### Complete Candidate-QP KKT Residual Breakdown (Budgets 100 and 200)

| Scenario | Budget | Stat | Iter | Label | Primal Infeas | Dual Infeas | Complementarity | Proj Residual |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `airborne` | 100 | 0 | 0 | converged | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `airborne` | 200 | 0 | 0 | converged | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `asymmetric_pose` | 100 | 1 | 100 | exhausted | 0.00e+00 | 3.66e-04 | 4.73e-03 | 7.80e-05 |
| `asymmetric_pose` | 200 | 0 | 121 | converged | 0.00e+00 | 9.16e-05 | 8.87e-04 | 1.14e-05 |
| `boundary_onset_exact` | 100 | 0 | 27 | converged | 0.00e+00 | 5.53e-05 | 6.31e-05 | 6.14e-06 |
| `boundary_onset_exact` | 200 | 0 | 27 | converged | 0.00e+00 | 5.53e-05 | 6.31e-05 | 6.14e-06 |
| `boundary_separated_2mm` | 100 | 0 | 0 | converged | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `boundary_separated_2mm` | 200 | 0 | 0 | converged | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `combined_rotation_motion` | 100 | 0 | 0 | converged | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `combined_rotation_motion` | 200 | 0 | 0 | converged | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `contact_onset` | 100 | 0 | 95 | converged | 0.00e+00 | 9.16e-05 | 9.51e-04 | 1.19e-05 |
| `contact_onset` | 200 | 0 | 95 | converged | 0.00e+00 | 9.16e-05 | 9.51e-04 | 1.19e-05 |
| `crouched_pose` | 100 | 1 | 100 | exhausted | 0.00e+00 | 6.01e-03 | 1.39e-01 | 2.44e-03 |
| `crouched_pose` | 200 | 1 | 200 | exhausted | 0.00e+00 | 1.22e-04 | 3.57e-04 | 8.17e-06 |
| `heel_only_contact_realistic` | 100 | 0 | 48 | converged | 0.00e+00 | 5.82e-05 | 7.48e-05 | 6.91e-06 |
| `heel_only_contact_realistic` | 200 | 0 | 48 | converged | 0.00e+00 | 5.82e-05 | 7.48e-05 | 6.91e-06 |
| `high_condition_mass_matrix` | 100 | 1 | 100 | exhausted | 0.00e+00 | 1.01e-01 | 2.32e-01 | 5.08e-03 |
| `high_condition_mass_matrix` | 200 | 1 | 200 | exhausted | 0.00e+00 | 2.58e-03 | 6.60e-03 | 1.34e-04 |
| `impedance_transition_edge` | 100 | 0 | 71 | converged | 0.00e+00 | 9.16e-05 | 6.52e-05 | 8.67e-06 |
| `impedance_transition_edge` | 200 | 0 | 71 | converged | 0.00e+00 | 9.16e-05 | 6.52e-05 | 8.67e-06 |
| `impedance_transition_mid` | 100 | 0 | 50 | converged | 0.00e+00 | 7.72e-05 | 4.53e-05 | 6.23e-06 |
| `impedance_transition_mid` | 200 | 0 | 50 | converged | 0.00e+00 | 7.72e-05 | 4.53e-05 | 6.23e-06 |
| `impedance_transition_shallow` | 100 | 0 | 40 | converged | 0.00e+00 | 6.77e-05 | 3.41e-05 | 4.68e-06 |
| `impedance_transition_shallow` | 200 | 0 | 40 | converged | 0.00e+00 | 6.77e-05 | 3.41e-05 | 4.68e-06 |
| `near_contact_separation` | 100 | 0 | 97 | converged | 0.00e+00 | 1.22e-04 | 6.96e-04 | 1.10e-05 |
| `near_contact_separation` | 200 | 0 | 97 | converged | 0.00e+00 | 1.22e-04 | 6.96e-04 | 1.10e-05 |
| `nominal_standing_realistic` | 100 | 0 | 72 | converged | 0.00e+00 | 8.68e-05 | 7.51e-05 | 8.17e-06 |
| `nominal_standing_realistic` | 200 | 0 | 72 | converged | 0.00e+00 | 8.68e-05 | 7.51e-05 | 8.17e-06 |
| `nonzero_applied_force` | 100 | 1 | 100 | exhausted | 0.00e+00 | 5.80e-04 | 3.90e-03 | 4.91e-05 |
| `nonzero_applied_force` | 200 | 0 | 114 | converged | 0.00e+00 | 1.22e-04 | 1.20e-03 | 1.53e-05 |
| `nonzero_base_angvel` | 100 | 0 | 0 | converged | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `nonzero_base_angvel` | 200 | 0 | 0 | converged | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `randomized_friction_corpus` | 100 | 1 | 100 | exhausted | 0.00e+00 | 1.29e-04 | 3.60e-05 | 1.27e-05 |
| `randomized_friction_corpus` | 200 | 0 | 107 | converged | 0.00e+00 | 6.96e-05 | 2.04e-05 | 6.84e-06 |
| `randomized_model_standing` | 100 | 0 | 96 | converged | 0.00e+00 | 6.10e-05 | 5.81e-04 | 1.24e-05 |
| `randomized_model_standing` | 200 | 0 | 96 | converged | 0.00e+00 | 6.10e-05 | 5.81e-04 | 1.24e-05 |
| `single_support_moving_vel` | 100 | 0 | 88 | converged | 0.00e+00 | 9.16e-05 | 1.54e-03 | 1.17e-05 |
| `single_support_moving_vel` | 200 | 0 | 88 | converged | 0.00e+00 | 9.16e-05 | 1.54e-03 | 1.17e-05 |
| `single_support_zero_vel` | 100 | 0 | 87 | converged | 0.00e+00 | 6.10e-05 | 7.99e-04 | 8.58e-06 |
| `single_support_zero_vel` | 200 | 0 | 87 | converged | 0.00e+00 | 6.10e-05 | 7.99e-04 | 8.58e-06 |
| `sliding_lateral_velocity` | 100 | 0 | 79 | converged | 0.00e+00 | 9.06e-05 | 4.13e-05 | 4.74e-06 |
| `sliding_lateral_velocity` | 200 | 0 | 79 | converged | 0.00e+00 | 9.06e-05 | 4.13e-05 | 4.74e-06 |
| `standing_moving_vel` | 100 | 0 | 94 | converged | 0.00e+00 | 1.22e-04 | 7.43e-04 | 1.05e-05 |
| `standing_moving_vel` | 200 | 0 | 94 | converged | 0.00e+00 | 1.22e-04 | 7.43e-04 | 1.05e-05 |
| `standing_zero_vel` | 100 | 0 | 95 | converged | 0.00e+00 | 9.16e-05 | 9.24e-04 | 1.05e-05 |
| `standing_zero_vel` | 200 | 0 | 95 | converged | 0.00e+00 | 9.16e-05 | 9.24e-04 | 1.05e-05 |
| `tilted_landing` | 100 | 0 | 97 | converged | 0.00e+00 | 1.22e-04 | 2.05e-03 | 1.53e-05 |
| `tilted_landing` | 200 | 0 | 97 | converged | 0.00e+00 | 1.22e-04 | 2.05e-03 | 1.53e-05 |
| `toe_only_contact_realistic` | 100 | 0 | 53 | converged | 0.00e+00 | 6.39e-05 | 8.04e-05 | 7.15e-06 |
| `toe_only_contact_realistic` | 200 | 0 | 53 | converged | 0.00e+00 | 6.39e-05 | 8.04e-05 | 7.15e-06 |

**Diagnostic Notes**:
1. **Convergence vs Iteration Budget**: The public API default remains `max_iters = 100`. At 100 iterations, 20 of 25 cases achieve status 0 (5 exhausted: `asymmetric_pose`, `crouched_pose`, `high_condition_mass_matrix`, `nonzero_applied_force`, `randomized_friction_corpus`). Expanding to the qualification budget (`max_iters = 200`) brings 23 of 25 cases to status 0 (with `asymmetric_pose`, `nonzero_applied_force`, and `randomized_friction_corpus` reaching status 0).
2. **Status 1 Bounded Non-Convergence**: In the remaining 2 cases (`crouched_pose` and `high_condition_mass_matrix`), the solver exhausts the 200 iteration budget. These cases remain bounded diagnostics: their accelerations and forces remain completely finite, with physical acceleration error $\le 0.00329 \text{ m/s}^2$ (comfortably within the $0.05 / 0.15$ gates) and force error $\le 6.23 \times 10^{-5} \text{ N}$ (within $0.05 \text{ N}$).

---

## 6. Physical Qualification Gates Summary

| Quantity | Target Qualification Gate | Measured Max Across All 25 Scenarios | Margin of Satisfaction | Pass / Fail |
|---|:---:|:---:|:---:|:---:|
| **Contact Manifold Count** | Exact $n_{\text{con}} = n_{\text{ref}}$ | Exact match on all 25 scenarios | 100% agreement | **PASS** |
| **Contact Positions** | $\le 10^{-4}$ m | $1.96 \times 10^{-8}$ m | $5000\times$ below gate | **PASS** |
| **Contact Distances** | $\le 10^{-4}$ m | $1.61 \times 10^{-8}$ m | $6000\times$ below gate | **PASS** |
| **Jacobian Matrix $J$** | $\le 10^{-5}$ | $2.83 \times 10^{-8}$ | $350\times$ below gate | **PASS** |
| **Reference Acc $a_{\text{ref}}$** | $\le 10^{-3} \text{ m/s}^2$ | $7.69 \times 10^{-5} \text{ m/s}^2$ | $13\times$ below gate | **PASS** |
| **Regularization $R$** | $\le 10^{-4}$ | $2.84 \times 10^{-5}$ | $3.5\times$ below gate | **PASS** |
| **Linear Force Error** | $\le 0.05$ N | $6.23 \times 10^{-5}$ N | $800\times$ below gate | **PASS** |
| **Torque Error** | $\le 0.05$ Nm | $3.04 \times 10^{-6}$ Nm | $16,000\times$ below gate | **PASS** |
| **Joint Torque Error** | $\le 0.05$ Nm | $1.46 \times 10^{-6}$ Nm | $34,000\times$ below gate | **PASS** |
| **Linear Acceleration** | $\le 0.05 \text{ m/s}^2$ | $7.25 \times 10^{-5} \text{ m/s}^2$ | $690\times$ below gate | **PASS** |
| **Angular Acceleration** | $\le 0.05 \text{ rad/s}^2$ ($0.15$ exc.) | $0.00129 \text{ rad/s}^2$ | $38\times$ below gate | **PASS** |
| **Joint Acceleration** | $\le 0.05 \text{ rad/s}^2$ | $0.00329 \text{ rad/s}^2$ | $15\times$ below gate | **PASS** |
| **Primal Infeasibility** | $\le 10^{-6}$ | $0.00$ | Exact feasibility | **PASS** |
| **Dual Infeasibility (Status 0)** | $\le 2 \times 10^{-4}$ | $1.22 \times 10^{-4}$ | Within float32 precision | **PASS** |
| **Projected Gradient Step (Status 0)** | $\le 2 \times 10^{-5}$ | $1.53 \times 10^{-5}$ | Within convergence gate | **PASS** |
| **Buffer Overflow Safety** | Controlled invalidation & NaN | $s_{\text{contact}}=-6, s_{\text{assembly}}=-7$ | Output invalidated with NaN | **PASS** |
| **Neighbor Isolation ($B=2$)** | Failed world does not affect valid world | World 0 matches valid ref, World 1 is NaN | Exact isolation | **PASS** |
| **Active Count Range Safety** | Out-of-range counts caught on GPU | $s_{\text{count}}=-4$ | Output invalidated with NaN | **PASS** |
| **Friction Positivity Safety** | Non-positive/NaN caught on GPU | $s_{\text{friction}}=-3$ | Output invalidated with NaN | **PASS** |

---

## 7. Full Repository Test Summary

All 289 unit and integration tests across the repository pass without error in 5.26s:

```bash
tests/test_cad_contacts_and_assembly.py ................................ [ 11%]
........................................................................ [ 35%]
.........................................................                [ 55%]
tests/test_metal_constraint_solver.py .................................. [ 67%]
.........................................                                [ 81%]
tests/test_native_dynamics_and_solves.py ............................... [ 92%]
.......                                                                  [ 94%]
tests/test_representative_physics.py .......                             [ 97%]
tests/test_shared_buffer.py ......                                       [ 99%]
tests/test_task_inventory.py ..                                          [100%]

============================= 289 passed in 5.26s ==============================
```
