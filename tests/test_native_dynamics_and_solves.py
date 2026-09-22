"""Automated Test Suite for Native Metal Dynamics (CRBA/RNE) and Native Cholesky Solves.

Verifies:
1. Native forward kinematics position and orientation parity vs CPU oracle corpus (< 1e-6).
2. Native CRBA generalized mass matrix (M_eff) parity vs CPU oracle corpus (< 1e-6).
3. Native RNE Coriolis/centrifugal/gravity bias force parity vs CPU oracle corpus (< 1e-5).
4. Native Cholesky factorization accuracy ||L L^T - M|| (< 1e-6).
5. Native linear solves relative residual ||M X - B|| / (||M|| ||X|| + ||B||) (< 1e-5).
6. Direct linear solve solution parity ||X - X_cpu|| vs CPU MuJoCo oracle (< 1e-3).
7. Deterministic detection of non-positive pivots, NaNs, infinities, and non-finite RHS.
8. Output buffer invalidation (NaN filling) preventing stale data leakage across steps.
9. Downstream failure propagation in constrained solve.
10. Strict batch/shape/dtype/device guards and safe dynamic buffer resizing.
11. Heterogeneous batch domain randomization independence on GPU.
"""

from pathlib import Path
import sys
import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.canonical_model_loader import load_canonical_model
from src.representative_physics_slice import RepresentativePhysicsSlice

CORPUS_DIR = PROJECT_ROOT / "corpus"


@pytest.fixture(scope="module")
def canonical():
    return load_canonical_model()


@pytest.fixture(scope="module")
def engine(canonical):
    return RepresentativePhysicsSlice(batch_size=1, canonical=canonical)


ALL_SCENARIOS = [
    "standing_zero_vel",
    "standing_moving_vel",
    "single_support_zero_vel",
    "single_support_moving_vel",
    "tilted_landing",
    "airborne",
    "nonzero_base_angvel",
    "near_contact_separation",
    "contact_onset",
    "randomized_model_standing",
    "crouched_pose",
    "asymmetric_pose",
    "combined_rotation_motion",
    "high_condition_mass_matrix",
    "nonzero_applied_force",
]


@pytest.mark.parametrize("scenario", ALL_SCENARIOS)
def test_oracle_kinematics_and_dynamics_parity(engine, scenario):
    """Verifies that native GPU FK, M_eff, and qfrc_bias match the independent CPU oracle."""
    npz = np.load(CORPUS_DIR / f"{scenario}.npz")
    qpos = torch.from_numpy(npz["qpos"].astype(np.float32)).unsqueeze(0).to(engine.device)
    qvel = torch.from_numpy(npz["qvel"].astype(np.float32)).unsqueeze(0).to(engine.device)

    p_mass = torch.from_numpy(npz["per_world_mass"].astype(np.float32)).unsqueeze(0).to(engine.device) if "per_world_mass" in npz else None
    p_ipos = torch.from_numpy(npz["per_world_ipos"].astype(np.float32)).unsqueeze(0).to(engine.device) if "per_world_ipos" in npz else None
    p_arm = torch.from_numpy(npz["per_world_armature"].astype(np.float32)).unsqueeze(0).to(engine.device) if "per_world_armature" in npz else None

    engine.compute_native_dynamics(qpos, qvel, per_world_mass=p_mass, per_world_ipos=p_ipos, per_world_armature=p_arm)
    torch.mps.synchronize()

    xpos_gpu = engine.body_xpos[0].cpu().numpy()
    xmat_gpu = engine.body_xmat[0].cpu().numpy()
    M_gpu = engine.M_eff[0].cpu().numpy()
    bias_gpu = engine.qfrc_bias[0].cpu().numpy()

    err_xpos = float(np.max(np.abs(xpos_gpu - npz["xpos"])))
    err_xmat = float(np.max(np.abs(xmat_gpu - npz["xmat"])))
    err_M = float(np.max(np.abs(M_gpu - npz["M"])))
    err_bias = float(np.max(np.abs(bias_gpu - npz["qfrc_bias"])))

    assert err_xpos < 1e-6, f"[{scenario}] Body position FK error {err_xpos:.2e} >= 1e-6 m"
    assert err_xmat < 1e-6, f"[{scenario}] Body orientation FK error {err_xmat:.2e} >= 1e-6"
    assert err_M < 1e-6, f"[{scenario}] Mass matrix CRBA error {err_M:.2e} >= 1e-6"
    assert err_bias < 1e-5, f"[{scenario}] Bias force RNE error {err_bias:.2e} >= 1e-5"


@pytest.mark.parametrize("scenario", ALL_SCENARIOS)
def test_native_cholesky_factorization_and_solve(engine, scenario):
    """Verifies that native GPU Cholesky factorization and linear solves achieve < 1e-5 relative residual and < 1e-3 solution error."""
    npz = np.load(CORPUS_DIR / f"{scenario}.npz")
    qpos = torch.from_numpy(npz["qpos"].astype(np.float32)).unsqueeze(0).to(engine.device)
    qvel = torch.from_numpy(npz["qvel"].astype(np.float32)).unsqueeze(0).to(engine.device)

    p_mass = torch.from_numpy(npz["per_world_mass"].astype(np.float32)).unsqueeze(0).to(engine.device) if "per_world_mass" in npz else None
    p_ipos = torch.from_numpy(npz["per_world_ipos"].astype(np.float32)).unsqueeze(0).to(engine.device) if "per_world_ipos" in npz else None
    p_arm = torch.from_numpy(npz["per_world_armature"].astype(np.float32)).unsqueeze(0).to(engine.device) if "per_world_armature" in npz else None

    engine.compute_native_dynamics(qpos, qvel, per_world_mass=p_mass, per_world_ipos=p_ipos, per_world_armature=p_arm)
    X_bias, status = engine.compute_native_cholesky_solve(engine.M_eff, engine.qfrc_bias)
    torch.mps.synchronize()

    assert status.item() == 0, f"[{scenario}] Cholesky factorization failed with status {status.item()}"

    L_gpu = engine.L_factor[0].cpu().numpy()
    M_gpu = engine.M_eff[0].cpu().numpy()
    X_gpu = X_bias[0].cpu().numpy()
    bias_gpu = engine.qfrc_bias[0].cpu().numpy()

    assert np.all(np.isfinite(L_gpu)), f"[{scenario}] L_factor contains non-finite values"
    assert np.all(np.isfinite(X_gpu)), f"[{scenario}] Solution X contains non-finite values"

    # Factorization accuracy
    res_L = float(np.max(np.abs(L_gpu @ L_gpu.T - M_gpu)))
    assert res_L < 1e-6, f"[{scenario}] Cholesky reconstruction error {res_L:.2e} >= 1e-6"

    # Linear solve relative residual: ||M X - B|| / (||M|| ||X|| + ||B||)
    b_norm = np.linalg.norm(bias_gpu)
    denom = np.linalg.norm(M_gpu) * np.linalg.norm(X_gpu) + b_norm
    rel_residual = float(np.linalg.norm(M_gpu @ X_gpu - bias_gpu[:, None]) / denom) if denom > 0 else 0.0
    assert rel_residual < 1e-5, f"[{scenario}] Solve relative residual {rel_residual:.2e} >= 1e-5"

    # Direct solution error vs independent CPU MuJoCo solve
    err_sol = float(np.max(np.abs(X_gpu.squeeze() - npz["x_bias"])))
    assert err_sol < 1e-3, f"[{scenario}] Direct solution error {err_sol:.2e} >= 1e-3"


def test_native_cholesky_multi_rhs_columns(engine):
    """Verifies that native GPU Cholesky accurately solves M X = B for K=4 RHS columns simultaneously."""
    npz = np.load(CORPUS_DIR / "standing_zero_vel.npz")
    qpos = torch.from_numpy(npz["qpos"].astype(np.float32)).unsqueeze(0).to(engine.device)
    qvel = torch.from_numpy(npz["qvel"].astype(np.float32)).unsqueeze(0).to(engine.device)

    engine.compute_native_dynamics(qpos, qvel)

    B_multi = npz["B_multi"].astype(np.float32)
    B_t = torch.from_numpy(B_multi).unsqueeze(0).to(engine.device)

    X_gpu, status = engine.compute_native_cholesky_solve(engine.M_eff, B_t)
    torch.mps.synchronize()

    assert status.item() == 0
    X_np = X_gpu[0].cpu().numpy()
    M_np = engine.M_eff[0].cpu().numpy()

    # Relative residual on all 4 columns
    denom = np.linalg.norm(M_np) * np.linalg.norm(X_np) + np.linalg.norm(B_multi)
    rel_residual = float(np.linalg.norm(M_np @ X_np - B_multi) / denom)
    assert rel_residual < 1e-5, f"Multi-RHS relative residual {rel_residual:.2e} >= 1e-5"

    # Direct solution comparison
    err_multi = float(np.max(np.abs(X_np - npz["X_multi"])))
    assert err_multi < 1e-3, f"Multi-RHS direct solution error {err_multi:.2e} >= 1e-3"


def test_cholesky_failure_modes_and_invalidation(engine):
    """Verifies that bad pivots, NaNs, Infs, and non-finite RHS write NaNs to outputs and set proper status."""
    B = 1
    dev = engine.device

    # Case 1: Negative pivot (non-positive)
    M_neg = torch.eye(20, dtype=torch.float32, device=dev).unsqueeze(0)
    M_neg[0, 4, 4] = -1.0
    rhs = torch.ones((1, 20), dtype=torch.float32, device=dev)
    X, status = engine.compute_native_cholesky_solve(M_neg, rhs)
    torch.mps.synchronize()
    assert status.item() == -1
    assert torch.all(torch.isnan(X))
    assert torch.all(torch.isnan(engine.L_factor))

    # Case 2: Zero pivot
    M_zero = torch.eye(20, dtype=torch.float32, device=dev).unsqueeze(0)
    M_zero[0, 8, 8] = 0.0
    X, status = engine.compute_native_cholesky_solve(M_zero, rhs)
    torch.mps.synchronize()
    assert status.item() == -1
    assert torch.all(torch.isnan(X))
    assert torch.all(torch.isnan(engine.L_factor))

    # Case 3: NaN in matrix M
    M_nan = torch.eye(20, dtype=torch.float32, device=dev).unsqueeze(0)
    M_nan[0, 2, 3] = float("nan")
    X, status = engine.compute_native_cholesky_solve(M_nan, rhs)
    torch.mps.synchronize()
    assert status.item() == -1
    assert torch.all(torch.isnan(X))
    assert torch.all(torch.isnan(engine.L_factor))

    # Case 4: Positive infinity in matrix M
    M_inf = torch.eye(20, dtype=torch.float32, device=dev).unsqueeze(0)
    M_inf[0, 2, 2] = float("inf")
    X, status = engine.compute_native_cholesky_solve(M_inf, rhs)
    torch.mps.synchronize()
    assert status.item() == -1
    assert torch.all(torch.isnan(X))
    assert torch.all(torch.isnan(engine.L_factor))

    # Case 5: NaN in RHS vector B
    M_valid = torch.eye(20, dtype=torch.float32, device=dev).unsqueeze(0)
    rhs_nan = torch.ones((1, 20), dtype=torch.float32, device=dev)
    rhs_nan[0, 5] = float("nan")
    X, status = engine.compute_native_cholesky_solve(M_valid, rhs_nan)
    torch.mps.synchronize()
    assert status.item() == -2
    assert torch.all(torch.isnan(X))
    assert torch.all(torch.isnan(engine.L_factor))


def test_mixed_batch_buffer_reuse_no_stale_data(canonical):
    """Verifies that persistent buffers in mixed valid/invalid batches do not leak stale data."""
    B = 2
    engine_b2 = RepresentativePhysicsSlice(batch_size=B, canonical=canonical)
    dev = engine_b2.device

    M_spd = torch.eye(20, dtype=torch.float32, device=dev)
    M_spd[2, 2] = 4.0
    M_bad = torch.eye(20, dtype=torch.float32, device=dev)
    M_bad[4, 4] = -2.0

    rhs_1 = torch.ones((B, 20), dtype=torch.float32, device=dev)
    rhs_2 = torch.full((B, 20), 3.0, dtype=torch.float32, device=dev)

    # Pre-allocated explicit persistent solution buffer to test reuse directly
    X_persistent = torch.zeros((B, 20, 1), dtype=torch.float32, device=dev)
    ptr_initial = X_persistent.data_ptr()

    # Step 1: Both worlds valid
    M_batch = torch.stack([M_spd, M_spd])
    X, status = engine_b2.compute_native_cholesky_solve(M_batch, rhs_1, X_out=X_persistent)
    torch.mps.synchronize()
    assert X.data_ptr() == ptr_initial
    assert status[0].item() == 0 and status[1].item() == 0
    assert torch.all(torch.isfinite(X[0])) and torch.all(torch.isfinite(X[1]))
    prev_valid_sol_1 = X[1].clone()

    # Step 2: World 0 valid, World 1 INVALID.
    # World 1 must be overwritten with NaNs and NOT retain prev_valid_sol_1 in X_persistent!
    M_batch = torch.stack([M_spd, M_bad])
    X, status = engine_b2.compute_native_cholesky_solve(M_batch, rhs_2, X_out=X_persistent)
    torch.mps.synchronize()
    assert X.data_ptr() == ptr_initial
    assert status[0].item() == 0
    assert status[1].item() == -1
    assert torch.all(torch.isfinite(X[0]))
    assert torch.all(torch.isnan(X[1])) # Must be NaNs, not stale values!
    assert torch.all(torch.isnan(engine_b2.L_factor[1]))

    # Step 3: World 0 INVALID, World 1 VALID.
    M_batch = torch.stack([M_bad, M_spd])
    X, status = engine_b2.compute_native_cholesky_solve(M_batch, rhs_1, X_out=X_persistent)
    torch.mps.synchronize()
    assert X.data_ptr() == ptr_initial
    assert status[0].item() == -1
    assert status[1].item() == 0
    assert torch.all(torch.isnan(X[0]))
    assert torch.all(torch.isfinite(X[1]))


def test_downstream_constrained_solve_failure_propagation(canonical):
    """Verifies that upstream solver failure propagates to constrained solve outputs (qacc, qfrc_c = NaN)."""
    B = 2
    engine_b2 = RepresentativePhysicsSlice(batch_size=B, canonical=canonical)
    dev = engine_b2.device

    # Create dummy inputs
    qpos = torch.tile(torch.from_numpy(canonical.model.key_qpos[0].astype(np.float32)), (B, 1)).to(dev)
    qvel = torch.zeros((B, 20), dtype=torch.float32, device=dev)

    # Corrupt world 1 mass with NaN
    mass_batch = torch.tile(torch.from_numpy(canonical.body_mass.astype(np.float32)), (B, 1)).to(dev)
    mass_batch[1, 2] = float("nan")

    out = engine_b2.forward(qpos, qvel, per_world_mass=mass_batch)
    torch.mps.synchronize()

    # World 0 must succeed
    assert out.solver_status[0].item() == 0
    assert torch.all(torch.isfinite(out.qacc[0]))
    assert torch.all(torch.isfinite(out.qfrc_constraint[0]))

    # World 1 must fail and propagate NaNs to physics outputs
    assert out.solver_status[1].item() != 0
    assert torch.all(torch.isnan(out.qacc[1]))
    assert torch.all(torch.isnan(out.qfrc_constraint[1]))


def test_helper_batch_and_shape_guards(engine):
    """Verifies input shape, dtype, device guards and safe dynamic buffer resizing on public native helpers."""
    dev = engine.device

    # 1. compute_native_dynamics guards
    qpos_good = torch.zeros((2, 21), dtype=torch.float32, device=dev)
    qvel_good = torch.zeros((2, 20), dtype=torch.float32, device=dev)

    # CPU rejection
    with pytest.raises((ValueError, TypeError)):
        engine.compute_native_dynamics(qpos_good.cpu(), qvel_good)

    # Non-float32 rejection
    with pytest.raises(TypeError):
        engine.compute_native_dynamics(qpos_good.to(torch.float64), qvel_good)

    # Bad shape rejection
    with pytest.raises(ValueError):
        engine.compute_native_dynamics(torch.zeros((2, 20), dtype=torch.float32, device=dev), qvel_good)

    # Batch mismatch rejection
    with pytest.raises(ValueError):
        engine.compute_native_dynamics(qpos_good, torch.zeros((3, 20), dtype=torch.float32, device=dev))

    # Safe buffer resizing test (calling with B=4 when engine initialized with B=1)
    qpos_b4 = torch.zeros((4, 21), dtype=torch.float32, device=dev)
    qvel_b4 = torch.zeros((4, 20), dtype=torch.float32, device=dev)
    engine.compute_native_dynamics(qpos_b4, qvel_b4)
    assert engine.batch_size == 4
    assert engine.M_eff.shape == (4, 20, 20)

    # 2. compute_native_cholesky_solve guards
    M_good = torch.eye(20, dtype=torch.float32, device=dev).unsqueeze(0).expand(4, -1, -1)
    rhs_good = torch.ones((4, 20), dtype=torch.float32, device=dev)

    # CPU rejection
    with pytest.raises((ValueError, TypeError)):
        engine.compute_native_cholesky_solve(M_good.cpu(), rhs_good)

    # Non-float32 rejection
    with pytest.raises(TypeError):
        engine.compute_native_cholesky_solve(M_good, rhs_good.to(torch.int32))

    # Shape rejection
    with pytest.raises(ValueError):
        engine.compute_native_cholesky_solve(torch.zeros((4, 19, 19), dtype=torch.float32, device=dev), rhs_good)

    # Safe resizing to B=2
    M_b2 = torch.eye(20, dtype=torch.float32, device=dev).unsqueeze(0).expand(2, -1, -1)
    rhs_b2 = torch.ones((2, 20), dtype=torch.float32, device=dev)
    X, status = engine.compute_native_cholesky_solve(M_b2, rhs_b2)
    assert engine.batch_size == 2
    assert X.shape == (2, 20, 1)


def test_heterogeneous_batch_domain_randomization(engine, canonical):
    """Verifies that a batch with per-world randomized mass/armature runs completely independently on GPU."""
    B = 3
    engine_b3 = RepresentativePhysicsSlice(batch_size=B, canonical=canonical)

    qpos_std = canonical.model.key_qpos[0].copy()
    qvel_zero = np.zeros(canonical.nv, dtype=np.float64)

    # 3 worlds:
    # World 0: default mass, default armature
    # World 1: +10% trunk mass
    # World 2: +20% leg armature
    mass_batch = np.tile(canonical.body_mass, (B, 1))
    mass_batch[1, 2] *= 1.10

    arm_batch = np.tile(canonical.dof_armature, (B, 1))
    arm_batch[2, 6:11] *= 1.20

    qpos_batch = torch.from_numpy(np.tile(qpos_std, (B, 1)).astype(np.float32)).to(engine.device)
    qvel_batch = torch.from_numpy(np.tile(qvel_zero, (B, 1)).astype(np.float32)).to(engine.device)
    mass_t = torch.from_numpy(mass_batch.astype(np.float32)).to(engine.device)
    arm_t = torch.from_numpy(arm_batch.astype(np.float32)).to(engine.device)

    engine_b3.compute_native_dynamics(qpos_batch, qvel_batch, per_world_mass=mass_t, per_world_armature=arm_t)
    torch.mps.synchronize()

    # Assert world 0 and world 1 have different mass matrix (mass perturbation)
    M0 = engine_b3.M_eff[0].cpu().numpy()
    M1 = engine_b3.M_eff[1].cpu().numpy()
    M2 = engine_b3.M_eff[2].cpu().numpy()

    diff_0_1 = np.max(np.abs(M0 - M1))
    diff_0_2 = np.max(np.abs(M0 - M2))

    assert diff_0_1 > 0.01, f"Expected M to differ for randomized mass: diff={diff_0_1}"
    assert np.isclose(diff_0_2, 0.000361548, rtol=1e-3), f"Expected exact armature difference 0.0003615, got {diff_0_2}"


def test_forward_input_validation_zero_dispatches(engine, monkeypatch):
    """Verifies that malformed inputs to forward() raise errors with zero GPU kernel launches."""
    dev = engine.device
    dispatches = []

    original_launch = engine.km.launch
    def mock_launch(*args, **kwargs):
        dispatches.append(args[0] if args else kwargs.get("kernel_name"))
        return original_launch(*args, **kwargs)

    monkeypatch.setattr(engine.km, "launch", mock_launch)

    # 1. Invalid qpos shape (e.g. (1, 15) instead of (1, 21))
    dispatches.clear()
    with pytest.raises(ValueError, match="Expected qpos shape"):
        engine.forward(torch.zeros((1, 15), dtype=torch.float32, device=dev), torch.zeros((1, 20), dtype=torch.float32, device=dev))
    assert len(dispatches) == 0, f"Expected 0 dispatches on bad qpos shape, got {dispatches}"

    # 2. Invalid dtype (int32 instead of float32)
    dispatches.clear()
    with pytest.raises(TypeError, match="Inputs must be float32"):
        engine.forward(torch.zeros((1, 21), dtype=torch.int32, device=dev), torch.zeros((1, 20), dtype=torch.float32, device=dev))
    assert len(dispatches) == 0, f"Expected 0 dispatches on int32 qpos, got {dispatches}"

    # 3. CPU device tensor
    dispatches.clear()
    with pytest.raises(ValueError, match="Inputs must be on device"):
        engine.forward(torch.zeros((1, 21), dtype=torch.float32), torch.zeros((1, 20), dtype=torch.float32))
    assert len(dispatches) == 0, f"Expected 0 dispatches on CPU tensor, got {dispatches}"

    # 4. Batch size 0 (empty batch)
    dispatches.clear()
    with pytest.raises(ValueError, match="Batch size must be positive"):
        engine.forward(torch.zeros((0, 21), dtype=torch.float32, device=dev), torch.zeros((0, 20), dtype=torch.float32, device=dev))
    assert len(dispatches) == 0, f"Expected 0 dispatches on empty batch, got {dispatches}"

    # 5. Batch mismatch between qpos and qvel
    dispatches.clear()
    with pytest.raises(ValueError, match="Batch dimension mismatch"):
        engine.forward(torch.zeros((2, 21), dtype=torch.float32, device=dev), torch.zeros((1, 20), dtype=torch.float32, device=dev))
    assert len(dispatches) == 0, f"Expected 0 dispatches on batch mismatch, got {dispatches}"


def test_solve_status_isolation_when_cloned(engine):
    """Verifies that cloning status preserves failure state across subsequent successful dispatches."""
    dev = engine.device
    M_bad = torch.eye(20, dtype=torch.float32, device=dev).unsqueeze(0)
    M_bad[0, 4, 4] = -1.0  # Invalid (negative pivot)
    M_good = torch.eye(20, dtype=torch.float32, device=dev).unsqueeze(0)  # Valid SPD
    rhs = torch.ones((1, 20), dtype=torch.float32, device=dev)

    # Solve 1: failure
    _, status1_ref = engine.compute_native_cholesky_solve(M_bad, rhs)
    torch.mps.synchronize()
    assert status1_ref.item() == -1

    # Caller clones status to preserve it
    status1_saved = status1_ref.clone()

    # Solve 2: success
    _, status2_ref = engine.compute_native_cholesky_solve(M_good, rhs)
    torch.mps.synchronize()
    assert status2_ref.item() == 0

    # Underlying buffer was overwritten by Solve 2
    assert status1_ref.item() == 0
    # But cloned status remains -1!
    assert status1_saved.item() == -1

