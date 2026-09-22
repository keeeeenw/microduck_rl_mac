"""Automated Test Suite for Milestone 3A: Metal Constraint Solver Qualification on Oracle Constraints.

Tests the actual Metal constraint solver kernel (kernel_oracle_constrained_solve)
against independent CPU MuJoCo oracle constraints, separating solver qualification
from autonomous contact generation as mandated by the Milestone 3A review.

Validation Triad:
1. Tier 1 (Implementation Parity): Metal PGS vs independent CPU Python PGS on identical (A, b, lambda >= 0) QP (< 1e-4).
2. Tier 2 (Solver Convergence): True KKT primal feasibility, dual feasibility (g >= 0), complementarity (|lambda * g|),
   and diagonally scaled projected-gradient residuals, evaluated independently on CPU.
3. Tier 3 (Newton Parity Diagnostic): Metal PGS vs canonical MuJoCo CPU mjSOL_NEWTON reference with dimensionally
   separated metrics (linear N vs angular N*m, linear m/s^2 vs angular rad/s^2).
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
ALL_SCENARIOS = sorted([p.stem for p in CORPUS_DIR.glob("*.npz")])
CONTACT_SCENARIOS = [
    "standing_zero_vel",
    "standing_moving_vel",
    "single_support_zero_vel",
    "single_support_moving_vel",
    "tilted_landing",
    "near_contact_separation",
    "contact_onset",
    "randomized_model_standing",
    "crouched_pose",
    "asymmetric_pose",
    "high_condition_mass_matrix",
    "nonzero_applied_force",
]


@pytest.fixture(scope="module")
def canonical():
    return load_canonical_model()


@pytest.fixture(scope="module")
def engine(canonical):
    return RepresentativePhysicsSlice(batch_size=1, canonical=canonical)


def cpu_assemble_and_pgs(
    L_np: np.ndarray,
    f_smooth_np: np.ndarray,
    J_np: np.ndarray,
    aref_np: np.ndarray,
    R_np: np.ndarray,
    max_iters: int = 100,
    tol: float = 1e-5,
):
    """Independent deterministic CPU reference implementation of Delassus assembly and PGS."""
    nefc = len(aref_np)
    if nefc == 0:
        y0 = np.zeros(20, dtype=np.float32)
        for i in range(20):
            s = f_smooth_np[i]
            for p in range(i):
                s -= L_np[i, p] * y0[p]
            y0[i] = s / L_np[i, i]
        a0 = np.zeros(20, dtype=np.float32)
        for i in range(19, -1, -1):
            s = y0[i]
            for p in range(i + 1, 20):
                s -= L_np[p, i] * a0[p]
            a0[i] = s / L_np[i, i]
        return np.zeros(0, dtype=np.float32), np.zeros(20, dtype=np.float32), a0, 0, 0.0

    # 1. Forward substitution: L Y = J^T => Y = L^-1 J^T
    Y = np.zeros((nefc, 20), dtype=np.float32)
    for i in range(nefc):
        for k in range(20):
            s = J_np[i, k]
            for p in range(k):
                s -= L_np[k, p] * Y[i, p]
            Y[i, k] = s / L_np[k, k]

    # 2. Delassus matrix A = Y Y^T + diag(R)
    A = Y @ Y.T + np.diag(R_np)

    # 3. Unconstrained a0
    y0 = np.zeros(20, dtype=np.float32)
    for i in range(20):
        s = f_smooth_np[i]
        for p in range(i):
            s -= L_np[i, p] * y0[p]
        y0[i] = s / L_np[i, i]
    a0 = np.zeros(20, dtype=np.float32)
    for i in range(19, -1, -1):
        s = y0[i]
        for p in range(i + 1, 20):
            s -= L_np[p, i] * a0[p]
        a0[i] = s / L_np[i, i]

    # 4. Linear term b = J a0 - aref
    b = J_np @ a0 - aref_np

    # 5. PGS solve with gradient tracking
    lam = np.zeros(nefc, dtype=np.float32)
    g = b.copy()
    iters = max_iters
    for it in range(max_iters):
        max_delta = 0.0
        for i in range(nefc):
            delta = -g[i] / A[i, i]
            lam_new = max(0.0, lam[i] + delta)
            d_act = lam_new - lam[i]
            if abs(d_act) > 1e-12:
                g += A[:, i] * d_act
                lam[i] = lam_new
                max_delta = max(max_delta, abs(d_act))
        if max_delta < tol:
            iters = it + 1
            break

    # 6. Projected gradient residual
    proj_res = float(np.max(np.abs(lam - np.maximum(0.0, lam - g / np.diag(A)))))

    # 7. Constraint force f_c = J^T lam
    f_c = J_np.T @ lam

    # 8. delta_a = (L^T)^-1 (Y lam)
    y_c = Y.T @ lam
    delta_a = np.zeros(20, dtype=np.float32)
    for i in range(19, -1, -1):
        s = y_c[i]
        for p in range(i + 1, 20):
            s -= L_np[p, i] * delta_a[p]
        delta_a[i] = s / L_np[i, i]
    qacc = a0 + delta_a

    return lam, f_c, qacc, iters, proj_res


# ==============================================================================
# Tier 1: Implementation Parity (Metal PGS vs Independent CPU PGS on Same QP)
# ==============================================================================

@pytest.mark.parametrize("scenario", CONTACT_SCENARIOS)
def test_metal_pgs_vs_cpu_pgs_same_qp(engine, scenario):
    """Tier 1: Verifies that Metal PGS matches independent CPU PGS on the exact same bounded QP."""
    npz = np.load(CORPUS_DIR / f"{scenario}.npz")
    nefc_val = int(npz["nefc"])
    capacity = 32

    L_np = npz["L"].astype(np.float32)
    f_smooth_np = npz["qfrc_smooth"].astype(np.float32)
    J_sub = npz["efc_J"].astype(np.float32)
    aref_sub = npz["efc_aref"].astype(np.float32)
    R_sub = npz["efc_R"].astype(np.float32)

    # CPU same-QP solve
    lam_cpu, fc_cpu, qacc_cpu, iters_cpu, res_cpu = cpu_assemble_and_pgs(
        L_np, f_smooth_np, J_sub, aref_sub, R_sub, max_iters=100, tol=1e-5
    )

    # Metal solve with actual efc_type from fixture
    L = torch.from_numpy(L_np).unsqueeze(0).to(engine.device)
    f_smooth = torch.from_numpy(f_smooth_np).unsqueeze(0).to(engine.device)
    J = torch.zeros((1, capacity, 20), dtype=torch.float32, device=engine.device)
    J[0, :nefc_val] = torch.from_numpy(J_sub)
    aref = torch.zeros((1, capacity), dtype=torch.float32, device=engine.device)
    aref[0, :nefc_val] = torch.from_numpy(aref_sub)
    R = torch.ones((1, capacity), dtype=torch.float32, device=engine.device)
    R[0, :nefc_val] = torch.from_numpy(R_sub)
    nefc = torch.tensor([nefc_val], dtype=torch.int32, device=engine.device)
    efc_type = torch.full((1, capacity), 6, dtype=torch.int32, device=engine.device)
    efc_type[0, :nefc_val] = torch.from_numpy(npz["efc_type"].astype(np.int32))

    res = engine.solve_oracle_constraints(L, f_smooth, J, aref, R, nefc, efc_type=efc_type, capacity=capacity)
    torch.mps.synchronize()

    lam_metal = res["lambda"][0, :nefc_val].cpu().numpy()
    fc_metal = res["qfrc_constraint"][0].cpu().numpy()
    qacc_metal = res["qacc"][0].cpu().numpy()

    err_lam = float(np.max(np.abs(lam_metal - lam_cpu)))
    err_fc = float(np.max(np.abs(fc_metal - fc_cpu)))
    err_qacc = float(np.max(np.abs(qacc_metal - qacc_cpu)))

    assert err_lam < 1e-4, f"[{scenario}] Metal vs CPU PGS lambda error {err_lam:.2e} >= 1e-4"
    assert err_fc < 1e-4, f"[{scenario}] Metal vs CPU PGS force error {err_fc:.2e} >= 1e-4"
    assert err_qacc < 1e-3, f"[{scenario}] Metal vs CPU PGS qacc error {err_qacc:.2e} >= 1e-3"


# ==============================================================================
# Tier 2: Solver Convergence (Independent KKT Checks & Synthetic QP)
# ==============================================================================

@pytest.mark.parametrize("scenario", ALL_SCENARIOS)
def test_metal_solver_kkt_convergence_residuals(engine, scenario):
    """Tier 2: Independently reconstructs A and b on CPU and verifies primal, dual, and complementarity KKT conditions."""
    npz = np.load(CORPUS_DIR / f"{scenario}.npz")
    nefc_val = int(npz["nefc"])
    capacity = 32

    L = torch.from_numpy(npz["L"].astype(np.float32)).unsqueeze(0).to(engine.device)
    f_smooth = torch.from_numpy(npz["qfrc_smooth"].astype(np.float32)).unsqueeze(0).to(engine.device)
    J = torch.zeros((1, capacity, 20), dtype=torch.float32, device=engine.device)
    aref = torch.zeros((1, capacity), dtype=torch.float32, device=engine.device)
    R = torch.ones((1, capacity), dtype=torch.float32, device=engine.device)
    efc_type = torch.full((1, capacity), 6, dtype=torch.int32, device=engine.device)

    if nefc_val > 0:
        J[0, :nefc_val] = torch.from_numpy(npz["efc_J"].astype(np.float32))
        aref[0, :nefc_val] = torch.from_numpy(npz["efc_aref"].astype(np.float32))
        R[0, :nefc_val] = torch.from_numpy(npz["efc_R"].astype(np.float32))
        efc_type[0, :nefc_val] = torch.from_numpy(npz["efc_type"].astype(np.int32))

    nefc = torch.tensor([nefc_val], dtype=torch.int32, device=engine.device)
    res = engine.solve_oracle_constraints(L, f_smooth, J, aref, R, nefc, efc_type=efc_type, capacity=capacity)
    torch.mps.synchronize()

    status = res["solver_status"].item()
    assert status in (0, 1), f"[{scenario}] Unexpected solver error status {status}"

    lam = res["lambda"][0, :nefc_val].cpu().numpy()

    if nefc_val > 0:
        # Independent CPU reconstruction of A and b
        J_np = npz["efc_J"]
        aref_np = npz["efc_aref"]
        R_np = npz["efc_R"]
        Y_cpu = np.linalg.solve(npz["L"], J_np.T).T
        A_cpu = Y_cpu @ Y_cpu.T + np.diag(R_np)
        a0_cpu = np.linalg.solve(npz["M"], npz["qfrc_smooth"])
        b_cpu = J_np @ a0_cpu - aref_np

        # Independent dual gradient computation: g = A lam + b
        g = A_cpu @ lam + b_cpu

        # 1. Primal feasibility: lambda >= -1e-6
        primal_infeas = float(np.max(np.maximum(0.0, -lam)))
        assert primal_infeas <= 1e-6, f"[{scenario}] Primal infeasibility {primal_infeas:.2e} > 1e-6"

        # 2. Dual feasibility: g >= -epsilon
        dual_infeas = float(np.max(np.maximum(0.0, -g)))

        # 3. Complementarity: |lambda * g|
        comp = float(np.max(np.abs(lam * g)))

        # 4. Diagonally scaled projected-gradient residual: |lambda - max(0, lambda - g / diag(A))|
        proj_res = float(np.max(np.abs(lam - np.maximum(0.0, lam - g / np.diag(A_cpu)))))

        if status == 0:
            # Fully converged solves must satisfy strict dual feasibility, complementarity, and projected residual
            assert dual_infeas <= 2e-4, f"[{scenario}] Converged dual infeasibility {dual_infeas:.2e} > 2e-4"
            assert comp <= 2.5e-3, f"[{scenario}] Converged complementarity {comp:.2e} > 2.5e-3"
            assert proj_res <= 2e-5, f"[{scenario}] Converged projected gradient residual {proj_res:.2e} > 2e-5"
        else:
            # Unconverged solves (crouched_pose, high_condition, asymmetric, nonzero_force) remain bounded
            dual_bound = 0.15 if scenario == "high_condition_mass_matrix" else 0.01
            assert dual_infeas <= dual_bound, f"[{scenario}] Unconverged dual infeasibility {dual_infeas:.2e} > {dual_bound}"
            assert comp <= 0.25, f"[{scenario}] Unconverged complementarity {comp:.2e} > 0.25"
            assert proj_res <= 0.01, f"[{scenario}] Unconverged projected residual {proj_res:.2e} > 0.01"
    else:
        assert res["dual_residual"].item() == 0.0


def test_synthetic_spd_qp_negative_g_and_insufficient_iters(engine):
    """Demonstrates that negative dual gradients (g < 0) and insufficient iterations are strictly detected."""
    # Construct a coupled 2-variable SPD QP where lambda=0 has negative dual gradient (g = b = -5.0 < 0)
    # L = I_20, coupled rows in J produce non-zero off-diagonal Delassus entry A_01 = A_10 = 1.0.
    capacity = 32
    L = torch.eye(20, dtype=torch.float32, device=engine.device).unsqueeze(0)
    f_smooth = torch.zeros((1, 20), dtype=torch.float32, device=engine.device)

    J = torch.zeros((1, capacity, 20), dtype=torch.float32, device=engine.device)
    J[0, 0, 0] = 1.0
    J[0, 0, 1] = 1.0 # row 0 norm^2 = 2
    J[0, 1, 0] = 1.0
    J[0, 1, 2] = 1.0 # row 1 norm^2 = 2, dot(row 0, row 1) = 1.0

    aref = torch.zeros((1, capacity), dtype=torch.float32, device=engine.device)
    aref[0, 0] = 5.0 # causes b_0 = -5.0 < 0
    aref[0, 1] = 5.0 # causes b_1 = -5.0 < 0

    R = torch.ones((1, capacity), dtype=torch.float32, device=engine.device)
    nefc = torch.tensor([2], dtype=torch.int32, device=engine.device)

    # Test A: Insufficient iterations (max_iters = 1) -> must report status 1 and high residual (0.370)
    res_insufficient = engine.solve_oracle_constraints(L, f_smooth, J, aref, R, nefc, max_iters=1, capacity=capacity)
    torch.mps.synchronize()

    assert res_insufficient["solver_status"].item() == 1, "Expected status 1 for insufficient iterations"
    assert res_insufficient["actual_iters"].item() == 1
    assert res_insufficient["dual_residual"].item() > 0.1, "Expected significant residual for insufficient iterations"

    # Test B: Sufficient iterations (max_iters = 30) -> converges to analytical optimum lambda* = [1.25, 1.25], status 0
    res_converged = engine.solve_oracle_constraints(L, f_smooth, J, aref, R, nefc, max_iters=30, capacity=capacity)
    torch.mps.synchronize()

    assert res_converged["solver_status"].item() == 0, "Expected status 0 for converged synthetic solve"
    assert res_converged["dual_residual"].item() < 1e-4
    lam_out = res_converged["lambda"][0, :2].cpu().numpy()
    # A = [[3, 1], [1, 3]]. A^-1 [5, 5] = [1.25, 1.25]
    assert np.allclose(lam_out, [1.25, 1.25], atol=1e-4)


# ==============================================================================
# Tier 3: Canonical MuJoCo Newton Approximation Parity
# ==============================================================================

@pytest.mark.parametrize("scenario", ALL_SCENARIOS)
def test_metal_solver_canonical_newton_parity(engine, scenario):
    """Tier 3: Measures practical approximation error against canonical MuJoCo CPU Newton solver.

    Dimensionally separated:
    - Linear constraint force (0:3) in N
    - Angular constraint torque (3:6) in N*m
    - Joint constraint torque (6:20) in N*m
    - Linear acceleration (0:3) in m/s^2
    - Angular/joint acceleration (3:20) in rad/s^2
    """
    npz = np.load(CORPUS_DIR / f"{scenario}.npz")
    nefc_val = int(npz["nefc"])
    capacity = 32

    L = torch.from_numpy(npz["L"].astype(np.float32)).unsqueeze(0).to(engine.device)
    f_smooth = torch.from_numpy(npz["qfrc_smooth"].astype(np.float32)).unsqueeze(0).to(engine.device)
    J = torch.zeros((1, capacity, 20), dtype=torch.float32, device=engine.device)
    aref = torch.zeros((1, capacity), dtype=torch.float32, device=engine.device)
    R = torch.ones((1, capacity), dtype=torch.float32, device=engine.device)
    efc_type = torch.full((1, capacity), 6, dtype=torch.int32, device=engine.device)

    if nefc_val > 0:
        J[0, :nefc_val] = torch.from_numpy(npz["efc_J"].astype(np.float32))
        aref[0, :nefc_val] = torch.from_numpy(npz["efc_aref"].astype(np.float32))
        R[0, :nefc_val] = torch.from_numpy(npz["efc_R"].astype(np.float32))
        efc_type[0, :nefc_val] = torch.from_numpy(npz["efc_type"].astype(np.int32))

    nefc = torch.tensor([nefc_val], dtype=torch.int32, device=engine.device)
    res = engine.solve_oracle_constraints(L, f_smooth, J, aref, R, nefc, efc_type=efc_type, capacity=capacity)
    torch.mps.synchronize()

    qfrc_c_gpu = res["qfrc_constraint"][0].cpu().numpy()
    qacc_gpu = res["qacc"][0].cpu().numpy()
    f_ref = npz["qfrc_constraint"]
    a_ref = npz["qacc"]

    err_f_lin = float(np.max(np.abs(qfrc_c_gpu[:3] - f_ref[:3])))
    err_f_rot = float(np.max(np.abs(qfrc_c_gpu[3:6] - f_ref[3:6])))
    err_f_jnt = float(np.max(np.abs(qfrc_c_gpu[6:20] - f_ref[6:20])))
    err_a_lin = float(np.max(np.abs(qacc_gpu[:3] - a_ref[:3])))
    err_a_rot = float(np.max(np.abs(qacc_gpu[3:20] - a_ref[3:20])))

    # Acceptance criteria: nominal scenarios satisfy <= 0.05 N, N*m, m/s^2, rad/s^2
    assert err_f_lin <= 0.05, f"[{scenario}] Linear force error {err_f_lin:.2e} N > 0.05 N"
    assert err_f_rot <= 0.05, f"[{scenario}] Angular torque error {err_f_rot:.2e} N*m > 0.05 N*m"
    assert err_f_jnt <= 0.05, f"[{scenario}] Joint torque error {err_f_jnt:.2e} N*m > 0.05 N*m"
    assert err_a_lin <= 0.05, f"[{scenario}] Linear acceleration error {err_a_lin:.2e} m/s^2 > 0.05 m/s^2"

    # Rotational acceleration: high-condition mass matrix (kappa ~ 1000) is tracked as an explicit diagnostic exception
    rot_acc_limit = 0.15 if scenario == "high_condition_mass_matrix" else 0.05
    assert err_a_rot <= rot_acc_limit, f"[{scenario}] Angular/joint acc error {err_a_rot:.2e} rad/s^2 > {rot_acc_limit}"


# ==============================================================================
# Configurable Capacity & Output Layout Safety Tests
# ==============================================================================

def test_configurable_capacity_layout_integrity(engine):
    """Verifies that capacity < 32 (e.g. 12, 16) produces correct uncorrupted memory layout across multiple worlds."""
    npz_single = np.load(CORPUS_DIR / "single_support_zero_vel.npz") # nefc = 12
    B = 2
    capacity = 16

    L = torch.cat([
        torch.from_numpy(npz_single["L"].astype(np.float32)).unsqueeze(0),
        torch.from_numpy(npz_single["L"].astype(np.float32)).unsqueeze(0),
    ], dim=0).to(engine.device)

    f_smooth = torch.cat([
        torch.from_numpy(npz_single["qfrc_smooth"].astype(np.float32)).unsqueeze(0),
        torch.from_numpy(npz_single["qfrc_smooth"].astype(np.float32)).unsqueeze(0),
    ], dim=0).to(engine.device)

    J = torch.zeros((B, capacity, 20), dtype=torch.float32, device=engine.device)
    aref = torch.zeros((B, capacity), dtype=torch.float32, device=engine.device)
    R = torch.ones((B, capacity), dtype=torch.float32, device=engine.device)
    efc_type = torch.full((B, capacity), 6, dtype=torch.int32, device=engine.device)

    J[:, :12] = torch.from_numpy(npz_single["efc_J"][:12].astype(np.float32))
    aref[:, :12] = torch.from_numpy(npz_single["efc_aref"][:12].astype(np.float32))
    R[:, :12] = torch.from_numpy(npz_single["efc_R"][:12].astype(np.float32))
    efc_type[:, :12] = torch.from_numpy(npz_single["efc_type"][:12].astype(np.int32))
    nefc = torch.tensor([12, 12], dtype=torch.int32, device=engine.device)

    res = engine.solve_oracle_constraints(L, f_smooth, J, aref, R, nefc, efc_type=efc_type, capacity=capacity)
    torch.mps.synchronize()

    # Verify tensor shape matches requested capacity (2, 16)
    assert res["lambda"].shape == (B, capacity), f"Expected shape ({B}, {capacity}), got {res['lambda'].shape}"

    # Verify world 0 and world 1 have identical valid forces and no stale memory bleed
    lam_w0 = res["lambda"][0].cpu().numpy()
    lam_w1 = res["lambda"][1].cpu().numpy()
    assert np.allclose(lam_w0, lam_w1, atol=1e-6)
    assert np.all(lam_w0[12:] == 0.0)
    assert np.all(lam_w1[12:] == 0.0)

    # Capacity change reuse: switch from capacity=16 to capacity=32 on same engine
    res_32 = engine.solve_oracle_constraints(
        L[:1], f_smooth[:1],
        torch.zeros((1, 32, 20), dtype=torch.float32, device=engine.device),
        torch.zeros((1, 32), dtype=torch.float32, device=engine.device),
        torch.ones((1, 32), dtype=torch.float32, device=engine.device),
        torch.tensor([0], dtype=torch.int32, device=engine.device),
        capacity=32,
    )
    torch.mps.synchronize()
    assert res_32["lambda"].shape == (1, 32)


def test_capacity_and_parameter_guards(engine):
    """Verifies that invalid capacities, non-positive max_iters, or non-finite tol raise clean Python ValueErrors."""
    L = torch.eye(20, dtype=torch.float32, device=engine.device).unsqueeze(0)
    f = torch.zeros((1, 20), dtype=torch.float32, device=engine.device)
    J = torch.zeros((1, 32, 20), dtype=torch.float32, device=engine.device)
    aref = torch.zeros((1, 32), dtype=torch.float32, device=engine.device)
    R = torch.ones((1, 32), dtype=torch.float32, device=engine.device)
    nefc = torch.tensor([0], dtype=torch.int32, device=engine.device)

    # Invalid capacity > 32
    with pytest.raises(ValueError, match="Capacity must be an integer in"):
        engine.solve_oracle_constraints(L, f, J, aref, R, nefc, capacity=33)

    # Invalid capacity < 1
    with pytest.raises(ValueError, match="Capacity must be an integer in"):
        engine.solve_oracle_constraints(L, f, J, aref, R, nefc, capacity=0)

    # Invalid max_iters <= 0
    with pytest.raises(ValueError, match="max_iters must be a positive integer"):
        engine.solve_oracle_constraints(L, f, J, aref, R, nefc, max_iters=0, capacity=32)

    # Invalid tol <= 0 or non-finite
    with pytest.raises(ValueError, match="tol must be a finite positive number"):
        engine.solve_oracle_constraints(L, f, J, aref, R, nefc, tol=-1e-4, capacity=32)


# ==============================================================================
# Numerical Overflow & Failure Invalidation Tests
# ==============================================================================

def test_numerical_overflow_detection_and_invalidation(engine):
    """Verifies that finite but overflowing inputs (tiny pivot causing float overflow) halt with status -3 and NaNs."""
    npz = np.load(CORPUS_DIR / "airborne.npz")
    capacity = 32

    # L with tiny pivot (1e-25) causes unconstrained substitution to overflow float32
    L_overflow = torch.from_numpy(npz["L"].astype(np.float32)).unsqueeze(0).to(engine.device)
    L_overflow[0, 0, 0] = 1e-25

    f_smooth = torch.from_numpy(npz["qfrc_smooth"].astype(np.float32)).unsqueeze(0).to(engine.device)
    f_smooth[0, 0] = 10.0 # non-zero numerator guarantees overflow

    J = torch.zeros((1, capacity, 20), dtype=torch.float32, device=engine.device)
    aref = torch.zeros((1, capacity), dtype=torch.float32, device=engine.device)
    R = torch.ones((1, capacity), dtype=torch.float32, device=engine.device)
    nefc = torch.tensor([0], dtype=torch.int32, device=engine.device)

    res = engine.solve_oracle_constraints(L_overflow, f_smooth, J, aref, R, nefc, capacity=capacity)
    torch.mps.synchronize()

    assert res["solver_status"].item() == -3, "Expected status -3 for numerical overflow"
    assert torch.all(torch.isnan(res["qacc"]))
    assert torch.isnan(res["dual_residual"]).item()


def test_mixed_batch_numerical_overflow_isolation(engine):
    """Verifies that in a batch with world 0 valid and world 1 overflowing, world 0 succeeds while world 1 fails."""
    npz = np.load(CORPUS_DIR / "airborne.npz")
    capacity = 32

    L = torch.cat([
        torch.from_numpy(npz["L"].astype(np.float32)).unsqueeze(0),
        torch.from_numpy(npz["L"].astype(np.float32)).unsqueeze(0),
    ], dim=0).to(engine.device)
    L[1, 0, 0] = 1e-25 # world 1 overflows

    f_smooth = torch.cat([
        torch.from_numpy(npz["qfrc_smooth"].astype(np.float32)).unsqueeze(0),
        torch.from_numpy(npz["qfrc_smooth"].astype(np.float32)).unsqueeze(0),
    ], dim=0).to(engine.device)
    f_smooth[1, 0] = 10.0

    J = torch.zeros((2, capacity, 20), dtype=torch.float32, device=engine.device)
    aref = torch.zeros((2, capacity), dtype=torch.float32, device=engine.device)
    R = torch.ones((2, capacity), dtype=torch.float32, device=engine.device)
    nefc = torch.tensor([0, 0], dtype=torch.int32, device=engine.device)

    res = engine.solve_oracle_constraints(L, f_smooth, J, aref, R, nefc, capacity=capacity)
    torch.mps.synchronize()

    assert res["solver_status"][0].item() == 0, "World 0 should succeed"
    assert res["solver_status"][1].item() == -3, "World 1 should fail with status -3"
    assert torch.all(torch.isfinite(res["qacc"][0]))
    assert torch.all(torch.isnan(res["qacc"][1]))


# ==============================================================================
# Chained Native Dynamics to Constraint Solve Integration Test
# ==============================================================================

def test_chained_native_dynamics_to_constraint_solve(engine):
    """Closes integration boundary: native dynamics -> native Cholesky -> native oracle constraint solve."""
    npz = np.load(CORPUS_DIR / "standing_zero_vel.npz")
    nefc_val = int(npz["nefc"])
    capacity = 32

    qpos = torch.from_numpy(npz["qpos"].astype(np.float32)).unsqueeze(0).to(engine.device)
    qvel = torch.from_numpy(npz["qvel"].astype(np.float32)).unsqueeze(0).to(engine.device)
    f_smooth = torch.from_numpy(npz["qfrc_smooth"].astype(np.float32)).unsqueeze(0).to(engine.device)

    # Step 1: Native GPU Dynamics
    engine.compute_native_dynamics(qpos, qvel)

    # Step 2: Native GPU Cholesky solve
    _, chol_status = engine.compute_native_cholesky_solve(engine.M_eff, f_smooth)
    torch.mps.synchronize()
    assert chol_status.item() == 0

    # Step 3: Native GPU Oracle Constrained Solve using native L_factor
    J = torch.zeros((1, capacity, 20), dtype=torch.float32, device=engine.device)
    J[0, :nefc_val] = torch.from_numpy(npz["efc_J"].astype(np.float32))
    aref = torch.zeros((1, capacity), dtype=torch.float32, device=engine.device)
    aref[0, :nefc_val] = torch.from_numpy(npz["efc_aref"].astype(np.float32))
    R = torch.ones((1, capacity), dtype=torch.float32, device=engine.device)
    R[0, :nefc_val] = torch.from_numpy(npz["efc_R"].astype(np.float32))
    nefc = torch.tensor([nefc_val], dtype=torch.int32, device=engine.device)
    efc_type = torch.full((1, capacity), 6, dtype=torch.int32, device=engine.device)
    efc_type[0, :nefc_val] = torch.from_numpy(npz["efc_type"].astype(np.int32))

    res = engine.solve_oracle_constraints(engine.L_factor, f_smooth, J, aref, R, nefc, efc_type=efc_type, capacity=capacity)
    torch.mps.synchronize()

    assert res["solver_status"].item() == 0
    qfrc_c_gpu = res["qfrc_constraint"][0].cpu().numpy()
    qacc_gpu = res["qacc"][0].cpu().numpy()

    err_force = float(np.max(np.abs(qfrc_c_gpu - npz["qfrc_constraint"])))
    err_qacc = float(np.max(np.abs(qacc_gpu - npz["qacc"])))

    assert err_force < 1e-4, f"Chained pipeline force error {err_force:.2e} >= 1e-4"
    assert err_qacc < 1e-3, f"Chained pipeline qacc error {err_qacc:.2e} >= 1e-3"


# ==============================================================================
# Special Scenarios & Input Safety Guards
# ==============================================================================

def test_zero_contact_airborne_parity(engine):
    """Verifies that nefc=0 yields exact unconstrained solve without spurious constraint forces."""
    npz = np.load(CORPUS_DIR / "airborne.npz")
    capacity = 32
    L = torch.from_numpy(npz["L"].astype(np.float32)).unsqueeze(0).to(engine.device)
    f_smooth = torch.from_numpy(npz["qfrc_smooth"].astype(np.float32)).unsqueeze(0).to(engine.device)
    J = torch.zeros((1, capacity, 20), dtype=torch.float32, device=engine.device)
    aref = torch.zeros((1, capacity), dtype=torch.float32, device=engine.device)
    R = torch.ones((1, capacity), dtype=torch.float32, device=engine.device)
    nefc = torch.tensor([0], dtype=torch.int32, device=engine.device)

    res = engine.solve_oracle_constraints(L, f_smooth, J, aref, R, nefc, capacity=capacity)
    torch.mps.synchronize()

    assert res["solver_status"].item() == 0
    assert res["actual_iters"].item() == 0
    assert res["dual_residual"].item() == 0.0
    assert torch.all(res["qfrc_constraint"] == 0.0)
    assert torch.all(res["lambda"] == 0.0)
    err_qacc = float(torch.max(torch.abs(res["qacc"][0] - torch.from_numpy(npz["qacc"].astype(np.float32)).to(engine.device))))
    assert err_qacc < 1e-5, f"Airborne qacc error {err_qacc:.2e} >= 1e-5"


def test_nonzero_applied_force_parity(engine):
    """Verifies that qfrc_smooth != -qfrc_bias (applied forces + actuator forces) solves correctly."""
    npz = np.load(CORPUS_DIR / "nonzero_applied_force.npz")
    nefc_val = int(npz["nefc"])
    capacity = 32

    L = torch.from_numpy(npz["L"].astype(np.float32)).unsqueeze(0).to(engine.device)
    f_smooth = torch.from_numpy(npz["qfrc_smooth"].astype(np.float32)).unsqueeze(0).to(engine.device)
    J = torch.zeros((1, capacity, 20), dtype=torch.float32, device=engine.device)
    J[0, :nefc_val] = torch.from_numpy(npz["efc_J"].astype(np.float32))
    aref = torch.zeros((1, capacity), dtype=torch.float32, device=engine.device)
    aref[0, :nefc_val] = torch.from_numpy(npz["efc_aref"].astype(np.float32))
    R = torch.ones((1, capacity), dtype=torch.float32, device=engine.device)
    R[0, :nefc_val] = torch.from_numpy(npz["efc_R"].astype(np.float32))
    nefc = torch.tensor([nefc_val], dtype=torch.int32, device=engine.device)

    res = engine.solve_oracle_constraints(L, f_smooth, J, aref, R, nefc, capacity=capacity)
    torch.mps.synchronize()

    assert res["solver_status"].item() in (0, 1)
    err_qacc = float(np.max(np.abs(res["qacc"][0].cpu().numpy() - npz["qacc"])))
    assert err_qacc < 0.01, f"Nonzero applied force qacc error {err_qacc:.2e} >= 0.01"


def test_unsupported_constraint_types_rejection(engine):
    """Verifies that constraint types other than mjCNSTR_CONTACT_PYRAMIDAL (type 6) halt with status -5."""
    npz = np.load(CORPUS_DIR / "standing_zero_vel.npz")
    capacity = 32
    L = torch.from_numpy(npz["L"].astype(np.float32)).unsqueeze(0).to(engine.device)
    f_smooth = torch.from_numpy(npz["qfrc_smooth"].astype(np.float32)).unsqueeze(0).to(engine.device)
    J = torch.zeros((1, capacity, 20), dtype=torch.float32, device=engine.device)
    aref = torch.zeros((1, capacity), dtype=torch.float32, device=engine.device)
    R = torch.ones((1, capacity), dtype=torch.float32, device=engine.device)
    nefc = torch.tensor([4], dtype=torch.int32, device=engine.device)

    # Pass unsupported row type: 0 (equality constraint)
    efc_type = torch.full((1, capacity), 0, dtype=torch.int32, device=engine.device)

    res = engine.solve_oracle_constraints(L, f_smooth, J, aref, R, nefc, efc_type=efc_type, capacity=capacity)
    torch.mps.synchronize()

    assert res["solver_status"].item() == -5, f"Expected status -5, got {res['solver_status'].item()}"
    assert torch.all(torch.isnan(res["qacc"]))
    assert torch.all(torch.isnan(res["qfrc_constraint"]))
    assert torch.all(torch.isnan(res["lambda"]))


def test_capacity_overflow_rejection(engine):
    """Verifies that nefc > capacity is rejected on GPU with status -4."""
    npz = np.load(CORPUS_DIR / "standing_zero_vel.npz")
    capacity = 16 # capacity is 16
    L = torch.from_numpy(npz["L"].astype(np.float32)).unsqueeze(0).to(engine.device)
    f_smooth = torch.from_numpy(npz["qfrc_smooth"].astype(np.float32)).unsqueeze(0).to(engine.device)
    J = torch.zeros((1, capacity, 20), dtype=torch.float32, device=engine.device)
    aref = torch.zeros((1, capacity), dtype=torch.float32, device=engine.device)
    R = torch.ones((1, capacity), dtype=torch.float32, device=engine.device)

    # Overflow: nefc = 20 > 16
    nefc_overflow = torch.tensor([20], dtype=torch.int32, device=engine.device)

    res = engine.solve_oracle_constraints(L, f_smooth, J, aref, R, nefc_overflow, capacity=capacity)
    torch.mps.synchronize()

    assert res["solver_status"].item() == -4, f"Expected status -4, got {res['solver_status'].item()}"
    assert torch.all(torch.isnan(res["qacc"]))
    assert torch.all(torch.isnan(res["qfrc_constraint"]))
    assert torch.all(torch.isnan(res["lambda"]))


def test_nonfinite_inputs_rejection(engine):
    """Verifies that nonfinite entries or nonpositive R halt with status -1 or -2."""
    npz = np.load(CORPUS_DIR / "standing_zero_vel.npz")
    capacity = 32
    L = torch.from_numpy(npz["L"].astype(np.float32)).unsqueeze(0).to(engine.device)
    f_smooth = torch.from_numpy(npz["qfrc_smooth"].astype(np.float32)).unsqueeze(0).to(engine.device)
    J = torch.zeros((1, capacity, 20), dtype=torch.float32, device=engine.device)
    aref = torch.zeros((1, capacity), dtype=torch.float32, device=engine.device)
    R = torch.ones((1, capacity), dtype=torch.float32, device=engine.device)
    nefc = torch.tensor([4], dtype=torch.int32, device=engine.device)

    # Non-positive regularization: R = 0.0
    R_bad = torch.zeros((1, capacity), dtype=torch.float32, device=engine.device)
    res = engine.solve_oracle_constraints(L, f_smooth, J, aref, R_bad, nefc, capacity=capacity)
    torch.mps.synchronize()
    assert res["solver_status"].item() in (-1, -2)
    assert torch.all(torch.isnan(res["qacc"]))

    # NaN in f_smooth
    f_nan = f_smooth.clone()
    f_nan[0, 5] = float("nan")
    res_nan = engine.solve_oracle_constraints(L, f_nan, J, aref, R, nefc, capacity=capacity)
    torch.mps.synchronize()
    assert res_nan["solver_status"].item() == -1
    assert torch.all(torch.isnan(res_nan["qacc"]))


def test_upstream_failure_propagation(engine):
    """Verifies that upstream status != 0 propagates to solver output with NaN invalidation."""
    npz = np.load(CORPUS_DIR / "standing_zero_vel.npz")
    capacity = 32
    L = torch.from_numpy(npz["L"].astype(np.float32)).unsqueeze(0).to(engine.device)
    f_smooth = torch.from_numpy(npz["qfrc_smooth"].astype(np.float32)).unsqueeze(0).to(engine.device)
    J = torch.zeros((1, capacity, 20), dtype=torch.float32, device=engine.device)
    aref = torch.zeros((1, capacity), dtype=torch.float32, device=engine.device)
    R = torch.ones((1, capacity), dtype=torch.float32, device=engine.device)
    nefc = torch.tensor([4], dtype=torch.int32, device=engine.device)

    upstream = torch.tensor([-3], dtype=torch.int32, device=engine.device)
    res = engine.solve_oracle_constraints(L, f_smooth, J, aref, R, nefc, upstream_status=upstream, capacity=capacity)
    torch.mps.synchronize()

    assert res["solver_status"].item() == -3
    assert torch.all(torch.isnan(res["qacc"]))
    assert torch.all(torch.isnan(res["qfrc_constraint"]))
    assert torch.all(torch.isnan(res["lambda"]))


def test_heterogeneous_mixed_batch_execution(engine):
    """Concurrently solves a heterogeneous batch: airborne (0), single support (12), standing (24)."""
    npz_air = np.load(CORPUS_DIR / "airborne.npz")
    npz_single = np.load(CORPUS_DIR / "single_support_zero_vel.npz")
    npz_stand = np.load(CORPUS_DIR / "standing_zero_vel.npz")
    capacity = 32

    # Single-world reference runs
    res_singles = []
    for npz in [npz_air, npz_single, npz_stand]:
        n_val = int(npz["nefc"])
        L = torch.from_numpy(npz["L"].astype(np.float32)).unsqueeze(0).to(engine.device)
        f = torch.from_numpy(npz["qfrc_smooth"].astype(np.float32)).unsqueeze(0).to(engine.device)
        J = torch.zeros((1, capacity, 20), dtype=torch.float32, device=engine.device)
        aref = torch.zeros((1, capacity), dtype=torch.float32, device=engine.device)
        R = torch.ones((1, capacity), dtype=torch.float32, device=engine.device)
        efc_type = torch.full((1, capacity), 6, dtype=torch.int32, device=engine.device)
        if n_val > 0:
            J[0, :n_val] = torch.from_numpy(npz["efc_J"].astype(np.float32))
            aref[0, :n_val] = torch.from_numpy(npz["efc_aref"].astype(np.float32))
            R[0, :n_val] = torch.from_numpy(npz["efc_R"].astype(np.float32))
            efc_type[0, :n_val] = torch.from_numpy(npz["efc_type"].astype(np.int32))
        nefc = torch.tensor([n_val], dtype=torch.int32, device=engine.device)
        r = engine.solve_oracle_constraints(L, f, J, aref, R, nefc, efc_type=efc_type, capacity=capacity)
        torch.mps.synchronize()
        res_singles.append({
            "qacc": r["qacc"].clone(),
            "qfrc": r["qfrc_constraint"].clone(),
            "status": r["solver_status"].clone(),
            "lam": r["lambda"].clone(),
        })

    # Batch of 3
    L_b = torch.cat([
        torch.from_numpy(npz_air["L"].astype(np.float32)).unsqueeze(0),
        torch.from_numpy(npz_single["L"].astype(np.float32)).unsqueeze(0),
        torch.from_numpy(npz_stand["L"].astype(np.float32)).unsqueeze(0),
    ], dim=0).to(engine.device)

    f_b = torch.cat([
        torch.from_numpy(npz_air["qfrc_smooth"].astype(np.float32)).unsqueeze(0),
        torch.from_numpy(npz_single["qfrc_smooth"].astype(np.float32)).unsqueeze(0),
        torch.from_numpy(npz_stand["qfrc_smooth"].astype(np.float32)).unsqueeze(0),
    ], dim=0).to(engine.device)

    J_b = torch.zeros((3, capacity, 20), dtype=torch.float32, device=engine.device)
    aref_b = torch.zeros((3, capacity), dtype=torch.float32, device=engine.device)
    R_b = torch.ones((3, capacity), dtype=torch.float32, device=engine.device)
    efc_type_b = torch.full((3, capacity), 6, dtype=torch.int32, device=engine.device)

    n_s = int(npz_single["nefc"])
    n_st = int(npz_stand["nefc"])
    J_b[1, :n_s] = torch.from_numpy(npz_single["efc_J"].astype(np.float32))
    aref_b[1, :n_s] = torch.from_numpy(npz_single["efc_aref"].astype(np.float32))
    R_b[1, :n_s] = torch.from_numpy(npz_single["efc_R"].astype(np.float32))
    efc_type_b[1, :n_s] = torch.from_numpy(npz_single["efc_type"].astype(np.int32))

    J_b[2, :n_st] = torch.from_numpy(npz_stand["efc_J"].astype(np.float32))
    aref_b[2, :n_st] = torch.from_numpy(npz_stand["efc_aref"].astype(np.float32))
    R_b[2, :n_st] = torch.from_numpy(npz_stand["efc_R"].astype(np.float32))
    efc_type_b[2, :n_st] = torch.from_numpy(npz_stand["efc_type"].astype(np.int32))

    nefc_b = torch.tensor([0, n_s, n_st], dtype=torch.int32, device=engine.device)

    res_batch = engine.solve_oracle_constraints(L_b, f_b, J_b, aref_b, R_b, nefc_b, efc_type=efc_type_b, capacity=capacity)
    torch.mps.synchronize()

    # Verify each world in the batch matches single-world run
    for i in range(3):
        diff_qacc = float(torch.max(torch.abs(res_batch["qacc"][i] - res_singles[i]["qacc"][0])))
        diff_qfrc = float(torch.max(torch.abs(res_batch["qfrc_constraint"][i] - res_singles[i]["qfrc"][0])))
        diff_lam = float(torch.max(torch.abs(res_batch["lambda"][i] - res_singles[i]["lam"][0])))
        assert diff_qacc < 1e-6, f"World {i} qacc mismatch {diff_qacc:.2e}"
        assert diff_qfrc < 1e-6, f"World {i} qfrc mismatch {diff_qfrc:.2e}"
        assert diff_lam < 1e-6, f"World {i} lambda mismatch {diff_lam:.2e}"
        assert res_batch["solver_status"][i].item() == res_singles[i]["status"].item()
