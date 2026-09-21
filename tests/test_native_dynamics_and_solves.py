"""Automated Test Suite for Native Metal Dynamics (CRBA/RNE) and Native Cholesky Solves.

Verifies:
1. Native forward kinematics position and orientation parity vs CPU oracle corpus (< 1e-6).
2. Native CRBA generalized mass matrix (M_eff) parity vs CPU oracle corpus (< 1e-6).
3. Native RNE Coriolis/centrifugal/gravity bias force parity vs CPU oracle corpus (< 1e-5).
4. Native Cholesky factorization accuracy ||L L^T - M|| (< 1e-6).
5. Native multi-RHS linear solves relative residual ||M X - B|| / (||M|| ||X|| + ||B||) (< 1e-6).
6. Deterministic detection of non-positive pivots without silent clamping (status = -1).
7. Heterogeneous batch domain randomization independence on GPU.
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
]


@pytest.mark.parametrize("scenario", ALL_SCENARIOS)
def test_oracle_kinematics_and_dynamics_parity(engine, canonical, scenario):
    """Verifies that native GPU FK, M_eff, and qfrc_bias match the independent CPU oracle."""
    npz = np.load(CORPUS_DIR / f"{scenario}.npz")
    qpos = torch.from_numpy(npz["qpos"].astype(np.float32)).unsqueeze(0).to(engine.device)
    qvel = torch.from_numpy(npz["qvel"].astype(np.float32)).unsqueeze(0).to(engine.device)

    p_mass = None
    p_ipos = None
    p_arm = None
    if "randomized" in scenario:
        base_id = 2
        m_mass = np.copy(canonical.body_mass)
        m_mass[base_id] *= 1.05
        m_ipos = np.copy(canonical.model.body_ipos)
        m_ipos[base_id] += np.array([0.005, -0.004, 0.003])
        m_arm = np.copy(canonical.dof_armature)
        m_arm[6:] *= 1.10
        p_mass = torch.from_numpy(m_mass.astype(np.float32)).unsqueeze(0).to(engine.device)
        p_ipos = torch.from_numpy(m_ipos.astype(np.float32)).unsqueeze(0).to(engine.device)
        p_arm = torch.from_numpy(m_arm.astype(np.float32)).unsqueeze(0).to(engine.device)

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
def test_native_cholesky_factorization_and_solve(engine, canonical, scenario):
    """Verifies that native GPU Cholesky factorization and linear solves achieve < 1e-6 relative residual."""
    npz = np.load(CORPUS_DIR / f"{scenario}.npz")
    qpos = torch.from_numpy(npz["qpos"].astype(np.float32)).unsqueeze(0).to(engine.device)
    qvel = torch.from_numpy(npz["qvel"].astype(np.float32)).unsqueeze(0).to(engine.device)

    p_mass = None
    p_ipos = None
    p_arm = None
    if "randomized" in scenario:
        base_id = 2
        m_mass = np.copy(canonical.body_mass)
        m_mass[base_id] *= 1.05
        m_ipos = np.copy(canonical.model.body_ipos)
        m_ipos[base_id] += np.array([0.005, -0.004, 0.003])
        m_arm = np.copy(canonical.dof_armature)
        m_arm[6:] *= 1.10
        p_mass = torch.from_numpy(m_mass.astype(np.float32)).unsqueeze(0).to(engine.device)
        p_ipos = torch.from_numpy(m_ipos.astype(np.float32)).unsqueeze(0).to(engine.device)
        p_arm = torch.from_numpy(m_arm.astype(np.float32)).unsqueeze(0).to(engine.device)

    engine.compute_native_dynamics(qpos, qvel, per_world_mass=p_mass, per_world_ipos=p_ipos, per_world_armature=p_arm)
    X_bias, status = engine.compute_native_cholesky_solve(engine.M_eff, engine.qfrc_bias)
    torch.mps.synchronize()

    assert status.item() == 0, f"[{scenario}] Cholesky factorization failed with status {status.item()}"

    L_gpu = engine.L_factor[0].cpu().numpy()
    M_gpu = engine.M_eff[0].cpu().numpy()
    X_gpu = X_bias[0].cpu().numpy()
    bias_gpu = engine.qfrc_bias[0].cpu().numpy()

    # Factorization accuracy
    res_L = float(np.max(np.abs(L_gpu @ L_gpu.T - M_gpu)))
    assert res_L < 1e-6, f"[{scenario}] Cholesky reconstruction error {res_L:.2e} >= 1e-6"

    # Linear solve relative residual: ||M X - B|| / (||M|| ||X|| + ||B||)
    b_norm = np.linalg.norm(bias_gpu)
    denom = np.linalg.norm(M_gpu) * np.linalg.norm(X_gpu) + b_norm
    rel_residual = float(np.linalg.norm(M_gpu @ X_gpu - bias_gpu[:, None]) / denom) if denom > 0 else 0.0
    assert rel_residual < 1e-6, f"[{scenario}] Solve relative residual {rel_residual:.2e} >= 1e-6"


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
    assert rel_residual < 1e-6, f"Multi-RHS relative residual {rel_residual:.2e} >= 1e-6"


def test_native_cholesky_non_positive_pivot_rejection(engine):
    """Verifies that non-positive pivots or non-finite entries trigger status = -1 without silent clamping."""
    M_bad = torch.eye(20, dtype=torch.float32, device=engine.device)
    M_bad[4, 4] = -1.0  # negative eigenvalue
    B = torch.ones((1, 20), dtype=torch.float32, device=engine.device)

    X, status = engine.compute_native_cholesky_solve(M_bad.unsqueeze(0), B)
    torch.mps.synchronize()

    assert status.item() == -1, f"Expected status -1 for non-SPD matrix, got {status.item()}"


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
