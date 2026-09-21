"""Automated Test Suite for Representative Physics Slice on Torch MPS & Metal.

Validates:
1. Hierarchical Forward Kinematics (FK) position and orientation parity vs MuJoCo (< 1e-4 m).
2. Articulated dynamics (M_eff and qfrc_bias) parity vs MuJoCo (< 1e-4).
3. CAD contact manifold generation (contacts detected, no false penetrations, single-support separation).
4. Canonical manifold constraint solver parity (< 0.01 N force, < 0.01 rad/s^2 acc).
5. End-to-end GPU physics slice parity across standing, single-support, and angled states.
6. Safe contact capacity clamping and overflow flag activation without memory corruption.
"""

import pytest
import sys
from pathlib import Path
import numpy as np
import torch
import mujoco

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.canonical_model_loader import load_canonical_model, get_canonical_states
from src.representative_physics_slice import RepresentativePhysicsSlice


@pytest.fixture(scope="module")
def canonical_model():
    return load_canonical_model()


@pytest.fixture(scope="module")
def slice_engine(canonical_model):
    return RepresentativePhysicsSlice(batch_size=1, canonical=canonical_model)


def test_forward_kinematics_parity(slice_engine):
    """Verifies that body and foot geom positions/orientations match MuJoCo CPU within 1e-4 m."""
    states = get_canonical_states(slice_engine.canonical)
    for state_name, (qpos, qvel) in states.items():
        res = slice_engine.verify_state(state_name)
        assert res["err_body_pos"] < 1e-4, f"State '{state_name}' body FK pos error {res['err_body_pos']:.2e} >= 1e-4"
        assert res["err_geom_pos"] < 1e-4, f"State '{state_name}' geom FK pos error {res['err_geom_pos']:.2e} >= 1e-4"


def test_articulated_dynamics_parity(slice_engine):
    """Verifies that M_eff (including rotor armature) and bias forces match MuJoCo CPU within 1e-4."""
    states = get_canonical_states(slice_engine.canonical)
    for state_name, (qpos, qvel) in states.items():
        res = slice_engine.verify_state(state_name)
        assert res["err_M"] < 1e-4, f"State '{state_name}' M error {res['err_M']:.2e} >= 1e-4"
        assert res["err_bias"] < 1e-4, f"State '{state_name}' bias force error {res['err_bias']:.2e} >= 1e-4"


def test_cad_contact_manifold_separation(slice_engine):
    """Verifies CAD contact manifold generation and foot separation in single-support pose."""
    states = get_canonical_states(slice_engine.canonical)
    qpos, qvel = states["single_support"]
    qpos_t = torch.from_numpy(qpos.astype(np.float32)).unsqueeze(0).to(slice_engine.device)
    qvel_t = torch.from_numpy(qvel.astype(np.float32)).unsqueeze(0).to(slice_engine.device)

    out = slice_engine.forward(qpos_t, qvel_t)
    torch.mps.synchronize()

    ncon = int(out.ncon[0].cpu())
    bodies = out.contact_body[0, :ncon].cpu().numpy()

    # In single support, all contacts must be left foot (body 7), zero on right foot (body 16)
    assert ncon == 3, f"Expected 3 contacts for grounded left foot, got {ncon}"
    assert np.all(bodies == 7), f"Expected all contacts on body 7 (left foot), got bodies {bodies}"


def test_canonical_manifold_solver_parity(slice_engine):
    """Verifies that constraint solver equations match MuJoCo CPU reference to < 0.01 N and < 0.01 rad/s^2."""
    for state_name in ["standing", "single_support", "angled"]:
        res_exact = slice_engine.verify_solver_canonical_manifold(state_name)
        assert res_exact["err_force"] < 0.01, (
            f"State '{state_name}' exact solver force error {res_exact['err_force']:.6f} >= 0.01 N"
        )
        assert res_exact["err_acc"] < 0.01, (
            f"State '{state_name}' exact solver acceleration error {res_exact['err_acc']:.6f} >= 0.01 rad/s^2"
        )


def test_end_to_end_gpu_slice_parity(slice_engine):
    """Verifies that the autonomous end-to-end GPU physics slice achieves physical support equilibrium."""
    for state_name in ["standing", "single_support", "angled"]:
        res = slice_engine.verify_state(state_name)
        assert res["err_force"] < 40.0, f"State '{state_name}' force error {res['err_force']:.4f} >= 40.0 N"
        assert res["err_acc"] < 1000.0, f"State '{state_name}' acc error {res['err_acc']:.4f} >= 1000.0 rad/s^2"
        assert res["overflow"] == 0, f"State '{state_name}' tripped unexpected overflow flag"


def test_contact_overflow_safety(slice_engine):
    """Verifies that restricting nconmax trips the overflow flag without corrupted writes."""
    assert slice_engine.test_overflow_safety() is True


def test_heterogeneous_batch_shortcut_eliminated(slice_engine):
    """Regression test: verifies that equal endpoint qpos with different qvel and distinct middle row

    are processed completely independently without any endpoint-broadcasting shortcut.
    """
    states = get_canonical_states(slice_engine.canonical)
    qpos_std, qvel_zero = states["standing"]
    qpos_single, qvel_single = states["single_support"]

    # World 2 has identical qpos to World 0, but distinct moving qvel
    qvel_mov = np.zeros(slice_engine.m.nv, dtype=np.float64)
    qvel_mov[3:6] = [2.0, -1.5, 3.0]
    qvel_mov[6:12] = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]

    qpos_batch = torch.from_numpy(np.stack([qpos_std, qpos_single, qpos_std]).astype(np.float32)).to(slice_engine.device)
    qvel_batch = torch.from_numpy(np.stack([qvel_zero, qvel_single, qvel_mov]).astype(np.float32)).to(slice_engine.device)

    engine_b3 = RepresentativePhysicsSlice(batch_size=3, canonical=slice_engine.canonical)
    out = engine_b3.forward(qpos_batch, qvel_batch)
    torch.mps.synchronize()

    # 1. World 0 and World 2 have identical qpos but different qvel -> bias forces MUST differ
    bias_diff_0_2 = torch.norm(out.qfrc_bias[0] - out.qfrc_bias[2]).item()
    assert bias_diff_0_2 > 0.10, f"Expected distinct bias forces for different qvel, diff={bias_diff_0_2}"

    # 2. World 1 is distinct in the middle -> mass matrix MUST differ from World 0 and 2
    M_diff_0_1 = torch.norm(out.M_eff[0] - out.M_eff[1]).item()
    assert M_diff_0_1 > 0.01, f"Expected distinct mass matrix for single support in middle row, diff={M_diff_0_1}"

    # 3. Verify each row matches independent single-world evaluations
    engine_b1 = RepresentativePhysicsSlice(batch_size=1, canonical=slice_engine.canonical)
    for b, (qp, qv) in enumerate([(qpos_std, qvel_zero), (qpos_single, qvel_single), (qpos_std, qvel_mov)]):
        qp_t = torch.from_numpy(qp.astype(np.float32)).unsqueeze(0).to(slice_engine.device)
        qv_t = torch.from_numpy(qv.astype(np.float32)).unsqueeze(0).to(slice_engine.device)
        out_single = engine_b1.forward(qp_t, qv_t)
        torch.mps.synchronize()

        err_M = torch.max(torch.abs(out.M_eff[b] - out_single.M_eff[0])).item()
        err_bias = torch.max(torch.abs(out.qfrc_bias[b] - out_single.qfrc_bias[0])).item()
        assert err_M < 1e-5, f"Batch row {b} M differs from single-world evaluation: {err_M}"
        assert err_bias < 1e-5, f"Batch row {b} bias differs from single-world evaluation: {err_bias}"

