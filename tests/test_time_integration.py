"""Tests for Milestone 4: Canonical 5 ms ImplicitFast Time Integration and Trajectory Qualification.

Verifies:
1. Isolated ImplicitFast integrator parity against analytical and CPU MuJoCo references.
2. Common-state one-step parity across all 25 corpus scenarios using separated metric gates.
3. Free-running CPU and Metal trajectories across 4 steps (20 ms), 20 steps (100 ms), and 200 steps (1.0 s).
4. Actuator torque transmission and the four-substep control interval.
5. Multi-world reset, failure isolation, and error invalidation.
"""

import math
import sys
from pathlib import Path
from typing import Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mujoco
import numpy as np
import pytest
import torch

from src.canonical_model_loader import (
    CanonicalMicroDuckModel,
    create_matching_cpu_data,
    create_matching_cpu_model,
    load_canonical_model,
)
from src.representative_physics_slice import RepresentativePhysicsSlice

SCENARIOS_ALL_25 = sorted([f.stem for f in (PROJECT_ROOT / "corpus").glob("*.npz")])
assert len(SCENARIOS_ALL_25) == 25, f"Expected 25 scenarios, found {len(SCENARIOS_ALL_25)}"


def compute_so3_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    """Computes geodesic angular distance on SO(3) invariant to q ~ -q."""
    q1_64 = q1.astype(np.float64)
    q2_64 = q2.astype(np.float64)
    q1_norm = q1_64 / max(1e-14, float(np.linalg.norm(q1_64)))
    q2_norm = q2_64 / max(1e-14, float(np.linalg.norm(q2_64)))
    dot = float(np.abs(np.dot(q1_norm, q2_norm)))
    dot_clamped = min(1.0, max(0.0, dot))
    return float(2.0 * np.arccos(dot_clamped))


def compute_separated_metrics(qp_gpu: np.ndarray, qv_gpu: np.ndarray, qp_cpu: np.ndarray, qv_cpu: np.ndarray) -> Dict[str, float]:
    """Computes non-pooled, separated metric differences between candidate and reference."""
    pos_err = float(np.max(np.abs(qp_gpu[0:3] - qp_cpu[0:3])))
    so3_err = compute_so3_distance(qp_gpu[3:7], qp_cpu[3:7])
    linvel_err = float(np.max(np.abs(qv_gpu[0:3] - qv_cpu[0:3])))
    angvel_err = float(np.max(np.abs(qv_gpu[3:6] - qv_cpu[3:6])))
    jnt_pos_err = float(np.max(np.abs(qp_gpu[7:21] - qp_cpu[7:21])))
    jnt_vel_err = float(np.max(np.abs(qv_gpu[6:20] - qv_cpu[6:20])))
    return {
        "pos_err": pos_err,
        "so3_err": so3_err,
        "linvel_err": linvel_err,
        "angvel_err": angvel_err,
        "jnt_pos_err": jnt_pos_err,
        "jnt_vel_err": jnt_vel_err,
    }


# ==============================================================================
# Stage 1: Isolated Integrator Parity
# ==============================================================================

def test_implicit_fast_pure_linear_translation():
    """Verifies pure translation integration v_{t+h} = v_t + dt * a, p_{t+h} = p_t + dt * v_{t+h}."""
    ps = RepresentativePhysicsSlice(batch_size=2)
    dt = 0.005

    qp = torch.zeros((2, 21), dtype=torch.float32, device=ps.device)
    qp[:, 3] = 1.0  # unit quaternion w=1
    qv = torch.zeros((2, 20), dtype=torch.float32, device=ps.device)
    qa = torch.zeros((2, 20), dtype=torch.float32, device=ps.device)
    status = torch.zeros((2,), dtype=torch.int32, device=ps.device)

    # World 0: v0 = [1, 2, 3], a = [0, 0, -9.81]
    qv[0, 0:3] = torch.tensor([1.0, 2.0, 3.0], device=ps.device)
    qa[0, 0:3] = torch.tensor([0.0, 0.0, -9.81], device=ps.device)

    # World 1: v0 = [-0.5, 0.5, 0.0], a = [10.0, -5.0, 2.0]
    qv[1, 0:3] = torch.tensor([-0.5, 0.5, 0.0], device=ps.device)
    qa[1, 0:3] = torch.tensor([10.0, -5.0, 2.0], device=ps.device)

    qp_out, qv_out, stat_out = ps.integrate_implicit_fast(qp, qv, qa, status, dt=dt)
    torch.mps.synchronize()

    assert torch.all(stat_out == 0)

    # Analytical checks
    for w in range(2):
        v_next = qv[w, 0:3].cpu().numpy() + dt * qa[w, 0:3].cpu().numpy()
        p_next = qp[w, 0:3].cpu().numpy() + dt * v_next
        assert np.allclose(qv_out[w, 0:3].cpu().numpy(), v_next, atol=1e-6)
        assert np.allclose(qp_out[w, 0:3].cpu().numpy(), p_next, atol=1e-6)


def test_implicit_fast_quaternion_rotation():
    """Verifies quaternion rotation q_{t+h} = normalize(q_t * dq(omega_{t+h} * dt))."""
    ps = RepresentativePhysicsSlice(batch_size=4)
    dt = 0.005

    qp = torch.zeros((4, 21), dtype=torch.float32, device=ps.device)
    qv = torch.zeros((4, 20), dtype=torch.float32, device=ps.device)
    qa = torch.zeros((4, 20), dtype=torch.float32, device=ps.device)
    status = torch.zeros((4,), dtype=torch.int32, device=ps.device)

    # World 0: pure X rotation, omega = 10.0 rad/s
    qp[0, 3] = 1.0
    qv[0, 3] = 10.0

    # World 1: pure Y rotation, omega = -5.0 rad/s, non-identity initial quat (45 deg around Z)
    s45 = math.sin(math.pi / 8)
    c45 = math.cos(math.pi / 8)
    qp[1, 3] = c45
    qp[1, 6] = s45
    qv[1, 4] = -5.0

    # World 2: small-angle branch (omega = 1e-13 rad/s)
    qp[2, 3] = 1.0
    qv[2, 3:6] = 1e-13

    # World 3: arbitrary oblique axis [1, 2, 3] / sqrt(14) with angular acceleration
    axis = np.array([1.0, 2.0, 3.0]) / math.sqrt(14.0)
    qp[3, 3] = 0.5
    qp[3, 4:7] = torch.tensor(axis * math.sqrt(0.75), dtype=torch.float32, device=ps.device)
    qv[3, 3:6] = torch.tensor(axis * 2.0, dtype=torch.float32, device=ps.device)
    qa[3, 3:6] = torch.tensor(axis * 4.0, dtype=torch.float32, device=ps.device)

    qp_out, qv_out, stat_out = ps.integrate_implicit_fast(qp, qv, qa, status, dt=dt)
    torch.mps.synchronize()

    assert torch.all(stat_out == 0)

    # Validate against MuJoCo CPU mju_quatIntegrate
    for w in range(4):
        q_cpu = qp[w, 3:7].cpu().numpy().astype(np.float64)
        v_next_cpu = qv[w, 3:6].cpu().numpy().astype(np.float64) + dt * qa[w, 3:6].cpu().numpy().astype(np.float64)
        q_adv = q_cpu.copy()
        mujoco.mju_quatIntegrate(q_adv, v_next_cpu, dt)
        mujoco.mju_normalize4(q_adv)

        q_gpu = qp_out[w, 3:7].cpu().numpy()
        so3_dist = compute_so3_distance(q_gpu, q_adv)
        assert so3_dist < 1e-6, f"World {w} SO(3) orientation error {so3_dist:.2e} >= 1e-6"


def test_implicit_fast_quaternion_antipodal_invariance():
    """Verifies that q and -q advance to identical physical orientations."""
    ps = RepresentativePhysicsSlice(batch_size=2)
    dt = 0.005

    qp = torch.zeros((2, 21), dtype=torch.float32, device=ps.device)
    qv = torch.zeros((2, 20), dtype=torch.float32, device=ps.device)
    qa = torch.zeros((2, 20), dtype=torch.float32, device=ps.device)
    status = torch.zeros((2,), dtype=torch.int32, device=ps.device)

    # Base quaternion: 30 deg rotation around [0, 1, 0]
    q_base = np.array([math.cos(math.pi / 12), 0.0, math.sin(math.pi / 12), 0.0])
    qp[0, 3:7] = torch.from_numpy(q_base.astype(np.float32)).to(ps.device)
    qp[1, 3:7] = torch.from_numpy((-q_base).astype(np.float32)).to(ps.device)

    qv[:, 3:6] = torch.tensor([1.5, -2.0, 3.0], device=ps.device)

    qp_out, _, stat_out = ps.integrate_implicit_fast(qp, qv, qa, status, dt=dt)
    torch.mps.synchronize()

    q0 = qp_out[0, 3:7].cpu().numpy()
    q1 = qp_out[1, 3:7].cpu().numpy()
    so3_dist = compute_so3_distance(q0, q1)
    assert so3_dist < 1e-6, f"Antipodal quat SO(3) distance {so3_dist:.2e} >= 1e-6"


def test_implicit_fast_joint_motion():
    """Verifies that all 14 hinge joints advance via qpos = qpos + dt * v_next."""
    ps = RepresentativePhysicsSlice(batch_size=1)
    dt = 0.005

    qp = torch.zeros((1, 21), dtype=torch.float32, device=ps.device)
    qp[0, 3] = 1.0
    qv = torch.zeros((1, 20), dtype=torch.float32, device=ps.device)
    qa = torch.zeros((1, 20), dtype=torch.float32, device=ps.device)
    status = torch.zeros((1,), dtype=torch.int32, device=ps.device)

    # Joint positions, velocities, accelerations
    jnt_init = np.linspace(-1.0, 1.0, 14, dtype=np.float32)
    jnt_vel = np.linspace(2.0, -2.0, 14, dtype=np.float32)
    jnt_acc = np.linspace(-5.0, 5.0, 14, dtype=np.float32)

    qp[0, 7:21] = torch.from_numpy(jnt_init).to(ps.device)
    qv[0, 6:20] = torch.from_numpy(jnt_vel).to(ps.device)
    qa[0, 6:20] = torch.from_numpy(jnt_acc).to(ps.device)

    qp_out, qv_out, stat_out = ps.integrate_implicit_fast(qp, qv, qa, status, dt=dt)
    torch.mps.synchronize()

    assert stat_out[0].item() == 0

    expected_vel = jnt_vel + dt * jnt_acc
    expected_pos = jnt_init + dt * expected_vel

    assert np.allclose(qv_out[0, 6:20].cpu().numpy(), expected_vel, atol=1e-6)
    assert np.allclose(qp_out[0, 7:21].cpu().numpy(), expected_pos, atol=1e-6)


def test_integrator_failure_invalidation():
    """Verifies that negative upstream solver/physics status sets error code and fills outputs with NAN."""
    ps = RepresentativePhysicsSlice(batch_size=3)

    qp = torch.zeros((3, 21), dtype=torch.float32, device=ps.device)
    qp[:, 3] = 1.0
    qv = torch.zeros((3, 20), dtype=torch.float32, device=ps.device)
    qa = torch.zeros((3, 20), dtype=torch.float32, device=ps.device)

    # Negative upstream statuses: Cholesky failure (-1), solver singular (-3), contact overflow (-6)
    status = torch.tensor([-1, -3, -6], dtype=torch.int32, device=ps.device)

    qp_out, qv_out, stat_out = ps.integrate_implicit_fast(qp, qv, qa, status, dt=0.005)
    torch.mps.synchronize()

    for w, s in enumerate([-1, -3, -6]):
        assert stat_out[w].item() == s
        assert torch.all(torch.isnan(qp_out[w]))
        assert torch.all(torch.isnan(qv_out[w]))


def test_integrator_non_finite_input_guards():
    """Verifies that non-finite qpos, qvel, qacc or invalid dt set error code -1 and outputs to NAN."""
    ps = RepresentativePhysicsSlice(batch_size=4)

    qp = torch.zeros((4, 21), dtype=torch.float32, device=ps.device)
    qp[:, 3] = 1.0
    qv = torch.zeros((4, 20), dtype=torch.float32, device=ps.device)
    qa = torch.zeros((4, 20), dtype=torch.float32, device=ps.device)
    status = torch.zeros((4,), dtype=torch.int32, device=ps.device)

    # World 0: NAN in qpos
    qp[0, 2] = float("nan")
    # World 1: INF in qvel
    qv[1, 0] = float("inf")
    # World 2: NAN in qacc
    qa[2, 6] = float("nan")

    qp_out, qv_out, stat_out = ps.integrate_implicit_fast(qp, qv, qa, status, dt=0.005)
    torch.mps.synchronize()

    for w in range(3):
        assert stat_out[w].item() == -1
        assert torch.all(torch.isnan(qp_out[w]))
        assert torch.all(torch.isnan(qv_out[w]))


def test_integrator_in_place_safety():
    """Verifies that in-place updates (qpos_out == qpos_in, qvel_out == qvel_in) advance correctly."""
    ps = RepresentativePhysicsSlice(batch_size=1)
    dt = 0.005

    qp = torch.zeros((1, 21), dtype=torch.float32, device=ps.device)
    qp[0, 0:3] = torch.tensor([1.0, 2.0, 3.0], device=ps.device)
    qp[0, 3] = 1.0
    qv = torch.zeros((1, 20), dtype=torch.float32, device=ps.device)
    qv[0, 0:3] = torch.tensor([0.5, -0.5, 1.0], device=ps.device)
    qa = torch.zeros((1, 20), dtype=torch.float32, device=ps.device)
    qa[0, 0:3] = torch.tensor([0.0, 0.0, -9.81], device=ps.device)
    status = torch.zeros((1,), dtype=torch.int32, device=ps.device)

    # Perform in-place integration
    ps.integrate_implicit_fast(qp, qv, qa, status, dt=dt, qpos_out=qp, qvel_out=qv)
    torch.mps.synchronize()

    expected_v = np.array([0.5, -0.5, 1.0 - 0.005 * 9.81])
    expected_p = np.array([1.0, 2.0, 3.0]) + dt * expected_v

    assert np.allclose(qv[0, 0:3].cpu().numpy(), expected_v, atol=1e-6)
    assert np.allclose(qp[0, 0:3].cpu().numpy(), expected_p, atol=1e-6)


def test_integrator_caller_output_buffer_validation():
    """Verifies that caller-provided output buffers and upstream_status are strictly validated."""
    ps = RepresentativePhysicsSlice(batch_size=2)
    dt = 0.005

    qp = torch.zeros((2, 21), dtype=torch.float32, device=ps.device)
    qp[:, 3] = 1.0
    qv = torch.zeros((2, 20), dtype=torch.float32, device=ps.device)
    qa = torch.zeros((2, 20), dtype=torch.float32, device=ps.device)
    stat = torch.zeros((2,), dtype=torch.int32, device=ps.device)

    # 1. Invalid qpos_out shape
    with pytest.raises(ValueError, match="Expected qpos_out shape"):
        bad_qp = torch.zeros((2, 10), dtype=torch.float32, device=ps.device)
        ps.integrate_implicit_fast(qp, qv, qa, stat, qpos_out=bad_qp)

    # 2. Invalid qvel_out shape
    with pytest.raises(ValueError, match="Expected qvel_out shape"):
        bad_qv = torch.zeros((2, 10), dtype=torch.float32, device=ps.device)
        ps.integrate_implicit_fast(qp, qv, qa, stat, qvel_out=bad_qv)

    # 3. Invalid status_out shape
    with pytest.raises(ValueError, match="Expected status_out shape"):
        bad_stat = torch.zeros((2, 2), dtype=torch.int32, device=ps.device)
        ps.integrate_implicit_fast(qp, qv, qa, stat, status_out=bad_stat)

    # 4. Invalid dtypes (MPS supports float32, float16, int32, int64)
    with pytest.raises(TypeError, match="qpos_out must be float32"):
        bad_qp_dtype = torch.zeros((2, 21), dtype=torch.float16, device=ps.device)
        ps.integrate_implicit_fast(qp, qv, qa, stat, qpos_out=bad_qp_dtype)

    with pytest.raises(TypeError, match="status_out must be int32"):
        bad_stat_dtype = torch.zeros((2,), dtype=torch.int64, device=ps.device)
        ps.integrate_implicit_fast(qp, qv, qa, stat, status_out=bad_stat_dtype)

    # 5. Invalid device (CPU tensor passed when device is MPS)
    with pytest.raises(ValueError, match="must be on device"):
        bad_dev = torch.zeros((2, 21), dtype=torch.float32, device="cpu")
        ps.integrate_implicit_fast(qp, qv, qa, stat, qpos_out=bad_dev)

    # 5. Non-contiguous buffer
    with pytest.raises(ValueError, match="must be contiguous"):
        non_contig = torch.zeros((2, 42), dtype=torch.float32, device=ps.device)[:, ::2]
        ps.integrate_implicit_fast(qp, qv, qa, stat, qpos_out=non_contig)

    # 6. Upstream status shape (B, 2) rejection
    with pytest.raises(ValueError, match="Expected upstream_status shape"):
        bad_up_stat = torch.zeros((2, 2), dtype=torch.int32, device=ps.device)
        ps.integrate_implicit_fast(qp, qv, qa, bad_up_stat)


def test_integrator_storage_aliasing_safety():
    """Verifies that exact in-place updates are permitted while unsafe overlapping views are rejected."""
    ps = RepresentativePhysicsSlice(batch_size=2)
    dt = 0.005

    qp = torch.zeros((2, 21), dtype=torch.float32, device=ps.device)
    qp[:, 3] = 1.0
    qv = torch.zeros((2, 20), dtype=torch.float32, device=ps.device)
    qa = torch.zeros((2, 20), dtype=torch.float32, device=ps.device)
    stat = torch.zeros((2,), dtype=torch.int32, device=ps.device)

    # Exact in-place must succeed
    qp_ret, qv_ret, stat_ret = ps.integrate_implicit_fast(qp, qv, qa, stat, dt=dt, qpos_out=qp, qvel_out=qv, status_out=stat)
    assert qp_ret.data_ptr() == qp.data_ptr()
    assert qv_ret.data_ptr() == qv.data_ptr()

    # Partial / shifted memory overlap must be rejected
    flat = torch.zeros(100, dtype=torch.float32, device=ps.device)
    qp1 = flat[0:21].unsqueeze(0)
    qp1[:, 3] = 1.0
    qp2 = flat[1:22].unsqueeze(0)  # Overlaps with qp1 by 20 elements
    qv1 = torch.zeros((1, 20), dtype=torch.float32, device=ps.device)
    qa1 = torch.zeros((1, 20), dtype=torch.float32, device=ps.device)
    stat1 = torch.zeros((1,), dtype=torch.int32, device=ps.device)

    with pytest.raises(ValueError, match="Unsafe memory aliasing / partial overlap"):
        ps.integrate_implicit_fast(qp1, qv1, qa1, stat1, qpos_out=qp2)

    # Cross-buffer aliasing between qpos_out and qvel_out must be rejected
    out_buf = torch.zeros((1, 21), dtype=torch.float32, device=ps.device)
    with pytest.raises(ValueError, match="Unsafe memory aliasing / partial overlap"):
        ps.integrate_implicit_fast(qp1, qv1, qa1, stat1, qpos_out=out_buf, qvel_out=out_buf[:, :20])


def test_integrator_state_overflow_and_neighbor_isolation():
    """Verifies that arithmetic overflow produces status -2 with NANs, isolating neighboring worlds."""
    ps = RepresentativePhysicsSlice(batch_size=2)
    dt = 0.005

    qp = torch.zeros((2, 21), dtype=torch.float32, device=ps.device)
    qp[:, 3] = 1.0
    qv = torch.zeros((2, 20), dtype=torch.float32, device=ps.device)
    qa = torch.zeros((2, 20), dtype=torch.float32, device=ps.device)
    stat = torch.zeros((2,), dtype=torch.int32, device=ps.device)

    # World 0: valid normal linear translation
    qv[0, 0:3] = torch.tensor([1.0, 2.0, 3.0], device=ps.device)
    qa[0, 0:3] = torch.tensor([0.0, 0.0, -9.81], device=ps.device)

    # World 1: finite float32 max velocity and acceleration -> overflow on addition
    qv[1] = 3.4e38
    qa[1] = 3.4e38

    qp_out, qv_out, stat_out = ps.integrate_implicit_fast(qp, qv, qa, stat, dt=dt)
    torch.mps.synchronize()

    # World 0 must advance correctly with status 0
    assert stat_out[0].item() == 0
    expected_v0 = np.array([1.0, 2.0, 3.0 - dt * 9.81])
    expected_p0 = dt * expected_v0
    assert np.allclose(qv_out[0, 0:3].cpu().numpy(), expected_v0, atol=1e-5)
    assert np.allclose(qp_out[0, 0:3].cpu().numpy(), expected_p0, atol=1e-5)

    # World 1 must fail safely with status -2 (arithmetic overflow) and NANs
    assert stat_out[1].item() == -2
    assert torch.all(torch.isnan(qp_out[1]))
    assert torch.all(torch.isnan(qv_out[1]))


def test_integrator_degenerate_quaternion_and_neighbor_isolation():
    """Verifies that degenerate quaternion produces status -3 with NANs, isolating neighboring worlds."""
    ps = RepresentativePhysicsSlice(batch_size=2)
    dt = 0.005

    qp = torch.zeros((2, 21), dtype=torch.float32, device=ps.device)
    qp[0, 3] = 1.0  # World 0: valid unit quaternion
    qp[1, 3:7] = 0.0  # World 1: degenerate zero quaternion
    qv = torch.zeros((2, 20), dtype=torch.float32, device=ps.device)
    qa = torch.zeros((2, 20), dtype=torch.float32, device=ps.device)
    stat = torch.zeros((2,), dtype=torch.int32, device=ps.device)

    qp_out, qv_out, stat_out = ps.integrate_implicit_fast(qp, qv, qa, stat, dt=dt)
    torch.mps.synchronize()

    # World 0 must succeed
    assert stat_out[0].item() == 0
    assert torch.all(torch.isfinite(qp_out[0]))
    assert torch.all(torch.isfinite(qv_out[0]))

    # World 1 must fail safely with status -3 and NANs
    assert stat_out[1].item() == -3
    assert torch.all(torch.isnan(qp_out[1]))
    assert torch.all(torch.isnan(qv_out[1]))


# ==============================================================================
# Stage 2: Common-State One-Step Parity Across All 25 Scenarios
# ==============================================================================

@pytest.mark.parametrize("sc_name", SCENARIOS_ALL_25)
def test_autonomous_one_step_all_25_scenarios(sc_name):
    """Verifies one-step 5 ms parity against matched CPU models across all 25 corpus scenarios."""
    ps = RepresentativePhysicsSlice(batch_size=1)
    canonical = load_canonical_model()

    fpath = Path("corpus") / f"{sc_name}.npz"
    assert fpath.exists(), f"Required fixture {fpath} does not exist"
    d_npz = dict(np.load(fpath))

    # Matched CPU reference model & fresh data
    m_matched = create_matching_cpu_model(canonical, d_npz)
    d_matched = create_matching_cpu_data(m_matched, d_npz)
    mujoco.mj_step(m_matched, d_matched)

    qp_t = torch.from_numpy(d_npz["qpos"].astype(np.float32)).unsqueeze(0).to(ps.device)
    qv_t = torch.from_numpy(d_npz["qvel"].astype(np.float32)).unsqueeze(0).to(ps.device)
    f_smooth = (
        torch.from_numpy(d_npz["qfrc_smooth"].astype(np.float32)).unsqueeze(0).to(ps.device)
        if ("qfrc_applied" in d_npz and np.any(d_npz["qfrc_applied"] != 0))
        else None
    )
    pwm = torch.from_numpy(d_npz["per_world_mass"].astype(np.float32)).unsqueeze(0).to(ps.device) if "per_world_mass" in d_npz and np.any(d_npz["per_world_mass"] != 0) else None
    pwi = torch.from_numpy(d_npz["per_world_ipos"].astype(np.float32)).unsqueeze(0).to(ps.device) if "per_world_ipos" in d_npz and np.any(d_npz["per_world_ipos"] != 0) else None
    pwa = torch.from_numpy(d_npz["per_world_armature"].astype(np.float32)).unsqueeze(0).to(ps.device) if "per_world_armature" in d_npz and np.any(d_npz["per_world_armature"] != 0) else None

    f_tensor = None
    if "randomized_friction" in sc_name and int(d_npz["ncon"]) > 0:
        f_tensor = torch.from_numpy(d_npz["contact_friction"][:, :2].astype(np.float32)).unsqueeze(0).to(ps.device)

    step_res = ps.step_autonomous(
        qp_t, qv_t, f_smooth=f_smooth, friction=f_tensor,
        per_world_mass=pwm, per_world_ipos=pwi, per_world_armature=pwa,
        max_iters=200, tol=1e-5, dt=0.005
    )
    torch.mps.synchronize()

    assert step_res.integration_status[0].item() == 0, f"{sc_name} failed integration"

    qp_gpu = step_res.qpos[0].cpu().numpy()
    qv_gpu = step_res.qvel[0].cpu().numpy()
    metrics = compute_separated_metrics(qp_gpu, qv_gpu, d_matched.qpos, d_matched.qvel)

    assert metrics["pos_err"] < 1e-3, f"{sc_name} pos_err {metrics['pos_err']:.2e} >= 1e-3 m"
    assert metrics["so3_err"] < 1e-3, f"{sc_name} so3_err {metrics['so3_err']:.2e} >= 1e-3 rad"
    assert metrics["linvel_err"] < 0.05, f"{sc_name} linvel_err {metrics['linvel_err']:.2e} >= 0.05 m/s"
    assert metrics["angvel_err"] < 0.05, f"{sc_name} angvel_err {metrics['angvel_err']:.2e} >= 0.05 rad/s"
    assert metrics["jnt_pos_err"] < 1e-3, f"{sc_name} jnt_pos_err {metrics['jnt_pos_err']:.2e} >= 1e-3 rad"
    assert metrics["jnt_vel_err"] < 0.05, f"{sc_name} jnt_vel_err {metrics['jnt_vel_err']:.2e} >= 0.05 rad/s"


# ==============================================================================
# Stage 3: Free-Running Trajectories
# ==============================================================================

def test_free_running_airborne_trajectory_20_and_200_steps():
    """Verifies free-running autonomous airborne flight without contacts across 20 steps (100 ms) and 200 steps (1.0 s)."""
    ps = RepresentativePhysicsSlice(batch_size=1)
    c = load_canonical_model()
    m = c.model
    d = mujoco.MjData(m)

    data = np.load("corpus/airborne.npz")
    qpos = data["qpos"].copy()
    qvel = data["qvel"].copy()
    # High altitude so it stays purely airborne across 200 steps
    qpos[2] = 10.0

    d.qpos[:] = qpos
    d.qvel[:] = qvel

    qp_t = torch.from_numpy(qpos.astype(np.float32)).unsqueeze(0).to(ps.device)
    qv_t = torch.from_numpy(qvel.astype(np.float32)).unsqueeze(0).to(ps.device)

    rollout = ps.rollout_trajectory(qp_t, qv_t, num_steps=200, dt=0.005)
    torch.mps.synchronize()

    # Step CPU reference
    cpu_qpos = [qpos.copy()]
    cpu_qvel = [qvel.copy()]
    for _ in range(200):
        mujoco.mj_step(m, d)
        cpu_qpos.append(d.qpos.copy())
        cpu_qvel.append(d.qvel.copy())

    gpu_qpos = rollout["qpos"][:, 0, :].cpu().numpy()
    gpu_qvel = rollout["qvel"][:, 0, :].cpu().numpy()

    # Check 20 steps (100 ms)
    m20 = compute_separated_metrics(gpu_qpos[20], gpu_qvel[20], cpu_qpos[20], cpu_qvel[20])
    assert m20["pos_err"] < 1e-4, f"20-step airborne pos_err {m20['pos_err']:.2e} >= 1e-4 m"
    assert m20["so3_err"] < 1e-4, f"20-step airborne so3_err {m20['so3_err']:.2e} >= 1e-4 rad"
    assert m20["linvel_err"] < 1e-4, f"20-step airborne linvel_err {m20['linvel_err']:.2e} >= 1e-4 m/s"

    # Check 200 steps (1.0 s)
    m200 = compute_separated_metrics(gpu_qpos[200], gpu_qvel[200], cpu_qpos[200], cpu_qvel[200])
    assert m200["pos_err"] < 1e-3, f"200-step airborne pos_err {m200['pos_err']:.2e} >= 1e-3 m"
    assert m200["so3_err"] < 1e-3, f"200-step airborne so3_err {m200['so3_err']:.2e} >= 1e-3 rad"
    assert m200["linvel_err"] < 1e-3, f"200-step airborne linvel_err {m200['linvel_err']:.2e} >= 1e-3 m/s"


def test_free_running_nominal_standing_4_and_20_steps():
    """Verifies free-running autonomous standing contact trajectory across 4 steps (20 ms) and 20 steps (100 ms)."""
    ps = RepresentativePhysicsSlice(batch_size=1)
    canonical = load_canonical_model()

    data = dict(np.load("corpus/nominal_standing_realistic.npz"))
    m_matched = create_matching_cpu_model(canonical, data)
    d_matched = create_matching_cpu_data(m_matched, data)

    qpos = data["qpos"].copy()
    qvel = data["qvel"].copy()

    qp_t = torch.from_numpy(qpos.astype(np.float32)).unsqueeze(0).to(ps.device)
    qv_t = torch.from_numpy(qvel.astype(np.float32)).unsqueeze(0).to(ps.device)

    rollout = ps.rollout_trajectory(qp_t, qv_t, num_steps=20, dt=0.005)
    torch.mps.synchronize()

    # Step CPU reference with matched model
    cpu_qpos = [qpos.copy()]
    cpu_qvel = [qvel.copy()]
    for _ in range(20):
        mujoco.mj_step(m_matched, d_matched)
        cpu_qpos.append(d_matched.qpos.copy())
        cpu_qvel.append(d_matched.qvel.copy())

    gpu_qpos = rollout["qpos"][:, 0, :].cpu().numpy()
    gpu_qvel = rollout["qvel"][:, 0, :].cpu().numpy()

    # Verify integration status throughout rollout
    assert torch.all(rollout["integration_status"] == 0), "Integration status failure during standing rollout"

    # Step 4 (20 ms, 1 control interval)
    m4 = compute_separated_metrics(gpu_qpos[4], gpu_qvel[4], cpu_qpos[4], cpu_qvel[4])
    assert m4["pos_err"] < 1e-3, f"4-step standing pos_err {m4['pos_err']:.2e} >= 1e-3 m"
    assert m4["so3_err"] < 1e-3, f"4-step standing so3_err {m4['so3_err']:.2e} >= 1e-3 rad"
    assert m4["linvel_err"] < 0.05, f"4-step standing linvel_err {m4['linvel_err']:.2e} >= 0.05 m/s"
    assert m4["angvel_err"] < 0.05, f"4-step standing angvel_err {m4['angvel_err']:.2e} >= 0.05 rad/s"
    assert m4["jnt_pos_err"] < 1e-3, f"4-step standing jnt_pos_err {m4['jnt_pos_err']:.2e} >= 1e-3 rad"
    assert m4["jnt_vel_err"] < 0.05, f"4-step standing jnt_vel_err {m4['jnt_vel_err']:.2e} >= 0.05 rad/s"

    # Step 20 (100 ms, 5 control intervals) - strict 1e-3 m and 1e-3 rad gates enforced
    m20 = compute_separated_metrics(gpu_qpos[20], gpu_qvel[20], cpu_qpos[20], cpu_qvel[20])
    assert m20["pos_err"] < 1e-3, f"20-step standing pos_err {m20['pos_err']:.2e} >= 1e-3 m"
    assert m20["so3_err"] < 1e-3, f"20-step standing so3_err {m20['so3_err']:.2e} >= 1e-3 rad"
    assert m20["linvel_err"] < 0.05, f"20-step standing linvel_err {m20['linvel_err']:.2e} >= 0.05 m/s"
    assert m20["angvel_err"] < 0.05, f"20-step standing angvel_err {m20['angvel_err']:.2e} >= 0.05 rad/s"
    assert m20["jnt_pos_err"] < 1e-3, f"20-step standing jnt_pos_err {m20['jnt_pos_err']:.2e} >= 1e-3 rad"
    assert m20["jnt_vel_err"] < 0.05, f"20-step standing jnt_vel_err {m20['jnt_vel_err']:.2e} >= 0.05 rad/s"


def test_free_running_landing_contact_transition():
    """Verifies contact transitions: falling through air, touchdown impact capture, and lift-off separation."""
    ps = RepresentativePhysicsSlice(batch_size=1)

    # 1. Touchdown impact transition: drop from 5 cm above ground
    data = np.load("corpus/nominal_standing_realistic.npz")
    qp_drop = data["qpos"].copy()
    qp_drop[2] += 0.05  # 5 cm above ground
    qv_drop = data["qvel"].copy()

    qp_t = torch.from_numpy(qp_drop.astype(np.float32)).unsqueeze(0).to(ps.device)
    qv_t = torch.from_numpy(qv_drop.astype(np.float32)).unsqueeze(0).to(ps.device)

    rollout_drop = ps.rollout_trajectory(qp_t, qv_t, num_steps=25, dt=0.005)
    torch.mps.synchronize()

    nefc_drop = rollout_drop["nefc"][:, 0].cpu().numpy()
    vz_drop = rollout_drop["qvel"][:, 0, 2].cpu().numpy()

    # Steps 0..18 must be airborne (0 active constraint rows)
    assert np.all(nefc_drop[:18] == 0), "Expected zero contact rows during initial free-fall"
    # Step 20+ must establish contact constraints upon touchdown
    assert nefc_drop[20] > 0, "Expected active constraint rows upon touchdown at step 20"
    assert nefc_drop[21] == 24, f"Expected full 24 constraint rows at full impact, got {nefc_drop[21]}"
    # Downward velocity (-0.98 m/s) must be arrested by normal contact forces
    assert vz_drop[19] < -0.9, f"Expected large downward velocity before impact, got {vz_drop[19]}"
    assert abs(vz_drop[24]) < 0.1, f"Expected downward velocity arrested near zero upon ground capture, got {vz_drop[24]}"

    # 2. Lift-off transition: contact_onset scenario (upward bounce / separation)
    data_onset = np.load("corpus/contact_onset.npz")
    qp_onset_t = torch.from_numpy(data_onset["qpos"].astype(np.float32)).unsqueeze(0).to(ps.device)
    qv_onset_t = torch.from_numpy(data_onset["qvel"].astype(np.float32)).unsqueeze(0).to(ps.device)

    rollout_onset = ps.rollout_trajectory(qp_onset_t, qv_onset_t, num_steps=20, dt=0.005)
    torch.mps.synchronize()

    nefc_onset = rollout_onset["nefc"][:, 0].cpu().numpy()
    # Initially in contact (24 rows), then separates into flight (0 rows)
    assert nefc_onset[0] == 24, "Expected initial ground contact"
    assert nefc_onset[-1] == 0, "Expected lift-off into airborne flight"


def test_free_running_sliding_lateral_dissipation():
    """Verifies friction dissipation on sliding trajectory."""
    ps = RepresentativePhysicsSlice(batch_size=1)
    data = np.load("corpus/sliding_lateral_velocity.npz")
    qpos = data["qpos"].copy()
    qvel = data["qvel"].copy()

    qp_t = torch.from_numpy(qpos.astype(np.float32)).unsqueeze(0).to(ps.device)
    qv_t = torch.from_numpy(qvel.astype(np.float32)).unsqueeze(0).to(ps.device)

    rollout = ps.rollout_trajectory(qp_t, qv_t, num_steps=20, dt=0.005)
    torch.mps.synchronize()

    vx_hist = rollout["qvel"][:, 0, 0].cpu().numpy()
    vy_hist = rollout["qvel"][:, 0, 1].cpu().numpy()
    speed_init = math.hypot(vx_hist[0], vy_hist[0])
    speed_final = math.hypot(vx_hist[-1], vy_hist[-1])

    # Friction forces should dissipate tangential sliding velocity
    assert speed_final < speed_init, f"Expected friction dissipation: speed_final {speed_final:.3f} >= speed_init {speed_init:.3f}"


# ==============================================================================
# Stage 4: Actuators, Control Interval & Substepping
# ==============================================================================

def test_actuator_torque_control_parity():
    """Verifies that 14 joint actuator torques passed via ctrl match MuJoCo CPU d.ctrl[:] = ctrl."""
    ps = RepresentativePhysicsSlice(batch_size=1)
    c = load_canonical_model()
    m = c.model
    d = mujoco.MjData(m)

    data = np.load("corpus/airborne.npz")
    qpos = data["qpos"].copy()
    qvel = data["qvel"].copy()

    # Controlled torque commands within actuator_forcerange [-1.06755, 1.06755]
    ctrl = np.linspace(-0.8, 0.8, 14, dtype=np.float64)

    d.qpos[:] = qpos
    d.qvel[:] = qvel
    d.ctrl[:] = ctrl
    mujoco.mj_step(m, d)

    qp_t = torch.from_numpy(qpos.astype(np.float32)).unsqueeze(0).to(ps.device)
    qv_t = torch.from_numpy(qvel.astype(np.float32)).unsqueeze(0).to(ps.device)
    ctrl_t = torch.from_numpy(ctrl.astype(np.float32)).unsqueeze(0).to(ps.device)

    step_res = ps.step_autonomous(qp_t, qv_t, ctrl=ctrl_t, dt=0.005)
    torch.mps.synchronize()

    qp_gpu = step_res.qpos[0].cpu().numpy()
    qv_gpu = step_res.qvel[0].cpu().numpy()

    metrics = compute_separated_metrics(qp_gpu, qv_gpu, d.qpos, d.qvel)

    assert metrics["pos_err"] < 1e-3, f"Actuator pos_err {metrics['pos_err']:.2e} >= 1e-3 m"
    assert metrics["so3_err"] < 1e-3, f"Actuator so3_err {metrics['so3_err']:.2e} >= 1e-3 rad"
    assert metrics["linvel_err"] < 0.05, f"Actuator linvel_err {metrics['linvel_err']:.2e} >= 0.05 m/s"
    assert metrics["angvel_err"] < 0.05, f"Actuator angvel_err {metrics['angvel_err']:.2e} >= 0.05 rad/s"
    assert metrics["jnt_pos_err"] < 1e-3, f"Actuator jnt_pos_err {metrics['jnt_pos_err']:.2e} >= 1e-3 rad"
    assert metrics["jnt_vel_err"] < 0.05, f"Actuator jnt_vel_err {metrics['jnt_vel_err']:.2e} >= 0.05 rad/s"


def test_actuator_forcerange_clamping():
    """Verifies that actuator commands outside [-1.06755, 1.06755] are clamped identically to MuJoCo CPU."""
    ps = RepresentativePhysicsSlice(batch_size=1)
    c = load_canonical_model()
    m = c.model
    d = mujoco.MjData(m)

    data = np.load("corpus/airborne.npz")
    qpos = data["qpos"].copy()
    qvel = data["qvel"].copy()

    # Exceed forcerange: +/- 5.0 N*m
    ctrl = np.linspace(-5.0, 5.0, 14, dtype=np.float64)

    d.qpos[:] = qpos
    d.qvel[:] = qvel
    d.ctrl[:] = ctrl
    mujoco.mj_step(m, d)

    qp_t = torch.from_numpy(qpos.astype(np.float32)).unsqueeze(0).to(ps.device)
    qv_t = torch.from_numpy(qvel.astype(np.float32)).unsqueeze(0).to(ps.device)
    ctrl_t = torch.from_numpy(ctrl.astype(np.float32)).unsqueeze(0).to(ps.device)

    step_res = ps.step_autonomous(qp_t, qv_t, ctrl=ctrl_t, dt=0.005)
    torch.mps.synchronize()

    qp_gpu = step_res.qpos[0].cpu().numpy()
    qv_gpu = step_res.qvel[0].cpu().numpy()

    metrics = compute_separated_metrics(qp_gpu, qv_gpu, d.qpos, d.qvel)

    assert metrics["pos_err"] < 1e-3, f"Clamped pos_err {metrics['pos_err']:.2e} >= 1e-3 m"
    assert metrics["so3_err"] < 1e-3, f"Clamped so3_err {metrics['so3_err']:.2e} >= 1e-3 rad"
    assert metrics["linvel_err"] < 0.05, f"Clamped linvel_err {metrics['linvel_err']:.2e} >= 0.05 m/s"
    assert metrics["jnt_vel_err"] < 0.05, f"Clamped jnt_vel_err {metrics['jnt_vel_err']:.2e} >= 0.05 rad/s"


def test_control_interval_substepping_parity():
    """Verifies that step_control_interval advances 4 substeps (20 ms) matching MuJoCo CPU."""
    ps = RepresentativePhysicsSlice(batch_size=1)
    c = load_canonical_model()
    m = c.model
    d = mujoco.MjData(m)

    data = np.load("corpus/airborne.npz")
    qpos = data["qpos"].copy()
    qvel = data["qvel"].copy()
    ctrl = np.linspace(-0.5, 0.5, 14, dtype=np.float64)

    d.qpos[:] = qpos
    d.qvel[:] = qvel
    d.ctrl[:] = ctrl

    # Step CPU 4 substeps holding ctrl constant
    for _ in range(4):
        mujoco.mj_step(m, d)

    qp_t = torch.from_numpy(qpos.astype(np.float32)).unsqueeze(0).to(ps.device)
    qv_t = torch.from_numpy(qvel.astype(np.float32)).unsqueeze(0).to(ps.device)
    ctrl_t = torch.from_numpy(ctrl.astype(np.float32)).unsqueeze(0).to(ps.device)

    qp_next, qv_next, substep_outs = ps.step_control_interval(
        qp_t, qv_t, ctrl_t, num_substeps=4, dt=0.005
    )
    torch.mps.synchronize()

    assert len(substep_outs) == 4

    metrics = compute_separated_metrics(qp_next[0].cpu().numpy(), qv_next[0].cpu().numpy(), d.qpos, d.qvel)

    assert metrics["pos_err"] < 1e-3, f"Control interval pos_err {metrics['pos_err']:.2e} >= 1e-3 m"
    assert metrics["so3_err"] < 1e-3, f"Control interval so3_err {metrics['so3_err']:.2e} >= 1e-3 rad"
    assert metrics["linvel_err"] < 0.05, f"Control interval linvel_err {metrics['linvel_err']:.2e} >= 0.05 m/s"
    assert metrics["angvel_err"] < 0.05, f"Control interval angvel_err {metrics['angvel_err']:.2e} >= 0.05 rad/s"
    assert metrics["jnt_pos_err"] < 1e-3, f"Control interval jnt_pos_err {metrics['jnt_pos_err']:.2e} >= 1e-3 rad"
    assert metrics["jnt_vel_err"] < 0.05, f"Control interval jnt_vel_err {metrics['jnt_vel_err']:.2e} >= 0.05 rad/s"


def test_forward_autonomous_simultaneous_ctrl_and_fsmooth_rejected():
    """Verifies that providing both ctrl and f_smooth simultaneously raises ValueError to prevent double counting."""
    ps = RepresentativePhysicsSlice(batch_size=1)
    qp = torch.zeros((1, 21), dtype=torch.float32, device=ps.device)
    qp[0, 3] = 1.0
    qv = torch.zeros((1, 20), dtype=torch.float32, device=ps.device)
    ctrl = torch.zeros((1, 14), dtype=torch.float32, device=ps.device)
    f_sm = torch.zeros((1, 20), dtype=torch.float32, device=ps.device)

    with pytest.raises(ValueError, match="mutually exclusive"):
        ps.forward_autonomous(qp, qv, ctrl=ctrl, f_smooth=f_sm)

    with pytest.raises(ValueError, match="mutually exclusive"):
        ps.step_autonomous(qp, qv, ctrl=ctrl, f_smooth=f_sm)


def test_substep_diagnostic_history_ownership():
    """Verifies that substep diagnostic outputs do not alias across iterations and match sequential steps."""
    ps = RepresentativePhysicsSlice(batch_size=1)
    canonical = load_canonical_model()

    # Drop onset scenario: starts airborne at step 0 (nefc=0), establishing contact during substeps
    data = dict(np.load("corpus/nominal_standing_realistic.npz"))
    qp_drop = data["qpos"].copy()
    qp_drop[2] += 0.005  # 5 mm drop: contacts change across substeps
    qv_drop = np.zeros(20, dtype=np.float32)
    ctrl = torch.zeros((1, 14), dtype=torch.float32, device=ps.device)

    qp_t = torch.from_numpy(qp_drop.astype(np.float32)).unsqueeze(0).to(ps.device)
    qv_t = torch.from_numpy(qv_drop).unsqueeze(0).to(ps.device)

    # 1. Run 4 substeps via step_control_interval
    qp_final, qv_final, substep_outs = ps.step_control_interval(
        qp_t, qv_t, ctrl, num_substeps=4, dt=0.005
    )
    torch.mps.synchronize()

    assert len(substep_outs) == 4

    # 2. Run 4 independent sequential steps
    qp_seq = qp_t.clone()
    qv_seq = qv_t.clone()
    seq_outs = []
    for _ in range(4):
        s_res = ps.step_autonomous(qp_seq, qv_seq, ctrl=ctrl, dt=0.005)
        qp_seq = s_res.qpos.clone()
        qv_seq = s_res.qvel.clone()
        seq_outs.append(s_res.clone())
    torch.mps.synchronize()

    # 3. Verify history integrity:
    # Substep outputs must not alias memory with each other
    for i in range(3):
        for j in range(i + 1, 4):
            assert substep_outs[i].qpos.data_ptr() != substep_outs[j].qpos.data_ptr()
            assert substep_outs[i].physics_outputs.J.data_ptr() != substep_outs[j].physics_outputs.J.data_ptr()
            assert substep_outs[i].physics_outputs.lambda_force.data_ptr() != substep_outs[j].physics_outputs.lambda_force.data_ptr()

    # Substep outputs must match sequential execution at every substep
    for s in range(4):
        assert torch.allclose(substep_outs[s].qpos, seq_outs[s].qpos, atol=1e-6)
        assert torch.allclose(substep_outs[s].qvel, seq_outs[s].qvel, atol=1e-6)
        assert substep_outs[s].integration_status.item() == seq_outs[s].integration_status.item()
        assert substep_outs[s].nefc.item() == seq_outs[s].nefc.item()
        assert torch.allclose(substep_outs[s].physics_outputs.lambda_force, seq_outs[s].physics_outputs.lambda_force, atol=1e-5)


# ==============================================================================
# Stage 5: Multi-World Reset, Failure Isolation & Error Recovery
# ==============================================================================

def test_mixed_world_failure_isolation():
    """Verifies that an invalid world (non-finite inputs) does not corrupt a neighboring valid world in batch."""
    ps = RepresentativePhysicsSlice(batch_size=2)
    data = np.load("corpus/nominal_standing_realistic.npz")
    qpos = data["qpos"].copy()
    qvel = data["qvel"].copy()

    qp_t = torch.from_numpy(np.tile(qpos, (2, 1)).astype(np.float32)).to(ps.device)
    qv_t = torch.from_numpy(np.tile(qvel, (2, 1)).astype(np.float32)).to(ps.device)

    # Invalidate World 1
    qp_t[1, 0] = float("nan")

    step_res = ps.step_autonomous(qp_t, qv_t, dt=0.005)
    torch.mps.synchronize()

    # World 0 must succeed cleanly
    assert step_res.integration_status[0].item() == 0
    assert torch.all(torch.isfinite(step_res.qpos[0]))
    assert torch.all(torch.isfinite(step_res.qvel[0]))

    # World 1 must fail safely without corrupting World 0
    assert step_res.integration_status[1].item() < 0
    assert torch.all(torch.isnan(step_res.qpos[1]))
    assert torch.all(torch.isnan(step_res.qvel[1]))


def test_mixed_world_reset_recovery():
    """Verifies valid -> invalid -> reset -> valid lifecycle across successive steps."""
    ps = RepresentativePhysicsSlice(batch_size=2)
    data = np.load("corpus/nominal_standing_realistic.npz")
    qpos_valid = data["qpos"].copy()
    qvel_valid = data["qvel"].copy()

    # Step 0: Both valid
    qp = torch.from_numpy(np.tile(qpos_valid, (2, 1)).astype(np.float32)).to(ps.device)
    qv = torch.from_numpy(np.tile(qvel_valid, (2, 1)).astype(np.float32)).to(ps.device)
    res0 = ps.step_autonomous(qp, qv, dt=0.005)
    assert res0.integration_status[0].item() == 0
    assert res0.integration_status[1].item() == 0

    # Step 1: Invalidate World 1
    qp_t1 = res0.qpos.clone()
    qv_t1 = res0.qvel.clone()
    qp_t1[1, 2] = float("nan")
    res1 = ps.step_autonomous(qp_t1, qv_t1, dt=0.005)
    assert res1.integration_status[0].item() == 0
    assert res1.integration_status[1].item() < 0
    assert torch.all(torch.isnan(res1.qpos[1]))

    # Step 2: Reset World 1 back to valid standing state
    qp_t2 = res1.qpos.clone()
    qv_t2 = res1.qvel.clone()
    qp_t2[1] = torch.from_numpy(qpos_valid.astype(np.float32)).to(ps.device)
    qv_t2[1] = torch.from_numpy(qvel_valid.astype(np.float32)).to(ps.device)
    res2 = ps.step_autonomous(qp_t2, qv_t2, dt=0.005)

    # World 1 must have cleanly recovered
    assert res2.integration_status[0].item() == 0
    assert res2.integration_status[1].item() == 0
    assert torch.all(torch.isfinite(res2.qpos[1]))
    assert torch.all(torch.isfinite(res2.qvel[1]))
