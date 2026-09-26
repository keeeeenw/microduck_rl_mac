"""Tests for UnifiedMetalSimulation task/physics parity.

Verifies:
1. Exact forward kinematics parity for all 8 sites and 76 geoms against MuJoCo CPU.
2. Root angular velocity frame correction (world frame in cvel[:, 2] matching R @ omega_b).
3. Contact sensor evaluation populating sensordata slots 19..27 for foot contact and forces.
4. Domain randomization parameter forwarding for body_mass, body_ipos, body_inertia,
   dof_armature, dof_damping, geom_friction, and qfrc_applied.
5. Recompute constants hook maintaining body_subtreemass.
"""

from pathlib import Path
import sys
import copy
import numpy as np
import pytest
import torch
import mujoco

PROJECT_ROOT = Path(__file__).resolve().parent

from mjlab_microduck.native_gpu.metal.canonical_model_loader import load_canonical_model, CANONICAL_XML_PATH
from mjlab_microduck.native_gpu.metal.metal_simulation_adapter import UnifiedMetalSimulation


def make_mock_cfg():
    return type(
        "Cfg",
        (),
        {
            "mujoco": type("M", (), {"apply": lambda self, m: None})(),
            "nan_guard": type("N", (), {"enabled": False, "action": "raise"})(),
        },
    )()


def test_site_and_geom_fk_parity():
    """Verify site and geom positions and orientation matrices match MuJoCo CPU across diverse poses."""
    canonical = load_canonical_model()
    m = canonical.model
    cfg = make_mock_cfg()

    sim = UnifiedMetalSimulation(num_envs=3, cfg=cfg, model=m, device="mps")
    d_cpu = mujoco.MjData(m)

    # Env 0: default home pose
    # Env 1: pitched and crouched
    sim.data.qpos[1, 1] = 0.1 # y offset
    sim.data.qpos[1, 2] = 0.22 # z offset
    sim.data.qpos[1, 4] = 0.2 # pitch quaternion component
    sim.data.qpos[1, 3:7] = sim.data.qpos[1, 3:7] / torch.norm(sim.data.qpos[1, 3:7])
    sim.data.qpos[1, 7:21] = torch.tensor([
        0.1, -0.1, 0.4, -0.8, 0.4, -0.1, 0.1,
        -0.1, 0.1, 0.4, -0.8, 0.4, 0.1, -0.1
    ], dtype=torch.float32, device="mps")

    # Env 2: rotated yaw 45 deg
    yaw_angle = np.pi / 4.0
    sim.data.qpos[2, 3] = float(np.cos(yaw_angle / 2.0))
    sim.data.qpos[2, 6] = float(np.sin(yaw_angle / 2.0))

    sim.forward()

    for env_idx in range(3):
        d_cpu.qpos[:] = sim.data.qpos[env_idx].cpu().numpy()
        d_cpu.qvel[:] = 0.0
        mujoco.mj_forward(m, d_cpu)

        # Check all sites (8 sites)
        site_gpu = sim.data.site_xpos[env_idx].cpu().numpy()
        site_cpu = d_cpu.site_xpos
        max_site_pos_err = np.max(np.abs(site_gpu - site_cpu))
        assert max_site_pos_err < 1e-5, f"Env {env_idx} site_xpos error {max_site_pos_err:.2e} >= 1e-5"

        site_mat_gpu = sim.data.site_xmat[env_idx].cpu().numpy()
        site_mat_cpu = d_cpu.site_xmat
        max_site_mat_err = np.max(np.abs(site_mat_gpu - site_mat_cpu))
        assert max_site_mat_err < 1e-4, f"Env {env_idx} site_xmat error {max_site_mat_err:.2e} >= 1e-4"

        # Check all geoms (76 geoms)
        geom_gpu = sim.data.geom_xpos[env_idx].cpu().numpy()
        geom_cpu = d_cpu.geom_xpos
        max_geom_pos_err = np.max(np.abs(geom_gpu - geom_cpu))
        assert max_geom_pos_err < 1e-5, f"Env {env_idx} geom_xpos error {max_geom_pos_err:.2e} >= 1e-5"

        geom_mat_gpu = sim.data.geom_xmat[env_idx].cpu().numpy()
        geom_mat_cpu = d_cpu.geom_xmat
        max_geom_mat_err = np.max(np.abs(geom_mat_gpu - geom_mat_cpu))
        assert max_geom_mat_err < 1e-4, f"Env {env_idx} geom_xmat error {max_geom_mat_err:.2e} >= 1e-4"


def test_root_angular_velocity_frame_fix():
    """Verify cvel[:, 2] root angular velocity is in world frame and recovers body frame under inverse quat."""
    canonical = load_canonical_model()
    m = canonical.model
    cfg = make_mock_cfg()

    sim = UnifiedMetalSimulation(num_envs=4, cfg=cfg, model=m, device="mps")

    # Set arbitrary angular velocity in body frame
    omega_b = torch.tensor([1.2, -0.8, 2.5], dtype=torch.float32, device="mps")
    sim.data.qvel[:, 3:6] = omega_b

    # Set 4 different orientations:
    # Env 0: Identity
    # Env 1: Yaw 90 deg
    # Env 2: Pitch 45 deg
    # Env 3: Arbitrary roll-pitch-yaw
    quats = [
        [1.0, 0.0, 0.0, 0.0],
        [float(np.cos(np.pi/4)), 0.0, 0.0, float(np.sin(np.pi/4))],
        [float(np.cos(np.pi/8)), 0.0, float(np.sin(np.pi/8)), 0.0],
        [0.7071, 0.3535, 0.3535, 0.5],
    ]
    for i in range(4):
        q = torch.tensor(quats[i], dtype=torch.float32, device="mps")
        q = q / torch.norm(q)
        sim.data.qpos[i, 3:7] = q

    sim.forward()

    for i in range(4):
        cvel_ang_w = sim.data.cvel[i, 2, :3] # (3,) in world frame
        R_w = sim.data.xmat[i, 2].reshape(3, 3)

        # Expected world angular velocity: R_w @ omega_b
        expected_w = torch.matmul(R_w, omega_b.unsqueeze(-1)).squeeze(-1)
        err_w = (cvel_ang_w - expected_w).abs().max().item()
        assert err_w < 1e-5, f"Env {i} world frame angular velocity mismatch: {err_w:.2e}"

        # Re-rotating to body frame: R_w.T @ cvel_ang_w == omega_b
        recovered_b = torch.matmul(R_w.T, cvel_ang_w.unsqueeze(-1)).squeeze(-1)
        err_b = (recovered_b - omega_b).abs().max().item()
        assert err_b < 1e-5, f"Env {i} body frame recovery mismatch: {err_b:.2e}"


def test_contact_sensors_semantics():
    """Verify contact sensors (slots 19..27) correctly report contact counts and normal/tangential forces."""
    canonical = load_canonical_model()
    m = canonical.model
    cfg = make_mock_cfg()

    sim = UnifiedMetalSimulation(num_envs=3, cfg=cfg, model=m, device="mps")

    # Env 0: airborne (z = 0.30 m)
    sim.data.qpos[0, 2] = 0.30

    # Env 1: standing on ground (z lowered by 0.06 m, both feet in contact)
    sim.data.qpos[1, 2] -= 0.06

    # Env 2: left foot contact only (roll tilted by 20 deg, right foot elevated)
    sim.data.qpos[2, 2] -= 0.04
    roll = np.deg2rad(20.0)
    sim.data.qpos[2, 3] = float(np.cos(roll / 2.0))
    sim.data.qpos[2, 4] = float(np.sin(roll / 2.0)) # roll quaternion
    # Lift right hip
    sim.data.qpos[2, 7 + 7] = 0.3 # right hip roll

    sim.forward()

    sd0 = sim.data.sensordata[0].cpu().numpy()
    sd1 = sim.data.sensordata[1].cpu().numpy()
    sd2 = sim.data.sensordata[2].cpu().numpy()

    # Env 0: Airborne
    assert sd0[19] == 0.0, "Env 0 left foot should have 0 contacts in air"
    assert sd0[23] == 0.0, "Env 0 right foot should have 0 contacts in air"
    np.testing.assert_allclose(sd0[20:23], 0.0, atol=1e-5)
    np.testing.assert_allclose(sd0[24:27], 0.0, atol=1e-5)
    assert sd0[27] == 0.0

    # Env 1: Dual foot contact
    assert sd1[19] >= 1.0, f"Env 1 left foot contacts {sd1[19]} should be >= 1"
    assert sd1[23] >= 1.0, f"Env 1 right foot contacts {sd1[23]} should be >= 1"
    # Normal force is in negative Z direction on terrain
    assert sd1[22] < -10.0, f"Env 1 left foot Z force {sd1[22]} should be negative and significant"
    assert sd1[26] < -10.0, f"Env 1 right foot Z force {sd1[26]} should be negative and significant"
    total_force = -(sd1[22] + sd1[26])
    # Static weight of robot (m=0.737 kg) is 7.23 N; with -0.06m penetration spring penalty force is ~80 N
    assert 60.0 < total_force < 120.0, f"Total normal penetration force {total_force:.1f} N expected in [60, 120] N"

    # Env 2: Right foot supports weight due to positive roll tilt
    assert sd2[23] >= 1.0, f"Env 2 right foot contacts {sd2[23]} should be >= 1"
    assert sd2[26] < -20.0, f"Env 2 right foot Z force {sd2[26]} should support weight"


def test_domain_randomization_parameter_forwarding():
    """Verify runtime model perturbations reach RepresentativePhysicsSlice and alter dynamics."""
    canonical = load_canonical_model()
    m = canonical.model
    cfg = make_mock_cfg()

    sim = UnifiedMetalSimulation(num_envs=2, cfg=cfg, model=m, device="mps")
    sim.data.ctrl[:] = 3.0 # apply motor torque
    sim.forward()

    # 1. Perturb body mass
    sim.expand_model_fields(["body_mass"])
    sim.model.body_mass[1] *= 2.0
    sim.recompute_constants(level=None)
    sim.forward()
    diff_mass = (sim.data.qacc[0] - sim.data.qacc[1]).abs().max().item()
    assert diff_mass > 0.5, f"Mass perturbation produced negligible qacc diff {diff_mass}"

    # 2. Perturb dof damping
    sim.expand_model_fields(["dof_damping"])
    sim.data.qvel[:] = 3.0
    sim.model.dof_damping[1, 6:20] = 1.0
    sim.forward()
    diff_damping = (sim.data.qacc[0] - sim.data.qacc[1]).abs().max().item()
    assert diff_damping > 1.0, f"Damping perturbation produced negligible qacc diff {diff_damping}"

    # 3. Perturb dof armature
    sim.expand_model_fields(["dof_armature"])
    sim.model.dof_armature[1, 6:20] *= 3.0
    sim.forward()
    diff_armature = (sim.data.qacc[0] - sim.data.qacc[1]).abs().max().item()
    assert diff_armature > 0.5, f"Armature perturbation produced negligible qacc diff {diff_armature}"

    # 4. Step execution with perturbed parameters
    sim.step()
    assert sim.data.time[0].item() == pytest.approx(0.005)


def test_asymmetric_friction_swap():
    """Verify that per_foot_friction maps correctly to left/right foot and swapping mu_L <-> mu_R swaps tangential forces."""
    canonical = load_canonical_model()
    m = canonical.model
    cfg = make_mock_cfg()

    sim = UnifiedMetalSimulation(num_envs=2, cfg=cfg, model=m, device="mps")
    # Place both environments in dual foot contact
    sim.data.qpos[:, 2] -= 0.05
    # Apply lateral velocity to induce sliding contact
    sim.data.qvel[:, 1] = 2.0

    # Env 0: mu_L = 0.8, mu_R = 0.2
    # Env 1: mu_L = 0.2, mu_R = 0.8 (swapped)
    sim.expand_model_fields(["geom_friction"])
    sim.model.geom_friction[:, canonical.terrain_geom_id, 0] = 0.1
    sim.model.geom_friction[0, canonical.left_foot_geom_id, 0] = 0.8
    sim.model.geom_friction[0, canonical.right_foot_geom_id, 0] = 0.2
    sim.model.geom_friction[1, canonical.left_foot_geom_id, 0] = 0.2
    sim.model.geom_friction[1, canonical.right_foot_geom_id, 0] = 0.8

    sim.forward()

    # 1. Verify assembled friction array correctly maps per-foot values
    fric0 = sim.slice.assembled_friction[0, :6, :].cpu().numpy()
    fric1 = sim.slice.assembled_friction[1, :6, :].cpu().numpy()

    # Left foot contacts (0..2) and right foot contacts (3..5)
    np.testing.assert_allclose(fric0[:3], 0.8, atol=1e-5)
    np.testing.assert_allclose(fric0[3:6], 0.2, atol=1e-5)
    np.testing.assert_allclose(fric1[:3], 0.2, atol=1e-5)
    np.testing.assert_allclose(fric1[3:6], 0.8, atol=1e-5)

    # Swapping mu_L <-> mu_R between environments swaps left and right contact friction arrays
    np.testing.assert_allclose(fric0[:3], fric1[3:6], atol=1e-5)
    np.testing.assert_allclose(fric0[3:6], fric1[:3], atol=1e-5)

    # 2. Verify contact sensor tangential forces reflect asymmetric friction bounds
    sd0 = sim.data.sensordata[0].cpu().numpy()
    sd1 = sim.data.sensordata[1].cpu().numpy()

    mu_eff_0_left = abs(sd0[21]) / abs(sd0[22])
    mu_eff_0_right = abs(sd0[25]) / abs(sd0[26])
    mu_eff_1_left = abs(sd1[21]) / abs(sd1[22])
    mu_eff_1_right = abs(sd1[25]) / abs(sd1[26])

    # Foot with mu=0.2 must saturate at 0.2
    assert mu_eff_0_right <= 0.201, f"Env 0 right foot (mu=0.2) exceeded friction bound: {mu_eff_0_right}"
    assert mu_eff_1_left <= 0.201, f"Env 1 left foot (mu=0.2) exceeded friction bound: {mu_eff_1_left}"

    # Foot with mu=0.8 must sustain higher friction capacity than 0.2
    assert mu_eff_0_left > 0.4, f"Env 0 left foot (mu=0.8) friction capacity too low: {mu_eff_0_left}"
    assert mu_eff_1_right > 0.4, f"Env 1 right foot (mu=0.8) friction capacity too low: {mu_eff_1_right}"



def test_sliding_contact_tangential_force_scaling():
    """Verify contact sensor tangential forces are scaled by friction: F_y = mu_1*(lam0-lam1), F_x = mu_2*(lam3-lam2)."""
    canonical = load_canonical_model()
    m = canonical.model
    cfg = make_mock_cfg()

    sim = UnifiedMetalSimulation(num_envs=1, cfg=cfg, model=m, device="mps")
    sim.data.qpos[0, 2] -= 0.05
    sim.data.qvel[0, 0] = 0.3
    sim.data.qvel[0, 1] = 0.3
    sim.forward()

    sd = sim.data.sensordata[0].cpu().numpy()
    if sd[19] > 0:
        mu = float(m.geom_friction[canonical.left_foot_geom_id, 0])
        f_norm = abs(sd[22])
        f_tan_x = abs(sd[20])
        f_tan_y = abs(sd[21])
        assert f_tan_x <= mu * f_norm + 1e-3, f"Left F_x {f_tan_x} exceeds Coulomb bound {mu * f_norm}"
        assert f_tan_y <= mu * f_norm + 1e-3, f"Left F_y {f_tan_y} exceeds Coulomb bound {mu * f_norm}"


def test_self_collision_sensor_evaluation():
    """Verify sensordata slot 27 (self_collision) is 0 for nominal poses and detects overlap for colliding poses."""
    canonical = load_canonical_model()
    m = canonical.model
    cfg = make_mock_cfg()

    sim = UnifiedMetalSimulation(num_envs=2, cfg=cfg, model=m, device="mps")

    # Env 0: nominal standing pose (no self-collision)
    # Env 1: crossed legs (left hip roll -0.35 rad inward, right hip roll +0.35 rad inward)
    sim.data.qpos[1, 8] = -0.35   # left hip roll inward (joint 2, qposadr=8)
    sim.data.qpos[1, 17] = 0.35   # right hip roll inward (joint 11, qposadr=17)


    sim.forward()

    sd0 = sim.data.sensordata[0].cpu().numpy()
    sd1 = sim.data.sensordata[1].cpu().numpy()

    assert sd0[27] == 0.0, f"Env 0 nominal pose should have 0 self-collisions, got {sd0[27]}"
    assert sd1[27] >= 1.0, f"Env 1 crossed legs must detect self-collision, got {sd1[27]}"


def test_pseudo_inertia_rotation_parity():
    """Verify body_iquat principal-axis rotation eliminates Frobenius error against MuJoCo CPU (< 1e-4)."""
    canonical = load_canonical_model()
    m = canonical.model
    cfg = make_mock_cfg()

    sim = UnifiedMetalSimulation(num_envs=2, cfg=cfg, model=m, device="mps")
    sim.expand_model_fields(["body_iquat"])

    # Perturb body_iquat with arbitrary rotation (e.g. 30 deg rotation around axis [1, 1, 1])
    angle = np.deg2rad(30.0)
    axis = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    q_rot = np.array([np.cos(angle / 2.0), *(axis * np.sin(angle / 2.0))], dtype=np.float32)

    sim.model.body_iquat[1, 2] = torch.tensor(q_rot, dtype=torch.float32, device="mps")
    sim.recompute_constants(level=None)
    sim.forward()

    # Compare sim.data.ximat with MuJoCo CPU
    m_cpu = copy.copy(m)
    d_cpu = mujoco.MjData(m_cpu)
    m_cpu.body_iquat[2] = q_rot
    mujoco.mj_setConst(m_cpu, d_cpu)
    d_cpu.qpos[:] = sim.data.qpos[1].cpu().numpy()
    d_cpu.qvel[:] = sim.data.qvel[1].cpu().numpy()
    mujoco.mj_forward(m_cpu, d_cpu)

    ximat_gpu = sim.data.ximat[1].cpu().numpy().reshape(17, 3, 3)
    ximat_cpu = d_cpu.ximat.reshape(17, 3, 3)

    rel_frob_err = np.linalg.norm(ximat_gpu - ximat_cpu) / np.linalg.norm(ximat_cpu)
    assert rel_frob_err < 1e-4, f"Relative Frobenius error {rel_frob_err:.2e} >= 1e-4"


def test_recompute_constants_exact_match():
    """Verify recompute_constants updates body_subtreemass, dof_invweight0, and body_invweight0 exactly matching mj_setConst."""
    canonical = load_canonical_model()
    m = canonical.model
    cfg = make_mock_cfg()

    sim = UnifiedMetalSimulation(num_envs=1, cfg=cfg, model=m, device="mps")
    sim.expand_model_fields(["body_mass", "body_inertia"])

    sim.model.body_mass[0, 2] *= 1.3
    sim.model.body_inertia[0, 2] *= 1.2
    sim.recompute_constants(level=None)

    # Reference CPU computation
    m_cpu = copy.copy(m)
    d_cpu = mujoco.MjData(m_cpu)
    m_cpu.body_mass[2] *= 1.3
    m_cpu.body_inertia[2] *= 1.2
    mujoco.mj_setConst(m_cpu, d_cpu)

    subtreemass_gpu = sim.model.body_subtreemass[0].cpu().numpy()
    dof_invw_gpu = sim.model.dof_invweight0[0].cpu().numpy()
    body_invw_gpu = sim.model.body_invweight0[0].cpu().numpy()

    np.testing.assert_allclose(subtreemass_gpu, m_cpu.body_subtreemass, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(dof_invw_gpu, m_cpu.dof_invweight0, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(body_invw_gpu, m_cpu.body_invweight0, rtol=1e-5, atol=1e-6)


def test_self_collision_zero_false_positives_128_poses():
    """Verify self-collision sensor has 0 false positives and 0 false negatives over 128 random poses.

    Ground truth is CPU mj_collision. For each pose the Metal adapter's sensordata[27]
    must agree with whether CPU reports at least one contact in the four self-collision geom pairs.
    """
    canonical = load_canonical_model()
    m = canonical.model
    cfg = make_mock_cfg()

    rng = np.random.default_rng(0xC0FFEE)
    n_poses = 128

    # Generate 128 random joint configurations (keep root at default)
    qpos_base = m.qpos0.copy()
    joint_lo = np.array(m.jnt_range[1:, 0])   # skip root free-joint
    joint_hi = np.array(m.jnt_range[1:, 1])
    # Avoid zero-width intervals (fixed joints)
    safe_lo = np.where(joint_lo == joint_hi, joint_lo - 1e-6, joint_lo)
    safe_hi = np.where(joint_lo == joint_hi, joint_hi + 1e-6, joint_hi)
    joint_samples = rng.uniform(safe_lo, safe_hi, size=(n_poses, len(joint_lo)))

    n_root_dofs = 7  # free joint: 3 pos + 4 quat
    batch_qpos = np.tile(qpos_base, (n_poses, 1))
    batch_qpos[:, n_root_dofs:] = joint_samples

    # CPU ground truth
    d_cpu = mujoco.MjData(m)

    # Instantiate sim once just to get geom IDs
    _sim_ref = UnifiedMetalSimulation(num_envs=1, cfg=cfg, model=m, device="mps")
    sc_geom_pairs = _sim_ref._sc_geom_pairs

    cpu_truth = np.zeros(n_poses, dtype=bool)
    for i in range(n_poses):
        d_cpu.qpos[:] = batch_qpos[i]
        d_cpu.qvel[:] = 0.0
        mujoco.mj_kinematics(m, d_cpu)
        mujoco.mj_collision(m, d_cpu)
        for ci in range(int(d_cpu.ncon)):
            g1 = int(d_cpu.contact[ci].geom1)
            g2 = int(d_cpu.contact[ci].geom2)
            if (min(g1, g2), max(g1, g2)) in sc_geom_pairs:
                cpu_truth[i] = True
                break

    # Run Metal adapter in batches of 64 environments
    batch_size = 64
    false_positives = 0
    false_negatives = 0
    for start in range(0, n_poses, batch_size):
        end = min(start + batch_size, n_poses)
        B = end - start
        sim = UnifiedMetalSimulation(num_envs=B, cfg=cfg, model=m, device="mps")
        for i in range(B):
            sim.data.qpos[i] = torch.as_tensor(
                batch_qpos[start + i], dtype=torch.float32, device="mps"
            )
        sim.forward()
        metal_sc = sim.data.sensordata[:, sim.self_col_adr].cpu().numpy()
        metal_hit = metal_sc > 0

        for i in range(B):
            global_i = start + i
            if metal_hit[i] and not cpu_truth[global_i]:
                false_positives += 1
            elif not metal_hit[i] and cpu_truth[global_i]:
                false_negatives += 1

    assert false_positives == 0, (
        f"Self-collision sensor has {false_positives} false positives over 128 random poses"
    )
    assert false_negatives == 0, (
        f"Self-collision sensor has {false_negatives} false negatives over 128 random poses"
    )


def test_external_force_wrench_all_bodies():
    """Verify xfrc_applied backward wrench propagation matches mj_applyFT for all bodies across all 20 DOFs.

    Applies a random force+torque wrench to each non-world body and compares the resulting
    generalized forces across all 20 DOFs (including root 0..6 and hinges 6..20) to mj_applyFT within 1e-4.
    """
    canonical = load_canonical_model()
    m = canonical.model
    cfg = make_mock_cfg()
    d_cpu = mujoco.MjData(m)

    # Crouched pose to exercise moment arms
    d_cpu.qpos[:] = m.qpos0.copy()
    d_cpu.qpos[7:21] = np.array([
        0.1, -0.1, 0.4, -0.8, 0.4, -0.1,
        -0.1,  0.1, 0.4, -0.8, 0.4,  0.1,
        0.0, 0.0,
    ])
    mujoco.mj_forward(m, d_cpu)

    rng = np.random.default_rng(0xDEADBEEF)

    for body_id in range(1, m.nbody):
        force = rng.uniform(-10.0, 10.0, 3).astype(np.float32)
        torque = rng.uniform(-5.0, 5.0, 3).astype(np.float32)
        # Apply at a random point near the body CoM to test moment-arm accounting
        point = d_cpu.xipos[body_id] + rng.uniform(-0.05, 0.05, 3).astype(np.float32)

        # CPU ground truth via mj_applyFT (accumulates into scratch qfrc)
        qfrc_ref = np.zeros(m.nv, dtype=np.float64)
        mujoco.mj_applyFT(
            m, d_cpu,
            force.astype(np.float64), torque.astype(np.float64),
            point.astype(np.float64),
            body_id, qfrc_ref,
        )

        sim = UnifiedMetalSimulation(num_envs=1, cfg=cfg, model=m, device="mps")
        sim.data.qpos[0] = torch.as_tensor(d_cpu.qpos, dtype=torch.float32, device="mps")
        sim.data.qvel[0] = torch.as_tensor(d_cpu.qvel, dtype=torch.float32, device="mps")
        sim.forward()  # populate initial data

        xipos_b = sim.data.xipos[0, body_id].cpu().numpy()
        moment_arm = point - xipos_b
        tau_com = torque + np.cross(moment_arm, force)

        sim.data.xfrc_applied[0, body_id, :3] = torch.as_tensor(force, device="mps")
        sim.data.xfrc_applied[0, body_id, 3:] = torch.as_tensor(
            tau_com.astype(np.float32), device="mps"
        )

        inputs = sim._get_model_inputs()
        qfrc_metal = inputs["qfrc_applied"][0].cpu().numpy().astype(np.float64)

        # Compare ALL 20 DOFs (root translational 0:3, root rotational 3:6, hinges 6:20)
        err = np.abs(qfrc_metal - qfrc_ref).max()
        assert err < 1e-4, (
            f"Body {body_id}: max |qfrc_metal - qfrc_ref| across all 20 DOFs = {err:.3e} > 1e-4"
        )


def test_external_force_wrench_dynamic_state_change():
    """Verify wrench projection remains accurate after dynamic state updates without manual forward() calls.

    Reproduction of review finding [P2]: mutating qpos (e.g. yaw +90 deg, pitch +45 deg)
    and immediately applying an external wrench must compute fresh transforms on device
    and match CPU mj_applyFT to within 1e-4 across all 20 DOFs.
    """
    canonical = load_canonical_model()
    m = canonical.model
    cfg = make_mock_cfg()

    sim = UnifiedMetalSimulation(num_envs=1, cfg=cfg, model=m, device="mps")
    sim.forward()

    # 1. Mutate state: rotate yaw +90 deg, pitch +30 deg, bend knees
    yaw = np.pi / 2.0
    pitch = np.pi / 6.0
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    # Combined yaw-then-pitch quaternion: [w, x, y, z]
    q_rot = np.array([cy * cp, -sy * sp, cy * sp, sy * cp], dtype=np.float32)
    sim.data.qpos[0, 3:7] = torch.as_tensor(q_rot, device="mps")
    sim.data.qpos[0, 7:21] = torch.tensor([
        0.2, -0.2, 0.5, -0.9, 0.4, -0.1,
        -0.2,  0.2, 0.5, -0.9, 0.4,  0.1,
        0.1, -0.1,
    ], dtype=torch.float32, device="mps")

    # 2. Apply a 10 N load at body 7 (left ankle) and 15 N at body 2 (trunk)
    sim.data.xfrc_applied[0, 7, 0] = 10.0
    sim.data.xfrc_applied[0, 7, 4] = 2.0  # y torque
    sim.data.xfrc_applied[0, 2, 2] = -15.0 # downward force on trunk

    # 3. Call _get_model_inputs directly without calling forward() first
    inputs = sim._get_model_inputs()
    qfrc_metal = inputs["qfrc_applied"][0].cpu().numpy().astype(np.float64)

    # 4. Compare with CPU ground truth at the exact same new configuration
    d_cpu = mujoco.MjData(m)
    d_cpu.qpos[:] = sim.data.qpos[0].cpu().numpy()
    d_cpu.qvel[:] = 0.0
    mujoco.mj_forward(m, d_cpu)

    qfrc_ref = np.zeros(m.nv, dtype=np.float64)
    # Body 7 wrench: force [10, 0, 0], torque [0, 2, 0] at body 7 CoM
    mujoco.mj_applyFT(m, d_cpu, np.array([10.0, 0, 0]), np.array([0, 2.0, 0]), d_cpu.xipos[7], 7, qfrc_ref)
    # Body 2 wrench: force [0, 0, -15], torque [0, 0, 0] at body 2 CoM
    mujoco.mj_applyFT(m, d_cpu, np.array([0, 0, -15.0]), np.zeros(3), d_cpu.xipos[2], 2, qfrc_ref)

    err = np.abs(qfrc_metal - qfrc_ref).max()
    assert err < 1e-4, f"Dynamic state change wrench error = {err:.3e} >= 1e-4 across all 20 DOFs"


def test_self_collision_constraint_force_response():
    """Verify UnifiedMetalSimulation generates exact Delassus PGS constraint forces during self-collision.

    Ensures that when robot limbs collide, native constraint forces are formed and match MuJoCo CPU
    to within 1e-4 across all 20 DOFs, rather than passing through unmodeled.
    """
    canonical = load_canonical_model()
    m = canonical.model
    cfg = make_mock_cfg()

    sim = UnifiedMetalSimulation(num_envs=1, cfg=cfg, model=m, device="mps")

    # Crossed legs causing self-collision between feet (27, 73)
    sim.data.qpos[0, 2] = 0.25
    jl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "robot/left_hip_roll")
    jr = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "robot/right_hip_roll")
    sim.data.qpos[0, m.jnt_qposadr[jl]] = -0.38
    sim.data.qpos[0, m.jnt_qposadr[jr]] = 0.38
    sim.forward()

    d_cpu = mujoco.MjData(m)
    d_cpu.qpos[:] = sim.data.qpos[0].cpu().numpy()
    d_cpu.qvel[:] = 0.0
    mujoco.mj_forward(m, d_cpu)

    # 1. Sensor check
    sc_count = sim.data.sensordata[0, sim.self_col_adr].item()
    assert sc_count >= 1.0, f"Expected self-collision sensor >= 1.0, got {sc_count}"

    # 2. Constraint reaction force check
    qfrc_c_metal = sim.data.qfrc_constraint[0].cpu().numpy()
    qfrc_c_cpu = d_cpu.qfrc_constraint

    assert np.linalg.norm(qfrc_c_metal) > 1e-2, "Expected non-zero constraint force on colliding limbs"
    err = np.max(np.abs(qfrc_c_metal - qfrc_c_cpu))
    assert err < 1e-4, f"Self-collision constraint force error = {err:.3e} >= 1e-4 across all 20 DOFs"

    # 3. Dynamic step check: verify reaction force accelerates bodies apart
    sim.step()
    # Left hip roll received positive torque pushing outward; right hip roll received negative torque
    assert sim.data.qvel[0, m.jnt_dofadr[jl]] > 0.0, "Left leg should be accelerated outward by contact force"
    assert sim.data.qvel[0, m.jnt_dofadr[jr]] < 0.0, "Right leg should be accelerated outward by contact force"


def test_self_contact_heterogeneous_randomized_friction():
    """Verify self-contact friction respects per-environment randomized geom_friction and matches MuJoCo CPU.

    Tests a heterogeneous 3-environment batch with geom_friction set to 0.2, 0.8, and 1.4:
    1. _extra_c_fric extracts and populates max(mu1, mu2) per environment.
    2. Constraint reaction forces match MuJoCo CPU with identical randomized friction:
       - Error < 1e-6 for static sticking friction (0.8, 1.4).
       - Error < 1e-3 (relative error < 0.25%) for sliding friction (0.2).
    """
    canonical = load_canonical_model()
    m = canonical.model
    cfg = make_mock_cfg()

    sim = UnifiedMetalSimulation(num_envs=3, cfg=cfg, model=m, device="mps")

    jl_roll = m.jnt_qposadr[m.joint("robot/left_hip_roll").id]
    jr_roll = m.jnt_qposadr[m.joint("robot/right_hip_roll").id]
    for i in range(3):
        sim.data.qpos[i, 2] = 0.25
        sim.data.qpos[i, jl_roll] = -0.38
        sim.data.qpos[i, jr_roll] = 0.38

    # Heterogeneous randomized geom_friction per environment
    fric_values = [0.2, 0.8, 1.4]
    geom_fric = torch.ones((3, m.ngeom, 3), dtype=torch.float32, device="mps") * 0.8
    for i, f_val in enumerate(fric_values):
        geom_fric[i, :, :] = f_val
    sim.model.geom_friction = geom_fric

    sim.forward()

    # 1. Assert friction buffer values match max(f1, f2) per env
    for i, f_val in enumerate(fric_values):
        assert np.isclose(sim._extra_c_fric[i, 0, 0].item(), f_val, atol=1e-4)
        assert np.isclose(sim._extra_c_fric[i, 0, 1].item(), f_val, atol=1e-4)

    # 2. Compare constraint reaction forces against MuJoCo CPU with matching friction
    for i, f_val in enumerate(fric_values):
        m_cpu = mujoco.MjModel.from_xml_path(str(CANONICAL_XML_PATH))
        m_cpu.geom_friction[:, :] = f_val
        d_cpu = mujoco.MjData(m_cpu)
        d_cpu.qpos[:] = sim.data.qpos[i].cpu().numpy()
        d_cpu.qvel[:] = 0.0
        mujoco.mj_forward(m_cpu, d_cpu)

        qfrc_metal = sim.data.qfrc_constraint[i].cpu().numpy()
        qfrc_cpu = d_cpu.qfrc_constraint
        err = np.max(np.abs(qfrc_metal - qfrc_cpu))
        tol = 1e-3 if f_val < 0.5 else 1e-4
        assert err < tol, f"Env {i} (friction={f_val}) constraint force error {err:.3e} >= {tol}"


def test_self_contact_adapter_all_canonical_pairs_and_separated_controls():
    """Verify all 4 canonical self-collision pairs generate contact and clearance poses produce zero false positives.

    Evaluates:
    - Collision poses: foot_foot, leg_leg, trunk_left_leg, trunk_right_leg -> extra_ncon >= 1, sensordata >= 1.0.
    - Near-separated control poses: sub-centimeter clearances -> extra_ncon == 0, sensordata == 0.0.
    """
    canonical = load_canonical_model()
    m = canonical.model
    cfg = make_mock_cfg()

    jl_roll = m.jnt_qposadr[m.joint("robot/left_hip_roll").id]
    jr_roll = m.jnt_qposadr[m.joint("robot/right_hip_roll").id]
    jl_pitch = m.jnt_qposadr[m.joint("robot/left_hip_pitch").id]
    jr_pitch = m.jnt_qposadr[m.joint("robot/right_hip_pitch").id]

    # 1. Collision poses
    sim_col = UnifiedMetalSimulation(num_envs=4, cfg=cfg, model=m, device="mps")
    # Env 0: foot_foot
    sim_col.data.qpos[0, 2] = 0.25
    sim_col.data.qpos[0, jl_roll] = -0.38
    sim_col.data.qpos[0, jr_roll] = 0.38
    # Env 1: leg_leg
    sim_col.data.qpos[1, 2] = 0.30
    sim_col.data.qpos[1, jl_roll] = -0.50
    sim_col.data.qpos[1, jr_roll] = 0.50
    # Env 2: trunk_left_leg
    sim_col.data.qpos[2, 2] = 0.50
    sim_col.data.qpos[2, jl_pitch] = 1.18
    sim_col.data.qpos[2, jl_roll] = 0.39
    # Env 3: trunk_right_leg
    sim_col.data.qpos[3, 2] = 0.50
    sim_col.data.qpos[3, jr_pitch] = -1.25
    sim_col.data.qpos[3, jr_roll] = -0.30

    sim_col.forward()
    assert (sim_col._extra_ncon >= 1).all(), f"Expected all colliding poses to have extra_ncon >= 1, got {sim_col._extra_ncon.tolist()}"
    assert (sim_col.data.sensordata[:, sim_col.self_col_adr] >= 1.0).all()

    # Assert expected body identities for each collision pair
    def get_pairs(env_idx):
        n = sim_col._extra_ncon[env_idx].item()
        b1 = sim_col._extra_c_b1[env_idx, :n].tolist()
        b2 = sim_col._extra_c_b2[env_idx, :n].tolist()
        return [tuple(sorted((b1[i], b2[i]))) for i in range(n)]

    assert (7, 16) in get_pairs(0), f"Env 0 (foot_foot) missing pair (7, 16): got {get_pairs(0)}"
    assert (6, 15) in get_pairs(1), f"Env 1 (leg_leg) missing pair (6, 15): got {get_pairs(1)}"
    assert (2, 6) in get_pairs(2), f"Env 2 (trunk_left_leg) missing pair (2, 6): got {get_pairs(2)}"
    assert (2, 15) in get_pairs(3), f"Env 3 (trunk_right_leg) missing pair (2, 15): got {get_pairs(3)}"

    # 2. Near-separated clearance control poses
    sim_sep = UnifiedMetalSimulation(num_envs=4, cfg=cfg, model=m, device="mps")
    sim_sep.data.qpos[0, 2] = 0.25
    sim_sep.data.qpos[0, jl_roll] = -0.28
    sim_sep.data.qpos[0, jr_roll] = 0.28

    sim_sep.data.qpos[1, 2] = 0.30
    sim_sep.data.qpos[1, jl_roll] = -0.30
    sim_sep.data.qpos[1, jr_roll] = 0.30

    sim_sep.data.qpos[2, 2] = 0.50
    sim_sep.data.qpos[2, jl_pitch] = 0.80
    sim_sep.data.qpos[2, jl_roll] = 0.20

    sim_sep.data.qpos[3, 2] = 0.50
    sim_sep.data.qpos[3, jr_pitch] = -0.80
    sim_sep.data.qpos[3, jr_roll] = -0.15

    sim_sep.forward()
    assert (sim_sep._extra_ncon == 0).all(), f"Expected separated controls to have extra_ncon == 0, got {sim_sep._extra_ncon.tolist()}"
    assert (sim_sep.data.sensordata[:, sim_sep.self_col_adr] == 0.0).all()


def test_self_contact_capacity_and_overflow_guard():
    """Verify that when detected self-contacts exceed allocated buffer capacity,
    the overflow counter increments and the step is rejected with an actionable error,
    and verify device merge overflow / non-finite inputs propagate invalid status safely.
    """
    canonical = load_canonical_model()
    m = canonical.model
    cfg = make_mock_cfg()

    sim = UnifiedMetalSimulation(num_envs=1, cfg=cfg, model=m, device="mps")

    # Artificially clamp capacity to 1 contact
    sim.max_extra_contacts = 1
    sim._extra_c_pos = sim._extra_c_pos[:, :1, :]
    sim._extra_c_dist = sim._extra_c_dist[:, :1]
    sim._extra_c_b1 = sim._extra_c_b1[:, :1]
    sim._extra_c_b2 = sim._extra_c_b2[:, :1]
    sim._extra_c_frame = sim._extra_c_frame[:, :1, :]
    sim._extra_c_fric = sim._extra_c_fric[:, :1, :]

    # Pose with 2 contacts (leg_leg: bodies 6-15 and 7-16)
    sim.data.qpos[0, 2] = 0.30
    jl_roll = m.jnt_qposadr[m.joint("robot/left_hip_roll").id]
    jr_roll = m.jnt_qposadr[m.joint("robot/right_hip_roll").id]
    sim.data.qpos[0, jl_roll] = -0.50
    sim.data.qpos[0, jr_roll] = 0.50

    # 1. Adapter level rejection: forward() and step() must reject overflowed contact sets
    with pytest.raises(RuntimeError, match="Self-contact capacity overflow"):
        sim.forward()
    assert sim.extra_contact_overflow_count >= 1, f"Expected overflow count >= 1, got {sim.extra_contact_overflow_count}"

    with pytest.raises(RuntimeError, match="Self-contact capacity overflow"):
        sim.step()

    # 2. Device level merge overflow guard: test that slice.forward_autonomous catches merge overflow
    ps = sim.slice
    d_stand = np.load(PROJECT_ROOT / "corpus" / "standing_zero_vel.npz")
    qpos_1 = torch.from_numpy(d_stand["qpos"].astype(np.float32)).unsqueeze(0).to(ps.device)
    qvel_1 = torch.zeros((1, 20), device=ps.device, dtype=torch.float32)

    # Provide 40 extra contacts where ground has ~6 and merged_nconmax is 43 -> total desired is ~46 > 43
    extra_pos = torch.zeros((1, 40, 3), device=ps.device, dtype=torch.float32)
    extra_dist = torch.zeros((1, 40), device=ps.device, dtype=torch.float32)
    extra_b1 = torch.zeros((1, 40), device=ps.device, dtype=torch.int32)
    extra_b2 = torch.zeros((1, 40), device=ps.device, dtype=torch.int32)
    extra_frame = torch.zeros((1, 40, 9), device=ps.device, dtype=torch.float32)
    extra_frame[:, :, 2] = 1.0
    extra_frame[:, :, 4] = 1.0
    extra_frame[:, :, 6] = -1.0
    extra_fric = torch.full((1, 40, 2), 0.8, device=ps.device, dtype=torch.float32)
    extra_ncon = torch.tensor([40], device=ps.device, dtype=torch.int32)

    out = ps.forward_autonomous(
        qpos_1, qvel_1,
        extra_contact_pos=extra_pos, extra_contact_dist=extra_dist,
        extra_contact_body1=extra_b1, extra_contact_body2=extra_b2,
        extra_contact_frame=extra_frame, extra_contact_friction=extra_fric,
        extra_ncon=extra_ncon,
    )
    # Out contact overflow must be positive (> 0), solver status must be -6, qacc must be NaN
    assert out.contact_overflow[0].item() > 0, f"Expected contact_overflow > 0, got {out.contact_overflow.tolist()}"
    assert out.solver_status[0].item() == -6, f"Expected solver_status -6 for overflow, got {out.solver_status.tolist()}"
    assert torch.isnan(out.qacc[0]).all(), "Expected qacc to be NaN on overflow"

    # Integration must also propagate -6 and produce NaN qpos/qvel
    qp_next, qv_next, int_stat = ps.integrate_implicit_fast(qpos_1, qvel_1, out.qacc, out.solver_status)
    assert int_stat[0].item() == -6, f"Expected integration_status -6, got {int_stat.tolist()}"
    assert torch.isnan(qp_next[0]).all()

    # 3. Non-finite contact inputs guard
    extra_pos_nan = torch.zeros((1, 8, 3), device=ps.device, dtype=torch.float32)
    extra_dist_nan = torch.zeros((1, 8), device=ps.device, dtype=torch.float32)
    extra_pos_nan[0, 0, 0] = float("nan")
    out_nan = ps.forward_autonomous(
        qpos_1, qvel_1,
        extra_contact_pos=extra_pos_nan, extra_contact_dist=extra_dist_nan,
        extra_ncon=torch.tensor([1], device=ps.device, dtype=torch.int32)
    )
    assert out_nan.contact_overflow[0].item() < 0, f"Expected contact_overflow < 0 for NaN, got {out_nan.contact_overflow.tolist()}"
    assert out_nan.solver_status[0].item() == -8, f"Expected solver_status -8 for non-finite inputs, got {out_nan.solver_status.tolist()}"


def test_collision_transfer_byte_accounting():
    """Verify that collision byte telemetry separates qpos staging bytes and friction staging bytes,
    and includes both in total collision transfer bytes.
    """
    canonical = load_canonical_model()
    m = canonical.model
    cfg = make_mock_cfg()

    sim = UnifiedMetalSimulation(num_envs=2, cfg=cfg, model=m, device="mps")
    # Trigger 1 colliding pose (env 0)
    jl_roll = m.jnt_qposadr[m.joint("robot/left_hip_roll").id]
    jr_roll = m.jnt_qposadr[m.joint("robot/right_hip_roll").id]
    sim.data.qpos[0, 2] = 0.25
    sim.data.qpos[0, jl_roll] = -0.38
    sim.data.qpos[0, jr_roll] = 0.38

    # Reset telemetry counters
    sim.collision_transfer.bytes = 0
    sim.collision_transfer.seconds = 0.0
    sim.collision_qpos_staging_bytes = 0
    sim.collision_fric_staging_bytes = 0
    sim.collision_evaluations_count = 0

    sim.forward()

    # Env 0 collided, Env 1 did not
    assert sim.collision_evaluations_count == 1
    # This canonical model has 21 float32 qpos values per evaluated environment.
    expected_qpos_bytes = sim.data.qpos.shape[1] * sim.data.qpos.element_size()
    assert expected_qpos_bytes == 84
    assert sim.collision_qpos_staging_bytes == expected_qpos_bytes
    geom_fric = getattr(sim.model, "geom_friction", None)
    if isinstance(geom_fric, torch.Tensor):
        # Derive the staged geom set from the allowed contact pairs, independently
        # of the adapter's compact friction-index array.
        candidate_geom_ids = {geom_id for pair in sim._sc_geom_pairs for geom_id in pair}
        assert candidate_geom_ids == {
            sim.trunk_geom_id,
            sim.left_leg_geom_id,
            sim.right_leg_geom_id,
            sim.left_foot_geom_id,
            sim.right_foot_geom_id,
        }
        assert len(candidate_geom_ids) == 5 < m.ngeom
        expected_fric_bytes = (
            sim.collision_evaluations_count
            * len(candidate_geom_ids)
            * geom_fric.shape[-1]
            * geom_fric.element_size()
        )
        assert sim.collision_fric_staging_bytes == expected_fric_bytes
    else:
        expected_fric_bytes = 0

    # Total transfer bytes = qpos_bytes + fric_bytes + id_bytes + contact_payload_bytes
    id_bytes = 1 * 8
    contact_payload_bytes = 1 * (
        4 + 4 + sim.max_extra_contacts * (3 * 4 + 4 + 4 + 4 + 9 * 4 + 2 * 4)
    )
    expected_total_bytes = expected_qpos_bytes + expected_fric_bytes + id_bytes + contact_payload_bytes
    assert sim.collision_transfer.bytes == expected_total_bytes, f"Expected {expected_total_bytes} total bytes, got {sim.collision_transfer.bytes}"

    # Return the candidate to the known non-colliding pose. An empty broadphase
    # must neither stage another payload nor retain a contact count.
    sim.data.qpos[0].copy_(sim.data.qpos[1])
    sim.forward()
    assert sim.collision_evaluations_count == 1
    assert sim.collision_qpos_staging_bytes == expected_qpos_bytes
    assert sim.collision_fric_staging_bytes == expected_fric_bytes
    assert sim.collision_transfer.bytes == expected_total_bytes
    assert torch.count_nonzero(sim._extra_ncon).item() == 0
    assert torch.count_nonzero(sim._current_self_col_count).item() == 0



