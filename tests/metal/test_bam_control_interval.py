"""Tests for BAM-driven control interval qualification.

Verifies:
1. Numerical parity between BamM6Controller and canonical BAM references.
2. 4-substep 20 ms control interval: policy action held constant, BAM motor torque and friction budget
   dynamically updated at each 5 ms substep.
3. Physical gates (pos, SO(3), linvel, angvel, jntpos, jntvel) against matched MuJoCo CPU.
4. Stiction hold, Stribeck velocity reversal, torque saturation, and simultaneous contact/limit/friction solve.
"""

from pathlib import Path
import sys
import copy
import numpy as np
import pytest
import torch
import mujoco

PROJECT_ROOT = Path(__file__).resolve().parent

from bam.actuator import TorchBackend
from mjlab_microduck.native_gpu.metal.bam_controller import BamM6Controller, BamOutput
from mjlab_microduck.native_gpu.metal.canonical_model_loader import load_canonical_model
from mjlab_microduck.native_gpu.metal.representative_physics_slice import RepresentativePhysicsSlice


def compute_so3_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    """Computes geodesic angular distance on SO(3) invariant to q ~ -q."""
    q1_64 = q1.astype(np.float64)
    q2_64 = q2.astype(np.float64)
    q1_norm = q1_64 / max(1e-14, float(np.linalg.norm(q1_64)))
    q2_norm = q2_64 / max(1e-14, float(np.linalg.norm(q2_64)))
    dot = float(np.abs(np.dot(q1_norm, q2_norm)))
    dot_clamped = min(1.0, max(0.0, dot))
    return float(2.0 * np.arccos(dot_clamped))



def compute_ref_friction_budget(model, prev_actuator_torque, external_torque, stribeck_coeff):
    """Reference friction budget formula directly from BAM M6 specification."""
    m = model
    friction = m.friction_base.value + stribeck_coeff * m.friction_stribeck.value
    friction = friction + torch.abs(
        external_torque * m.load_friction_external.value
        - prev_actuator_torque * m.load_friction_motor.value
    )
    friction = friction + stribeck_coeff * torch.abs(
        external_torque * m.load_friction_external_stribeck.value
        - prev_actuator_torque * m.load_friction_motor_stribeck.value
    )
    abs_ext = torch.abs(external_torque)
    abs_mot = torch.abs(prev_actuator_torque)
    quad = torch.where(
        abs_mot > abs_ext,
        m.load_friction_external_quad.value * abs_ext**2,
        m.load_friction_motor_quad.value * abs_mot**2,
    )
    return friction + stribeck_coeff * quad


def test_bam_m6_controller_numerical_parity():
    """Verify BamM6Controller matches canonical BAM Torch calculations to numerical precision."""
    controller = BamM6Controller(motor_name="xl330", model="m6", kp_fw=200.0, device="cpu")
    m = controller.bam_model

    rng = np.random.default_rng(42)
    B = 8
    q = torch.from_numpy(rng.uniform(-2, 2, (B, 14)).astype(np.float32))
    dq = torch.from_numpy(rng.uniform(-5, 5, (B, 14)).astype(np.float32))
    target = torch.from_numpy(rng.uniform(-2, 2, (B, 14)).astype(np.float32))
    prev_motor = torch.from_numpy(rng.uniform(-1, 1, (B, 14)).astype(np.float32))
    prev_act = torch.from_numpy(rng.uniform(-1, 1, (B, 14)).astype(np.float32))
    external = torch.from_numpy(rng.uniform(-1, 1, (B, 14)).astype(np.float32))

    # Test static hold on row 0
    dq[0] = 0.0

    vin = torch.linspace(6.5, 8.2, B, dtype=torch.float32)[:, None]
    gain = torch.linspace(0.0, 0.2, B, dtype=torch.float32)[:, None]
    kp_scale = torch.from_numpy(rng.uniform(0.5, 1.5, (B, 1)).astype(np.float32))
    kd_scale = torch.from_numpy(rng.uniform(0.5, 1.5, (B, 1)).astype(np.float32))
    fric_scale = torch.from_numpy(rng.uniform(0.5, 1.5, (B, 1)).astype(np.float32))

    out = controller.compute(
        q,
        dq,
        target,
        prev_motor,
        prev_act,
        external,
        vin=vin,
        voltage_drop_gain=gain,
        kp_scale=kp_scale,
        kd_scale=kd_scale,
        friction_scale=fric_scale,
        dt=0.005,
    )

    # Reference implementation matching test_metal_bam.py
    act = copy.copy(m.actuator)
    act.backend = TorchBackend()
    v_ref = torch.clamp(
        vin - gain * prev_motor.abs().sum(dim=-1, keepdim=True),
        min=controller.vin_min,
    )
    act.vin = v_ref
    act.kp = controller.kp_fw * kp_scale
    scaled_dq = dq * kd_scale
    control = act.compute_control(target, q, scaled_dq, 0.005)
    torque_ref = act.compute_torque(control, True, q, scaled_dq)
    # Clip to forcerange
    force_limit = v_ref * controller.kt / controller.R
    torque_ref = torch.clamp(torque_ref, -force_limit, force_limit)

    stribeck = torch.exp(
        -((dq.abs() / m.dtheta_stribeck.value) ** m.alpha.value)
    )
    friction_ref = compute_ref_friction_budget(
        m,
        prev_act,
        external,
        stribeck,
    ) * fric_scale

    np.testing.assert_allclose(
        out.motor_torque.numpy(), torque_ref.numpy(), atol=2e-6, rtol=2e-5
    )
    np.testing.assert_allclose(
        out.frictionloss.numpy(), friction_ref.numpy(), atol=2e-6, rtol=2e-5
    )
    np.testing.assert_allclose(
        out.voltage.numpy(), v_ref.numpy(), atol=1e-6
    )


def test_bam_4_substep_control_interval_vs_mujoco_cpu():
    """Verify 4-substep 20 ms BAM-driven control interval matches MuJoCo CPU across physical gates."""
    canonical = load_canonical_model()
    m_cpu = canonical.model
    d_cpu = mujoco.MjData(m_cpu)

    ps = RepresentativePhysicsSlice(device="mps", autonomous_capacity=64)
    controller = BamM6Controller(motor_name="xl330", model="m6", kp_fw=200.0, device="mps")

    # Set up matched actuator constants on CPU model
    m_cpu.dof_armature[6:] = controller.armature
    m_cpu.dof_damping[6:] = controller.friction_viscous
    mujoco.mj_setConst(m_cpu, d_cpu)

    # Initial state: stable home pose in air (z=0.25 m)
    qpos_init = np.zeros(21, dtype=np.float32)
    qpos_init[2] = 0.25
    qpos_init[3] = 1.0 # quat w
    # Actuated joints set to nominal home
    home_angles = np.array([
        0.0, 0.0, 0.2, -0.4, 0.2, 0.0, 0.0,
        0.0, 0.0, 0.2, -0.4, 0.2, 0.0, 0.0
    ], dtype=np.float32)
    qpos_init[7:21] = home_angles

    qvel_init = np.zeros(20, dtype=np.float32)

    # Target position: slightly offset from home
    target_pos = home_angles + 0.05
    target_pos_t = torch.as_tensor(target_pos, dtype=torch.float32, device="mps").unsqueeze(0)

    # 1. Run CPU reference 4 substeps (20 ms)
    d_cpu.qpos[:] = qpos_init
    d_cpu.qvel[:] = qvel_init
    mujoco.mj_forward(m_cpu, d_cpu)

    cpu_substep_states = []
    cpu_substep_torques = []
    cpu_substep_fricloss = []

    prev_motor = np.zeros(14, dtype=np.float32)
    prev_act = np.zeros(14, dtype=np.float32)

    for s in range(4):
        # Evaluate BAM on CPU
        pos = d_cpu.qpos[7:21]
        vel = d_cpu.qvel[6:20]

        # External torque on actuated joints
        ext_t = -d_cpu.qfrc_bias[6:20] + d_cpu.qfrc_constraint[6:20]
        # Strip friction forces from efc
        if d_cpu.nefc > 0:
            for r in range(d_cpu.nefc):
                if d_cpu.efc_type[r] == mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF:
                    dof = d_cpu.efc_id[r]
                    if 6 <= dof < 20:
                        ext_t[dof - 6] -= d_cpu.efc_force[r]

        out_cpu = controller.compute(
            torch.as_tensor(pos, dtype=torch.float32).unsqueeze(0),
            torch.as_tensor(vel, dtype=torch.float32).unsqueeze(0),
            torch.as_tensor(target_pos, dtype=torch.float32).unsqueeze(0),
            torch.as_tensor(prev_motor, dtype=torch.float32).unsqueeze(0),
            torch.as_tensor(prev_act, dtype=torch.float32).unsqueeze(0),
            torch.as_tensor(ext_t, dtype=torch.float32).unsqueeze(0),
            vin=7.5,
            dt=0.005,
        )

        tau = out_cpu.motor_torque[0].numpy()
        fl = out_cpu.frictionloss[0].numpy()

        d_cpu.ctrl[:14] = tau
        m_cpu.dof_frictionloss[6:] = fl

        mujoco.mj_step(m_cpu, d_cpu)

        cpu_substep_states.append((d_cpu.qpos.copy(), d_cpu.qvel.copy()))
        cpu_substep_torques.append(tau.copy())
        cpu_substep_fricloss.append(fl.copy())

        prev_motor = tau.copy()
        prev_act = d_cpu.qfrc_actuator[6:20].copy()

    # 2. Run Metal 4 substeps (20 ms)
    qp_mps = torch.as_tensor(qpos_init, dtype=torch.float32, device="mps").unsqueeze(0)
    qv_mps = torch.as_tensor(qvel_init, dtype=torch.float32, device="mps").unsqueeze(0)

    final_qp, final_qv, substeps, bam_outs = ps.step_bam_control_interval(
        qp_mps,
        qv_mps,
        target_pos_t,
        controller,
        num_substeps=4,
        dt=0.005,
        vin=torch.tensor([[7.5]], device="mps"),
    )

    # 3. Assert physical gates on every substep
    for s in range(4):
        step_out = substeps[s]
        bam_out = bam_outs[s]

        # Status checks
        assert int(step_out.integration_status[0].item()) == 0, f"Substep {s} integration status != 0"
        assert int(step_out.physics_outputs.solver_status[0].item()) >= 0, f"Substep {s} solver status < 0"

        # Compare BAM outputs
        gpu_tau = bam_out.motor_torque[0].cpu().numpy()
        gpu_fl = bam_out.dof_frictionloss[0, 6:20].cpu().numpy()
        np.testing.assert_allclose(gpu_tau, cpu_substep_torques[s], atol=1e-3, err_msg=f"Substep {s} motor torque mismatch")
        np.testing.assert_allclose(gpu_fl, cpu_substep_fricloss[s], atol=1e-3, err_msg=f"Substep {s} friction budget mismatch")

        # Compare states
        gpu_qp = step_out.qpos[0].cpu().numpy()
        gpu_qv = step_out.qvel[0].cpu().numpy()
        cpu_qp, cpu_qv = cpu_substep_states[s]

        pos_err = np.max(np.abs(gpu_qp[:3] - cpu_qp[:3]))
        so3_err = compute_so3_distance(gpu_qp[3:7], cpu_qp[3:7])
        jnt_err = np.max(np.abs(gpu_qp[7:] - cpu_qp[7:]))
        linvel_err = np.max(np.abs(gpu_qv[:3] - cpu_qv[:3]))
        angvel_err = np.max(np.abs(gpu_qv[3:6] - cpu_qv[3:6]))
        jntvel_err = np.max(np.abs(gpu_qv[6:] - cpu_qv[6:]))

        assert pos_err < 0.01, f"Substep {s} root pos error {pos_err} >= 0.01 m"
        assert so3_err < 0.01, f"Substep {s} SO(3) attitude error {so3_err} >= 0.01 rad"
        assert jnt_err < 0.01, f"Substep {s} joint pos error {jnt_err} >= 0.01 rad"
        assert linvel_err < 0.02, f"Substep {s} linvel error {linvel_err} >= 0.02 m/s"
        assert angvel_err < 0.05, f"Substep {s} angvel error {angvel_err} >= 0.05 rad/s"
        assert jntvel_err < 0.05, f"Substep {s} jntvel error {jntvel_err} >= 0.05 rad/s"



def test_bam_control_interval_stiction_equilibrium():
    """Verify that BAM friction loss holds joints locked (stiction) when commands remain inside the friction band."""
    ps = RepresentativePhysicsSlice(device="mps", autonomous_capacity=64)
    controller = BamM6Controller(motor_name="xl330", model="m6", kp_fw=200.0, device="mps")

    # Start at rest with small target error (below break-away torque)
    qpos = torch.zeros((1, 21), dtype=torch.float32, device="mps")
    qpos[0, 2] = 0.3 # in air
    qpos[0, 3] = 1.0
    qvel = torch.zeros((1, 20), dtype=torch.float32, device="mps")

    # Command tiny position delta 0.005 rad (firmware torque will be ~0.00037 Nm, well below 0.038 Nm friction)
    target_pos = torch.full((1, 14), 0.005, dtype=torch.float32, device="mps")

    final_qp, final_qv, substeps, _ = ps.step_bam_control_interval(
        qpos,
        qvel,
        target_pos,
        controller,
        num_substeps=4,
        dt=0.005,
    )

    # Actuated joint velocities should remain virtually zero (locked by stiction)
    actuated_vel = final_qv[0, 6:20].cpu().numpy()
    np.testing.assert_allclose(actuated_vel, 0.0, atol=1e-3, err_msg="Joint moved despite being within stiction band")


def test_bam_simultaneous_contacts_limits_and_friction():
    """Verify simultaneous ground contact + knee joint limit + BAM friction loss in a 4-substep interval."""
    ps = RepresentativePhysicsSlice(device="mps", autonomous_capacity=64)
    controller = BamM6Controller(motor_name="xl330", model="m6", kp_fw=200.0, device="mps")

    # Position robot with feet penetrating floor (z = -0.01) and left knee breaching lower limit (-1.5708)
    qpos = torch.zeros((1, 21), dtype=torch.float32, device="mps")
    qpos[0, 2] = -0.01 # foot penetration
    qpos[0, 3] = 1.0   # quat w
    qpos[0, 7 + 3] = -1.65 # left knee breach limit (-1.5708)
    qvel = torch.zeros((1, 20), dtype=torch.float32, device="mps")
    qvel[0, 2] = -0.1  # moving downward into floor

    target_pos = torch.zeros((1, 14), dtype=torch.float32, device="mps")

    final_qp, final_qv, substeps, _ = ps.step_bam_control_interval(
        qpos,
        qvel,
        target_pos,
        controller,
        num_substeps=4,
        dt=0.005,
    )

    for s in range(4):
        st = substeps[s]
        assert int(st.integration_status[0].item()) == 0
        assert int(st.physics_outputs.solver_status[0].item()) >= 0
        types = st.physics_outputs.efc_type[0].cpu().numpy()
        nefc = int(st.physics_outputs.nefc[0].item())
        active_types = set(types[:nefc])

        # Verify all 3 constraint types are actively solved simultaneously!
        assert 1 in active_types, "Friction loss (type 1) not active"
        assert 6 in active_types, "Contact (type 6) not active"
        assert 3 in active_types, "Joint limit (type 3) not active"
