"""PyTorch / Metal BAM M6 actuator controller for Microduck.

Vectorized on MPS/CPU with exact parameter fidelity to BAM M6 (XL330)
and bitwise-identical formulas to bam.mjlab.BamActuator and native_gpu.bam.BamM6.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional, Union
import torch

from bam.model import _resolve_json_path, load_model


class BamOutput(NamedTuple):
    motor_torque: torch.Tensor       # (B, 14)
    frictionloss: torch.Tensor       # (B, 14)
    damping: float                   # viscous damping coefficient
    voltage: torch.Tensor            # (B, 1)


@dataclass
class BamSubstepResult:
    motor_torque: torch.Tensor       # (B, 14)
    dof_frictionloss: torch.Tensor   # (B, 20)
    damping: float
    voltage: torch.Tensor            # (B, 1)
    external_torque: torch.Tensor    # (B, 14)


class BamM6Controller:
    """Vectorized PyTorch BAM M6 actuator controller for Microduck's 14 actuated joints.

    Actuated joints correspond to:
    - qpos[7:21] (indices 7 to 20 inclusive in the 21-dim qpos)
    - qvel[6:20] (indices 6 to 19 inclusive in the 20-dim qvel)
    - dof_frictionloss[6:20]
    - qfrc_actuator[6:20]
    """

    def __init__(
        self,
        motor_name: str = "xl330",
        model: str = "m6",
        json_path: Optional[str] = None,
        kp_fw: float = 200.0,
        vin_nominal: float = 7.5,
        vin_min: float = 6.0,
        vin_drop_gain: float = 0.0,
        device: Union[str, torch.device] = "mps",
    ):
        path = _resolve_json_path(json_path, motor_name, model)
        self.bam_model = load_model(path)
        m = self.bam_model
        act = m.actuator

        self.device = torch.device(device)
        self.motor_name = motor_name
        self.model_name = model
        self.json_path = path

        # Physical constants from BAM model
        self.kt = float(m.kt.value)
        self.R = float(m.R.value)
        self.armature = float(m.armature.value)
        self.friction_viscous = float(m.friction_viscous.value)
        self.friction_base = float(m.friction_base.value)
        self.friction_stribeck = float(m.friction_stribeck.value)
        self.alpha = float(m.alpha.value)
        self.dtheta_stribeck = float(m.dtheta_stribeck.value)
        self.load_friction_external = float(m.load_friction_external.value)
        self.load_friction_motor = float(m.load_friction_motor.value)
        self.load_friction_external_stribeck = float(m.load_friction_external_stribeck.value)
        self.load_friction_motor_stribeck = float(m.load_friction_motor_stribeck.value)
        self.load_friction_external_quad = float(m.load_friction_external_quad.value)
        self.load_friction_motor_quad = float(m.load_friction_motor_quad.value)

        # Firmware & Electrical parameters
        self.error_gain = float(act.error_gain)
        self.max_pwm = float(act.max_pwm)
        self.max_current = float(act.max_current) if getattr(act, "max_current", None) is not None else None
        self.kp_fw = float(kp_fw)
        self.vin_nominal = float(vin_nominal)
        self.vin_min = float(vin_min)
        self.default_vin_drop_gain = float(vin_drop_gain)

    def to(self, device: Union[str, torch.device]) -> "BamM6Controller":
        self.device = torch.device(device)
        return self

    def compute(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor,
        target: torch.Tensor,
        previous_motor_torque: torch.Tensor,
        previous_actuator_torque: torch.Tensor,
        external_torque: torch.Tensor,
        *,
        vin: Optional[torch.Tensor] = None,
        voltage_drop_gain: Optional[Union[float, torch.Tensor]] = None,
        kp_scale: Union[float, torch.Tensor] = 1.0,
        kd_scale: Union[float, torch.Tensor] = 1.0,
        friction_scale: Union[float, torch.Tensor] = 1.0,
        dt: float = 0.005,
    ) -> BamOutput:
        """Vectorized BAM M6 torque & friction computation.

        Args:
            position: (B, 14) current actuated joint angles [rad]
            velocity: (B, 14) current actuated joint velocities [rad/s]
            target: (B, 14) desired actuated joint positions [rad]
            previous_motor_torque: (B, 14) motor torque from previous step [Nm]
            previous_actuator_torque: (B, 14) actuator torque from previous step [Nm]
            external_torque: (B, 14) external torque on gearbox from previous step [Nm]
            vin: optional (B, 1) or float per-world nominal battery voltage [V]
            voltage_drop_gain: optional (B, 1) or float internal resistance gain [V/Nm]
            kp_scale: optional (B, 1) or float gain multiplier
            kd_scale: optional (B, 1) or float velocity / back-EMF multiplier
            friction_scale: optional (B, 1) or float friction budget multiplier
            dt: substep timestep [s] (default 0.005)
        Returns:
            BamOutput(motor_torque, frictionloss, damping, voltage)
        """
        B = position.shape[0]
        dev = position.device

        # 1. Effective supply voltage with load-dependent sag
        if vin is None:
            v_base = torch.full((B, 1), self.vin_nominal, dtype=torch.float32, device=dev)
        elif isinstance(vin, (int, float)):
            v_base = torch.full((B, 1), float(vin), dtype=torch.float32, device=dev)
        else:
            v_base = vin.to(device=dev, dtype=torch.float32)
            if v_base.ndim == 1:
                v_base = v_base.unsqueeze(-1)

        v_drop = self.default_vin_drop_gain if voltage_drop_gain is None else voltage_drop_gain
        if isinstance(v_drop, (int, float)):
            v_drop_tensor = torch.full((B, 1), float(v_drop), dtype=torch.float32, device=dev)
        else:
            v_drop_tensor = v_drop.to(device=dev, dtype=torch.float32)
            if v_drop_tensor.ndim == 1:
                v_drop_tensor = v_drop_tensor.unsqueeze(-1)

        if torch.any(v_drop_tensor > 0.0):
            load = torch.sum(torch.abs(previous_motor_torque), dim=-1, keepdim=True)
            v_eff = v_base - v_drop_tensor * load
            if self.vin_min is not None:
                v_eff = torch.clamp(v_eff, min=self.vin_min)
        else:
            v_eff = v_base

        # 2. Firmware PD control law
        scaled_vel = velocity * kd_scale
        duty = (target - position) * (self.kp_fw * kp_scale) * self.error_gain

        if self.max_current is not None:
            duty_span = self.R * self.max_current / v_eff
            duty_center = (self.kt * scaled_vel) / v_eff
            duty = torch.clamp(duty, duty_center - duty_span, duty_center + duty_span)

        duty = torch.clamp(duty, -self.max_pwm, self.max_pwm)
        v_control = v_eff * duty

        # 3. DC motor equation with back-EMF
        motor_torque = (self.kt * v_control / self.R) - ((self.kt ** 2) * scaled_vel / self.R)

        # Ceiling clamp to forcerange at effective voltage
        force_limit = v_eff * (self.kt / self.R)
        motor_torque = torch.clamp(motor_torque, -force_limit, force_limit)

        # 4. Stribeck velocity factor
        abs_vel = torch.abs(velocity)
        stribeck = torch.exp(-torch.pow(abs_vel / self.dtheta_stribeck, self.alpha))

        # 5. BAM M6 friction budget
        friction = self.friction_base + stribeck * self.friction_stribeck
        gearbox = torch.abs(
            external_torque * self.load_friction_external
            - previous_actuator_torque * self.load_friction_motor
        )
        friction = friction + gearbox

        gearbox_stribeck = torch.abs(
            external_torque * self.load_friction_external_stribeck
            - previous_actuator_torque * self.load_friction_motor_stribeck
        )
        friction = friction + stribeck * gearbox_stribeck

        abs_ext = torch.abs(external_torque)
        abs_mot = torch.abs(previous_actuator_torque)
        drive_mask = (abs_mot > abs_ext).to(dtype=torch.float32)
        quad_term = (
            drive_mask * self.load_friction_external_quad * (abs_ext ** 2)
            + (1.0 - drive_mask) * self.load_friction_motor_quad * (abs_mot ** 2)
        )
        friction = friction + stribeck * quad_term

        frictionloss = friction * friction_scale

        return BamOutput(
            motor_torque=motor_torque,
            frictionloss=frictionloss,
            damping=self.friction_viscous,
            voltage=v_eff,
        )

    def compute_substep(
        self,
        qpos: torch.Tensor,
        qvel: torch.Tensor,
        target: torch.Tensor,
        previous_motor_torque: torch.Tensor,
        previous_actuator_torque: torch.Tensor,
        qfrc_bias: torch.Tensor,
        qfrc_constraint: torch.Tensor,
        efc_type: torch.Tensor,
        efc_id: torch.Tensor,
        efc_force: torch.Tensor,
        nefc: torch.Tensor,
        *,
        vin: Optional[torch.Tensor] = None,
        voltage_drop_gain: Optional[Union[float, torch.Tensor]] = None,
        kp_scale: Union[float, torch.Tensor] = 1.0,
        kd_scale: Union[float, torch.Tensor] = 1.0,
        friction_scale: Union[float, torch.Tensor] = 1.0,
        dt: float = 0.005,
    ) -> BamSubstepResult:
        """Evaluates one substep of BAM control from full simulation states.

        Extracts actuated joints (qpos[7:21], qvel[6:20]), strips friction constraint
        forces from qfrc_constraint using efc, computes external load on gearboxes,
        and packs output frictionloss into a full (B, 20) tensor.
        """
        B = qpos.shape[0]
        dev = qpos.device

        pos = qpos[:, 7:21]
        vel = qvel[:, 6:20]

        # Scan active constraint rows to extract self-friction force per DOF
        n_max = efc_type.shape[1]
        valid = torch.arange(n_max, device=dev).unsqueeze(0) < nefc.unsqueeze(1)
        is_fric = valid & (efc_type == 1) # mjCNSTR_FRICTION_DOF = 1

        contrib = torch.where(is_fric, efc_force, torch.zeros_like(efc_force))
        idx = torch.where(is_fric, efc_id, torch.zeros_like(efc_id)).long()

        qfrc_friction = torch.zeros(B, 20, dtype=efc_force.dtype, device=dev)
        qfrc_friction.scatter_add_(1, idx, contrib)

        # External torque on actuated joints (excludes self friction)
        external_torque = (
            -qfrc_bias[:, 6:20]
            + qfrc_constraint[:, 6:20]
            - qfrc_friction[:, 6:20]
        )

        out = self.compute(
            pos,
            vel,
            target,
            previous_motor_torque,
            previous_actuator_torque,
            external_torque,
            vin=vin,
            voltage_drop_gain=voltage_drop_gain,
            kp_scale=kp_scale,
            kd_scale=kd_scale,
            friction_scale=friction_scale,
            dt=dt,
        )

        dof_fl = torch.zeros(B, 20, dtype=torch.float32, device=dev)
        dof_fl[:, 6:20] = out.frictionloss

        return BamSubstepResult(
            motor_torque=out.motor_torque,
            dof_frictionloss=dof_fl,
            damping=out.damping,
            voltage=out.voltage,
            external_torque=external_torque,
        )
