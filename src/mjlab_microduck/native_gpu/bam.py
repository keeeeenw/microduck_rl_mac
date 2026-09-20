"""JAX array adapter for the pinned BAM M6 controller used by Microduck.

The firmware voltage law and motor torque equation call BAM's own implementation.
Only its Torch-specific M6 friction expression is translated here and compared to
the original in conformance tests. Delays, random sampling and resets belong to
the environment; this function receives their resolved values explicitly.
"""

import copy
from typing import NamedTuple

import jax.numpy as jnp


class BamOutput(NamedTuple):
    motor_torque: object
    frictionloss: object
    damping: object
    voltage: object


class JaxClamp:
    def clamp(self, value, low, high):
        return jnp.clip(value, low, high)


class BamM6:
    def __init__(self, actuator_cfg):
        from bam.model import load_model
        from bam.actuator import VoltageControlledActuator

        self.cfg = actuator_cfg
        self.model = load_model(actuator_cfg._resolved_json_path)
        model = self.model
        if not all(
            (model.stribeck, model.load_dependent, model.directional, model.quadratic)
        ):
            raise NotImplementedError(
                "The native GPU adapter currently qualifies BAM M6 only"
            )
        if not isinstance(model.actuator, VoltageControlledActuator):
            raise NotImplementedError("Expected the BAM voltage-controlled actuator")
        if actuator_cfg.kp_fw is not None:
            model.actuator.kp = actuator_cfg.kp_fw
        if actuator_cfg.vin is not None:
            model.actuator.vin = actuator_cfg.vin

    def friction(
        self, previous_actuator_torque, external_torque, velocity, friction_scale
    ):
        m = self.model
        s = jnp.exp(
            -jnp.power(jnp.abs(velocity) / m.dtheta_stribeck.value, m.alpha.value)
        )
        friction = m.friction_base.value + s * m.friction_stribeck.value
        friction += jnp.abs(
            external_torque * m.load_friction_external.value
            - previous_actuator_torque * m.load_friction_motor.value
        )
        friction += s * jnp.abs(
            external_torque * m.load_friction_external_stribeck.value
            - previous_actuator_torque * m.load_friction_motor_stribeck.value
        )
        ext, mot = jnp.abs(external_torque), jnp.abs(previous_actuator_torque)
        quadratic = jnp.where(
            mot > ext,
            m.load_friction_external_quad.value * ext**2,
            m.load_friction_motor_quad.value * mot**2,
        )
        return (friction + s * quadratic) * friction_scale

    def compute(
        self,
        position,
        velocity,
        target,
        previous_motor_torque,
        previous_actuator_torque,
        external_torque,
        *,
        vin,
        voltage_drop_gain,
        kp_scale,
        kd_scale,
        friction_scale,
        dt,
    ):
        voltage = vin - voltage_drop_gain * jnp.sum(
            jnp.abs(previous_motor_torque), axis=-1, keepdims=True
        )
        if self.cfg.vin_min is not None:
            voltage = jnp.maximum(voltage, self.cfg.vin_min)
        # Never put tracers into a shared model: one ephemeral controller per trace.
        actuator = copy.copy(self.model.actuator)
        actuator.backend = JaxClamp()
        actuator.vin = voltage
        actuator.kp = self.model.actuator.kp * kp_scale
        scaled_velocity = velocity * kd_scale
        control = actuator.compute_control(target, position, scaled_velocity, dt)
        torque = actuator.compute_torque(control, True, position, scaled_velocity)
        friction = self.friction(
            previous_actuator_torque, external_torque, velocity, friction_scale
        )
        return BamOutput(
            torque,
            friction,
            jnp.asarray(self.model.friction_viscous.value, dtype=jnp.float32),
            voltage,
        )
