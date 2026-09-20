"""Compare the JAX actuator adapter with the locked BAM implementation."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("bam")
import jax.numpy as jnp
import torch
from bam.actuator import TorchBackend
from bam.mjlab import BamActuator
from mjlab_microduck.robot.microduck_constants import actuators
from mjlab_microduck.native_gpu.bam import BamM6


def test_bam_m6_matches_original_torch_controller_and_friction():
    model = BamM6(actuators)
    rng = np.random.default_rng(16)
    values = [rng.uniform(-2, 2, (8, 14)).astype(np.float32) for _ in range(6)]
    position, velocity, target, prev_motor, prev_actuator, external = values
    velocity[0] = 0  # static friction; both directions and drive/backdrive elsewhere
    vin = np.linspace(6.5, 8.2, 8, dtype=np.float32)[:, None]
    kp, kd, fs = [rng.uniform(0.5, 1.5, (8, 1)).astype(np.float32) for _ in range(3)]
    gain = np.linspace(0, 0.2, 8, dtype=np.float32)[:, None]
    output = jax.jit(
        lambda *args: model.compute(
            *args[:6],
            vin=args[6],
            voltage_drop_gain=args[7],
            kp_scale=args[8],
            kd_scale=args[9],
            friction_scale=args[10],
            dt=0.005,
        )
    )(*map(jnp.asarray, [*values, vin, gain, kp, kd, fs]))

    q, dq, goal, previous, previous_applied, load = map(torch.from_numpy, values)
    act = copy.copy(model.model.actuator)
    act.backend = TorchBackend()
    act.vin = torch.clamp(
        torch.from_numpy(vin)
        - torch.from_numpy(gain) * previous.abs().sum(dim=-1, keepdim=True),
        min=actuators.vin_min,
    )
    act.kp = model.model.actuator.kp * torch.from_numpy(kp)
    scaled = dq * torch.from_numpy(kd)
    torque = act.compute_torque(
        act.compute_control(goal, q, scaled, 0.005), True, q, scaled
    )
    s = torch.exp(
        -((dq.abs() / model.model.dtheta_stribeck.value) ** model.model.alpha.value)
    )
    friction = BamActuator._compute_friction_budget(
        SimpleNamespace(_bam_model=model.model),
        previous_applied,
        load,
        s,
    ) * torch.from_numpy(fs)
    np.testing.assert_allclose(
        np.asarray(output.motor_torque), torque.numpy(), atol=2e-6, rtol=2e-5
    )
    np.testing.assert_allclose(
        np.asarray(output.frictionloss), friction.numpy(), atol=2e-6, rtol=2e-5
    )
    np.testing.assert_allclose(np.asarray(output.voltage), act.vin.numpy(), atol=1e-6)
