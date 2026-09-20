"""GPU physics qualification using the real flat-walking scene and BAM controller.

This is a physics probe, not a replacement RL environment: the task's reward,
curriculum, observation noise, terrain rays and randomized resets are not wired
to a runner here. Never use this class as evidence that full training works.
"""

import copy
from typing import NamedTuple

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

from .bam import BamM6
from .contacts import ContactSensors


class PhysicsState(NamedTuple):
    data: object
    history: object
    previous_motor_torque: object
    key: object
    fresh: object


class FlatPhysics:
    def __init__(self, task="Mjlab-Velocity-Flat-MicroDuck"):
        from .collision import register_bounded_collisions

        register_bounded_collisions()
        import mjlab_microduck.tasks  # noqa: F401 - populate task registry.
        from mjlab.tasks.registry import load_env_cfg
        from mjlab.scene import Scene

        if task != "Mjlab-Velocity-Flat-MicroDuck":
            raise NotImplementedError(
                f"GPU physics qualification has not covered {task}"
            )
        self.cfg = load_env_cfg(task)
        self.cfg.scene.num_envs = 1
        # Host-side asset/model compilation; no CPU physics stepping occurs here.
        scene = Scene(self.cfg.scene, "cpu")
        self.host_model = model = scene.compile()
        self.cfg.sim.mujoco.apply(model)
        actuator_cfgs = self.cfg.scene.entities["robot"].articulation.actuators
        if len(actuator_cfgs) != 1:
            raise NotImplementedError("Expected one BAM actuator group")
        self.actuator_cfg = actuator_cfgs[0]
        self.bam = BamM6(self.actuator_cfg)
        if self.actuator_cfg.delay_hold_prob or self.actuator_cfg.delay_update_period:
            raise NotImplementedError(
                "Only the flat task's per-step delay sampling is covered"
            )
        if model.nu != 14 or np.any(
            model.actuator_trntype != mujoco.mjtTrn.mjTRN_JOINT
        ):
            raise NotImplementedError("Expected 14 joint-transmission actuators")
        joints = model.actuator_trnid[:, 0]
        self.dofs = np.asarray(model.jnt_dofadr[joints], dtype=np.int32)
        self.qpos_ids = np.asarray(model.jnt_qposadr[joints], dtype=np.int32)
        self.contacts = ContactSensors(model)
        # MJX allocates DOF-friction rows from a static mask. BAM starts these
        # fields at zero and writes a positive budget at each step. Allocate the
        # rows now, then restore exact values; this changes capacity, not friction.
        allocation_model = copy.copy(model)
        allocation_model.dof_frictionloss[self.dofs] = 1.0
        self.model = self.contacts.put_model(allocation_model).replace(
            dof_frictionloss=jnp.asarray(model.dof_frictionloss, dtype=jnp.float32)
        )
        self.home = jnp.asarray(model.key_qpos[0, self.qpos_ids], dtype=jnp.float32)
        initial = np.asarray(model.key_qpos[0], dtype=np.float32).copy()
        initial[2] += np.mean(self.cfg.events["reset_base"].params["pose_range"]["z"])
        self.initial = mjx.make_data(self.model, impl="jax").replace(
            qpos=jnp.asarray(initial)
        )

    def initial_state(self, seed=0):
        return PhysicsState(
            self.initial,
            jnp.broadcast_to(self.home, (self.actuator_cfg.delay_max_lag + 1, 14)),
            jnp.zeros(14, dtype=jnp.float32),
            jax.random.PRNGKey(seed),
            jnp.array(True),
        )

    def step(self, state, target, vin, voltage_drop_gain, friction_scale):
        """One 5 ms physics step with explicit fixture parameters and GPU delay RNG."""
        data, history, previous_motor_torque, key, fresh = state
        history = jnp.where(fresh, jnp.broadcast_to(target, history.shape), history)
        history = jnp.concatenate((target[None], history[:-1]))
        key, delay_key = jax.random.split(key)
        lag = jax.random.randint(
            delay_key,
            (),
            self.actuator_cfg.delay_min_lag,
            self.actuator_cfg.delay_max_lag + 1,
        )
        delayed = history[lag]
        # Remove our prior friction constraints from the external gearbox load.
        friction_rows = data._impl.efc_type == int(
            mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF
        )
        friction_force = data._impl.efc_J.T @ jnp.where(
            friction_rows, data._impl.efc_force, 0.0
        )
        external = (-data.qfrc_bias + data.qfrc_constraint - friction_force)[self.dofs]
        output = self.bam.compute(
            data.qpos[self.qpos_ids],
            data.qvel[self.dofs],
            delayed,
            previous_motor_torque,
            data.qfrc_actuator[self.dofs],
            external,
            vin=vin,
            voltage_drop_gain=voltage_drop_gain,
            kp_scale=jnp.array(1.0, jnp.float32),
            kd_scale=jnp.array(1.0, jnp.float32),
            friction_scale=friction_scale,
            dt=self.cfg.sim.mujoco.timestep,
        )
        model = self.model.replace(
            dof_frictionloss=self.model.dof_frictionloss.at[self.dofs].set(
                output.frictionloss
            ),
            dof_damping=self.model.dof_damping.at[self.dofs].set(output.damping),
        )
        data = self.contacts.step(model, data.replace(ctrl=output.motor_torque))
        return PhysicsState(data, history, output.motor_torque, key, jnp.array(False))
