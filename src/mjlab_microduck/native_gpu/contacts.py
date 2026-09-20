"""Explicit adapter for the flat task's MuJoCo contact sensor semantics.

MJX 3.10 lacks body/subtree matching and netforce reduction. Keep the original
MuJoCo model intact, reserve its sensor slots in the device model, and evaluate
these sensors from the actual GPU contact solve. Unsupported sensors fail closed.
"""

import copy
import importlib.metadata
from dataclasses import dataclass

import jax.numpy as jnp
import mujoco
from mujoco import mjx
from mujoco.mjx._src import support
import numpy as np


@dataclass(frozen=True)
class ContactSlot:
    address: int
    field: int
    primary: np.ndarray
    secondary: np.ndarray


def _geom_mask(model, kind, identity):
    if kind == mujoco.mjtObj.mjOBJ_UNKNOWN:
        return np.ones(model.ngeom, dtype=bool)
    if kind == mujoco.mjtObj.mjOBJ_GEOM:
        return np.arange(model.ngeom) == identity
    if kind == mujoco.mjtObj.mjOBJ_BODY:
        return model.geom_bodyid == identity
    if kind == mujoco.mjtObj.mjOBJ_XBODY:
        descendant = np.zeros(model.nbody, dtype=bool)
        for body in range(model.nbody):
            parent = body
            while True:
                if parent == identity:
                    descendant[body] = True
                    break
                if parent == 0:
                    break
                parent = model.body_parentid[parent]
        return descendant[model.geom_bodyid]
    raise NotImplementedError(f"Contact matching type {kind} is not qualified")


class ContactSensors:
    def __init__(self, model):
        self.ids = np.flatnonzero(model.sensor_type == mujoco.mjtSensor.mjSENS_CONTACT)
        self.slots = []
        for sensor in self.ids:
            field, reduction, _ = model.sensor_intprm[sensor]
            dimension = model.sensor_dim[sensor]
            if not (
                (field == 1 and dimension == 1 and reduction in (0, 3))
                or (field == 2 and dimension == 3 and reduction == 3)
            ):
                raise NotImplementedError(
                    f"Contact sensor {model.sensor(sensor).name}: only single-slot found "
                    "and netforce force are qualified candidates"
                )
            if model.sensor_cutoff[sensor] != 0:
                raise NotImplementedError("Contact sensor cutoff is not qualified")
            self.slots.append(
                ContactSlot(
                    int(model.sensor_adr[sensor]),
                    int(field),
                    _geom_mask(
                        model, model.sensor_objtype[sensor], model.sensor_objid[sensor]
                    ),
                    _geom_mask(
                        model, model.sensor_reftype[sensor], model.sensor_refid[sensor]
                    ),
                )
            )

    def put_model(self, model):
        """Compile unchanged dynamics; only contact sensor evaluation moves to this adapter."""
        if (
            mujoco.__version__ != "3.10.0"
            or importlib.metadata.version("mujoco-mjx") != "3.10.0"
        ):
            raise RuntimeError("Contact adapter is pinned to MuJoCo/MJX 3.10.0")
        device_source = copy.copy(model)
        device_source.sensor_objtype[self.ids] = mujoco.mjtObj.mjOBJ_UNKNOWN
        device_source.sensor_reftype[self.ids] = mujoco.mjtObj.mjOBJ_UNKNOWN
        device_source.sensor_objid[self.ids] = -1
        device_source.sensor_refid[self.ids] = -1
        device_source.sensor_intprm[self.ids, 1] = 0
        result = mjx.put_model(device_source, impl="jax")
        stages = result.sensor_needstage.copy()
        stages[self.ids] = mujoco.mjtStage.mjSTAGE_NONE
        return result.replace(sensor_needstage=stages)

    def evaluate(self, model, data):
        contact = data._impl.contact
        forces = jnp.zeros((contact.geom.shape[0], 6), dtype=jnp.float32)
        if any(slot.field == 2 for slot in self.slots):
            for dim in set(contact.dim):
                values, ids = support.contact_force_dim(model, data, int(dim))
                forces = forces.at[ids].set(values)
        # MuJoCo contact coordinates are (normal, tangent1, tangent2).
        world_force = jnp.einsum("ni,nij->nj", forces[:, :3], contact.frame)
        active = contact.dist < contact.includemargin
        g0, g1 = contact.geom[:, 0], contact.geom[:, 1]
        output = data.sensordata
        for slot in self.slots:
            primary, secondary = jnp.asarray(slot.primary), jnp.asarray(slot.secondary)
            forward = primary[g0] & secondary[g1]
            reverse = primary[g1] & secondary[g0]
            match = active & (forward | reverse)
            if slot.field == 1:
                output = output.at[slot.address].set(jnp.sum(match).astype(jnp.float32))
            else:
                direction = jnp.where(forward, 1.0, -1.0)
                force = jnp.sum(
                    jnp.where(match[:, None], direction[:, None] * world_force, 0.0),
                    axis=0,
                )
                output = output.at[slot.address : slot.address + 3].set(force)
        return data.replace(sensordata=output)

    def forward(self, model, data):
        return self.evaluate(model, mjx.forward(model, data))

    def step(self, model, data):
        return self.evaluate(model, mjx.step(model, data))
