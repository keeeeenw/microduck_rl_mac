"""MJX/Metal physics adapter for the original mjlab Torch managers.

All numerical work is on MPS. The pinned PJRT plugin does not expose a device
pointer/DLPack export, so framework transfers currently stage through host memory.
These are copies, not CPU physics/learning. Transfer bytes/time are measured.
"""

import copy
from types import SimpleNamespace
import time

import jax
from jax import numpy as jnp
import mujoco
from mujoco import mjx
from mujoco.mjx._src import smooth, support
import numpy as np
import torch
from mujoco_warp._src.types import Model as WarpModelSchema

from .contacts import ContactSensors
from .collision import register_bounded_collisions
from .linalg import register_mps_solver_lowerings


class Transfer:
    def __init__(self):
        self.bytes = 0
        self.seconds = 0.0

    def to_torch(self, value):
        start = time.perf_counter()
        array = np.asarray(value)
        self.bytes += array.nbytes
        result = torch.from_numpy(array.copy()).to("mps")
        self.seconds += time.perf_counter() - start
        return result

    def to_jax(self, value):
        start = time.perf_counter()
        array = value.detach().cpu().numpy()
        self.bytes += array.nbytes
        result = jnp.asarray(array)
        self.seconds += time.perf_counter() - start
        return result


class ModelView:
    def __init__(self, model, n):
        self._host = model
        self._n = n
        self._fields = {}

    def __getattr__(self, name):
        if name in self._fields:
            return self._fields[name]
        value = getattr(self._host, name)
        if isinstance(value, np.ndarray):
            dtype = torch.float32 if value.dtype.kind == "f" else None
            tensor = torch.as_tensor(value.copy(), dtype=dtype, device="mps")
            shape = getattr(WarpModelSchema.__annotations__.get(name), "shape", ())
            self._fields[name] = (
                tensor.unsqueeze(0).expand((self._n,) + tensor.shape).clone()
                if shape and shape[0] == "*"
                else tensor
            )
            return self._fields[name]
        return value


class MetalSimulation:
    def __init__(self, num_envs, cfg, model, device):
        if str(device) != "mps" or not torch.backends.mps.is_available():
            raise RuntimeError("Native Metal simulation requires MPS; no CPU fallback")
        register_mps_solver_lowerings()
        register_bounded_collisions()
        self.cfg, self.device, self.num_envs = cfg, "mps", num_envs
        self.mj_model = model
        cfg.mujoco.apply(model)
        self.mj_data = mujoco.MjData(model)  # allocation only, no CPU stepping
        self.model = ModelView(model, num_envs)
        self.transfer = Transfer()
        self.expanded_fields = set()
        self.default_model_fields = {}
        self._sensors = []
        self._contacts = ContactSensors(model)
        allocation = copy.copy(model)
        self._friction_dofs = model.jnt_dofadr[model.actuator_trnid[:, 0]]
        allocation.dof_frictionloss[self._friction_dofs] = 1.0
        self._base = self._contacts.put_model(allocation).replace(
            dof_frictionloss=jnp.asarray(model.dof_frictionloss, dtype=jnp.float32)
        )
        self._initial = mjx.make_data(self._base, impl="jax")
        self._state = jax.tree.map(
            lambda a: jnp.broadcast_to(a, (num_envs,) + a.shape), self._initial
        )
        self.data = SimpleNamespace(nworld=num_envs)
        self._output_fields = (
            "time",
            "qpos",
            "qvel",
            "qacc",
            "qacc_warmstart",
            "ctrl",
            "act",
            "qfrc_applied",
            "xfrc_applied",
            "qfrc_actuator",
            "qfrc_bias",
            "qfrc_constraint",
            "xpos",
            "xquat",
            "xmat",
            "xipos",
            "ximat",
            "geom_xpos",
            "geom_xmat",
            "site_xpos",
            "site_xmat",
            "cvel",
            "cacc",
            "subtree_com",
            "subtree_linvel",
            "subtree_angmom",
            "sensordata",
        )
        self._input_fields = (
            "qpos",
            "qvel",
            "qacc_warmstart",
            "ctrl",
            "act",
            "qfrc_applied",
            "xfrc_applied",
        )
        for field in self._output_fields:
            setattr(
                self.data, field, self.transfer.to_torch(getattr(self._state, field))
            )
        nefc = self._initial._impl.efc_J.shape[0]
        kinds = np.asarray(self._initial._impl.efc_type)
        ids = np.zeros(nefc, dtype=np.int32)
        ids[kinds == int(mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF)] = (
            self._friction_dofs
        )
        self.data.nefc = torch.full((num_envs,), nefc, device="mps", dtype=torch.int32)
        self.data.efc = SimpleNamespace(
            type=torch.as_tensor(kinds.copy(), device="mps")
            .expand(num_envs, -1)
            .clone(),
            id=torch.as_tensor(ids, device="mps").expand(num_envs, -1).clone(),
            force=torch.zeros((num_envs, nefc), device="mps"),
        )
        self._versions = {}
        self._device_fields = {}
        self._compiled = {}
        self.use_cuda_graph = False
        from mjlab.utils.nan_guard import NanGuard

        self.nan_guard = NanGuard(cfg.nan_guard, num_envs, model)

    def expand_model_fields(self, fields):
        for field in fields:
            if not hasattr(self._base, field):
                raise NotImplementedError(
                    f"MJX model randomization field {field} is unsupported"
                )
            getattr(self.model, field)
            self.expanded_fields.add(field)
        self._compiled.clear()

    def get_default_field(self, field):
        if field not in self.default_model_fields:
            self.default_model_fields[field] = torch.as_tensor(
                getattr(self.mj_model, field).copy(),
                device="mps",
                dtype=getattr(self.model, field).dtype,
            )
        return self.default_model_fields[field]

    def _fields(self):
        for name in sorted(self.expanded_fields):
            tensor = getattr(self.model, name)
            version = tensor._version
            if self._versions.get(name) != version:
                self._device_fields[name] = self.transfer.to_jax(tensor)
                self._versions[name] = version
        return dict(self._device_fields)

    def _model(self, fields):
        fields = dict(fields)
        mean = fields.pop("stat_meaninertia", self._base.stat.meaninertia)
        return self._base.replace(
            **fields, stat=self._base.stat.replace(meaninertia=mean)
        )

    def _sync_in(self):
        changes = {
            name: self.transfer.to_jax(getattr(self.data, name))
            for name in self._input_fields
        }
        self._state = self._state.replace(**changes)

    def _sync_out(self):
        for field in self._output_fields:
            getattr(self.data, field).copy_(
                self.transfer.to_torch(getattr(self._state, field))
            )
        self.data.efc.force.copy_(self.transfer.to_torch(self._state._impl.efc_force))

    def _run(self, mode):
        self._sync_in()
        fields = self._fields()
        if mode not in self._compiled:
            function = self._contacts.step if mode == "step" else self._contacts.forward
            self._compiled[mode] = jax.jit(
                jax.vmap(lambda d, p: function(self._model(p), d))
            )
        self._state = self._compiled[mode](self._state, fields)
        self._sync_out()

    def step(self):
        with self.nan_guard.watch(self.data):
            self._run("step")

    def forward(self):
        self._run("forward")

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device="mps")
        ids = self.transfer.to_jax(env_ids)
        self._state = jax.tree.map(
            lambda a, b: a.at[ids].set(b), self._state, self._initial
        )
        self._sync_out()

    def sense(self):
        for sensor in self._sensors:
            sensor.sense()

    def recompute_constants(self, level):
        # Preserve non-accumulating mass/inertia DR in the canonical task.
        # Unsupported mechanisms are rejected rather than retaining stale constants.
        if self.mj_model.ntendon or self.mj_model.neq:
            raise NotImplementedError(
                "Native randomized constants currently require no tendons/equalities"
            )
        for name in ("body_subtreemass", "dof_invweight0", "body_invweight0"):
            getattr(self.model, name)
            self.expanded_fields.add(name)
        fields = self._fields()
        if "constants" not in self._compiled:

            def constants(p):
                m = self._model(p)
                mass = m.body_mass
                for body in range(m.nbody - 1, 0, -1):
                    mass = mass.at[m.body_parentid[body]].add(mass[body])
                m = m.replace(body_subtreemass=mass)
                d = self._initial.replace(qpos=m.qpos0)
                for fn in (
                    smooth.kinematics,
                    smooth.com_pos,
                    smooth.crb,
                    smooth.factor_m,
                ):
                    d = fn(m, d)
                matrix = support.full_m(m, d)
                factor = jnp.linalg.cholesky(matrix)
                import jax.scipy.linalg as jl

                inverse = jl.cho_solve((factor, True), jnp.eye(m.nv))
                dof = jnp.diag(inverse)
                for joint, kind in enumerate(m.jnt_type):
                    adr = m.jnt_dofadr[joint]
                    if kind == mujoco.mjtJoint.mjJNT_FREE:
                        dof = dof.at[adr : adr + 3].set(dof[adr : adr + 3].mean())
                        dof = dof.at[adr + 3 : adr + 6].set(
                            dof[adr + 3 : adr + 6].mean()
                        )
                    elif kind == mujoco.mjtJoint.mjJNT_BALL:
                        dof = dof.at[adr : adr + 3].set(dof[adr : adr + 3].mean())
                weights = []
                for body in range(m.nbody):
                    jp, jr = support.jac(m, d, d.xipos[body], body)
                    trans = jnp.sum(jp * (inverse @ jp)) / 3
                    rot = jnp.sum(jr * (inverse @ jr)) / 3
                    trans, rot = (
                        jnp.where(trans < mujoco.mjMINVAL, rot, trans),
                        jnp.where(rot < mujoco.mjMINVAL, trans, rot),
                    )
                    weight = jnp.array([trans, rot])
                    weights.append(jnp.where(m.body_weldid[body] == 0, 0.0, weight))
                return {
                    "body_subtreemass": mass,
                    "dof_invweight0": dof,
                    "body_invweight0": jnp.stack(weights),
                    "stat_meaninertia": jnp.diag(matrix).mean(),
                }

            self._compiled["constants"] = jax.jit(jax.vmap(constants))
        computed = self._compiled["constants"](fields)
        for name, value in computed.items():
            if name == "stat_meaninertia":
                self._device_fields[name] = value
            else:
                getattr(self.model, name).copy_(self.transfer.to_torch(value))
        self._compiled.pop("step", None)
        self._compiled.pop("forward", None)
