"""Native CPU MuJoCo physics with the original mjlab managers and BAM on MPS.

One working model is reused sequentially, applying each world's randomized fields
before stepping its independent MjData. This avoids duplicating large CAD meshes.
"""

import copy
from types import SimpleNamespace
import time

import mujoco
import numpy as np
import torch

from .simulation import ModelView, Transfer, MetalSimulation


class CpuSimulation:
    physics_backend = "cpu"
    _output_fields = (
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
    _input_fields = (
        "qpos",
        "qvel",
        "qacc_warmstart",
        "ctrl",
        "act",
        "qfrc_applied",
        "xfrc_applied",
    )
    get_default_field = MetalSimulation.get_default_field
    sense = MetalSimulation.sense

    def __init__(self, num_envs, cfg, model, device):
        if str(device) != "mps" or not torch.backends.mps.is_available():
            raise RuntimeError("Hybrid training requires Torch MPS")
        self.cfg, self.device, self.num_envs = cfg, "mps", num_envs
        cfg.mujoco.apply(model)
        self.mj_model = model
        self.mj_data = mujoco.MjData(model)
        self.model = ModelView(model, num_envs)
        self._cpu_model = copy.copy(model)
        # COM/pose randomization invalidates these compiled shortcuts.
        self._cpu_model.body_sameframe[:] = 0
        self._cpu_model.body_simple[:] = 0
        self._cpu_model.dof_simplenum[:] = 0
        self._worlds = [mujoco.MjData(self._cpu_model) for _ in range(num_envs)]
        self.transfer = Transfer()
        self.expanded_fields = set()
        self.default_model_fields = {}
        self._sensors = []
        self._versions = {}
        self._host_fields = {}
        self._meaninertia = np.full(num_envs, model.stat.meaninertia)
        self.use_cuda_graph = False
        self.data = SimpleNamespace(nworld=num_envs)
        for field in self._output_fields:
            array = np.stack([getattr(d, field) for d in self._worlds]).astype(
                np.float32
            )
            setattr(self.data, field, self.transfer.to_torch(array))
        self._constraint_capacity = 0
        self.data.nefc = torch.zeros(num_envs, device="mps", dtype=torch.int32)
        self.data.efc = SimpleNamespace()
        self.ensure_constraint_capacity(128)
        from mjlab.utils.nan_guard import NanGuard

        self.nan_guard = NanGuard(cfg.nan_guard, num_envs, model)

    def expand_model_fields(self, fields):
        for name in fields:
            if not isinstance(getattr(self.mj_model, name, None), np.ndarray):
                raise NotImplementedError(f"Unsupported randomized field: {name}")
            getattr(self.model, name)
            self.expanded_fields.add(name)

    def _numpy(self, tensor):
        start = time.perf_counter()
        value = tensor.detach().cpu().numpy().copy()
        self.transfer.bytes += value.nbytes
        self.transfer.seconds += time.perf_counter() - start
        return value

    def _fields(self):
        for name in sorted(self.expanded_fields):
            tensor = getattr(self.model, name)
            if self._versions.get(name) != tensor._version:
                self._host_fields[name] = self._numpy(tensor)
                self._versions[name] = tensor._version
        return self._host_fields

    def _apply_model(self, index, fields):
        for name, value in fields.items():
            getattr(self._cpu_model, name)[:] = value[index]
        self._cpu_model.stat.meaninertia = self._meaninertia[index]

    def ensure_constraint_capacity(self, required, *, exact=False):
        """Grow the padded view without dropping any native constraint rows."""
        if required <= self._constraint_capacity and not exact:
            return
        capacity = max(128, 1 << (int(required) - 1).bit_length())
        if capacity == self._constraint_capacity:
            return
        for name, dtype in (
            ("type", torch.int32),
            ("id", torch.int32),
            ("force", torch.float32),
        ):
            value = torch.zeros((self.num_envs, capacity), device="mps", dtype=dtype)
            previous = getattr(self.data.efc, name, None)
            if previous is not None:
                retained = min(capacity, self._constraint_capacity)
                value[:, :retained].copy_(previous[:, :retained])
            setattr(self.data.efc, name, value)
        self._constraint_capacity = capacity

    def _copy_to_mps(self, target, value):
        # Copy into the persistent destination directly. to_torch(...).copy_()
        # allocated an intermediate MPS tensor and copied the data twice.
        start = time.perf_counter()
        target.copy_(torch.from_numpy(value))
        self.transfer.bytes += value.nbytes
        self.transfer.seconds += time.perf_counter() - start

    def _sync_out(self):
        for field in self._output_fields:
            value = np.stack([getattr(d, field) for d in self._worlds]).astype(
                np.float32
            )
            self._copy_to_mps(getattr(self.data, field), value)
        counts = np.array([d.nefc for d in self._worlds], dtype=np.int32)
        self.ensure_constraint_capacity(counts.max(initial=0))
        self._copy_to_mps(self.data.nefc, counts)
        for name, dtype in (
            ("type", np.int32),
            ("id", np.int32),
            ("force", np.float32),
        ):
            value = np.zeros((self.num_envs, self._constraint_capacity), dtype=dtype)
            for i, d in enumerate(self._worlds):
                value[i, : d.nefc] = getattr(d, "efc_" + name)
            self._copy_to_mps(getattr(self.data.efc, name), value)

    def _run(self, mode):
        fields = self._fields()
        inputs = {
            name: self._numpy(getattr(self.data, name)) for name in self._input_fields
        }
        fn = mujoco.mj_step if mode == "step" else mujoco.mj_forward
        for i, d in enumerate(self._worlds):
            self._apply_model(i, fields)
            for name, value in inputs.items():
                getattr(d, name)[:] = value[i]
            fn(self._cpu_model, d)
        self._sync_out()

    def step(self):
        with self.nan_guard.watch(self.data):
            self._run("step")

    def forward(self):
        self._run("forward")

    def reset(self, env_ids=None):
        ids = range(self.num_envs) if env_ids is None else self._numpy(env_ids)
        for i in ids:
            mujoco.mj_resetData(self._cpu_model, self._worlds[int(i)])
        self._sync_out()

    def recompute_constants(self, level):
        if self.mj_model.ntendon or self.mj_model.neq:
            raise NotImplementedError(
                "Hybrid constants currently require no tendons/equalities"
            )
        fields = self._fields()
        names = ("body_subtreemass", "dof_invweight0", "body_invweight0")
        values = {name: [] for name in names}
        scratch = mujoco.MjData(self._cpu_model)
        for i in range(self.num_envs):
            self._apply_model(i, fields)
            mujoco.mj_setConst(self._cpu_model, scratch)
            self._meaninertia[i] = self._cpu_model.stat.meaninertia
            for name in names:
                values[name].append(getattr(self._cpu_model, name).copy())
        for name in names:
            getattr(self.model, name).copy_(
                self.transfer.to_torch(np.stack(values[name]).astype(np.float32))
            )
            self.expanded_fields.add(name)

    def capture_physics(self):
        kind = mujoco.mjtState.mjSTATE_INTEGRATION
        arrays = []
        for d in self._worlds:
            value = np.empty(mujoco.mj_stateSize(self._cpu_model, kind))
            mujoco.mj_getState(self._cpu_model, d, value, kind)
            arrays.append(torch.from_numpy(value))
        return {
            "backend": "cpu",
            "states": arrays,
            "meaninertia": torch.from_numpy(self._meaninertia.copy()),
        }

    def restore_physics(self, saved):
        if saved["backend"] != "cpu" or len(saved["states"]) != self.num_envs:
            raise ValueError("Incompatible hybrid physics checkpoint")
        self._meaninertia = saved["meaninertia"].numpy().copy()
        self._versions.clear()
        fields = self._fields()
        for i, (d, value) in enumerate(zip(self._worlds, saved["states"])):
            self._apply_model(i, fields)
            mujoco.mj_setState(
                self._cpu_model, d, value.numpy(), mujoco.mjtState.mjSTATE_INTEGRATION
            )
            mujoco.mj_forward(self._cpu_model, d)
        self._sync_out()
