"""Original flat-task mjlab managers with native MPS simulation and terrain rays."""

from types import FunctionType

import mujoco
import numpy as np
import torch
from mjlab.entity.entity import Entity
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.scene import Scene
from mjlab.sensor.raycast_sensor import RayCastSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor
from mjlab.terrains.terrain_entity import TerrainEntity
from mjlab.utils.lab_api.math import quat_from_matrix

from .simulation import MetalSimulation


def _with_globals(function, **changes):
    result = FunctionType(
        function.__code__,
        dict(function.__globals__, **changes),
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    result.__kwdefaults__ = function.__kwdefaults__
    return result


class _TorchFactory:
    """MuJoCo's host doubles become float32 when creating MPS state buffers."""

    def __getattr__(self, name):
        return getattr(torch, name)

    def tensor(self, data, *args, **kwargs):
        if str(kwargs.get("device")) == "mps" and "dtype" not in kwargs:
            if isinstance(data, np.ndarray) and data.dtype == np.float64:
                kwargs["dtype"] = torch.float32
        return torch.tensor(data, *args, **kwargs)


_entity_initialize = _with_globals(Entity.initialize, torch=_TorchFactory())


class MetalTerrainHeight(TerrainHeightSensor):
    def initialize(self, mj_model, model, data, device):
        self._data, self._model, self._mj_model, self._device = (
            data,
            model,
            mj_model,
            device,
        )
        included = np.isin(mj_model.geom_group, self.cfg.include_geom_groups)
        candidates = np.flatnonzero(included)
        if (
            len(candidates) != 1
            or mj_model.geom_type[candidates[0]] != mujoco.mjtGeom.mjGEOM_PLANE
        ):
            raise NotImplementedError(
                "Native first task ray adapter requires one included terrain plane"
            )
        self._plane = int(candidates[0])
        self._ctx = self  # This native context owns and computes the ray buffers.
        frames = (
            self.cfg.frame if isinstance(self.cfg.frame, tuple) else (self.cfg.frame,)
        )
        self._frame_infos = []
        for frame in frames:
            obj = getattr(mj_model, frame.type)(frame.prefixed_name())
            self._frame_infos.append((frame.type, obj.id))
        self._num_frames = len(frames)
        self._local_offsets, self._local_directions = self.cfg.pattern.generate_rays(
            mj_model, device
        )
        self._num_rays_per_frame = len(self._local_offsets)
        self._num_rays = self._num_frames * self._num_rays_per_frame
        self.sense()

    @property
    def requires_sensor_context(self):
        return False

    def sense(self):
        positions, matrices = [], []
        for kind, identity in self._frame_infos:
            prefix = "" if kind == "body" else kind + "_"
            positions.append(getattr(self._data, prefix + "xpos")[:, identity])
            matrices.append(
                getattr(self._data, prefix + "xmat")[:, identity].reshape(-1, 3, 3)
            )
        pos, mat = torch.stack(positions, 1), torch.stack(matrices, 1)
        n, f = pos.shape[:2]
        rotation = self._compute_alignment_rotation(mat.reshape(-1, 3, 3)).reshape(
            n, f, 3, 3
        )
        origins = pos[:, :, None, :] + torch.einsum(
            "bfij,nj->bfni", rotation, self._local_offsets
        )
        rays = torch.einsum("bfij,nj->bfni", rotation, self._local_directions)
        origins, rays = origins.reshape(n, -1, 3), rays.reshape(n, -1, 3)
        plane_pos = self._data.geom_xpos[:, self._plane]
        normal = self._data.geom_xmat[:, self._plane].reshape(n, 3, 3)[:, :, 2]
        denominator = (rays * normal[:, None]).sum(-1)
        distance = ((plane_pos[:, None] - origins) * normal[:, None]).sum(
            -1
        ) / denominator
        # Native MuJoCo planes are one-sided: rays pointing away from the front
        # surface miss. Keep the existing height sensor's miss/fallback reduction.
        intersection = origins + rays * distance[:, :, None]
        plane_matrix = self._data.geom_xmat[:, self._plane].reshape(n, 3, 3)
        local_hit = torch.einsum(
            "bji,bnj->bni", plane_matrix, intersection - plane_pos[:, None]
        )
        size = self._model.geom_size[:, self._plane]
        in_rectangle = (
            (size[:, None, :2] <= 0) | (local_hit[:, :, :2].abs() <= size[:, None, :2])
        ).all(-1)
        hit = (
            (denominator < -mujoco.mjMINVAL)
            & (distance >= 0)
            & (distance <= self.cfg.max_distance)
            & in_rectangle
        )
        self._distances = torch.where(hit, distance, -1.0)
        self._normals_w = torch.where(hit[:, :, None], normal[:, None, :], 0.0)
        self._hit_pos_w = torch.where(
            hit[:, :, None], origins + rays * distance[:, :, None], origins
        )
        self._frame_pos_w = pos
        self._frame_quat_w = quat_from_matrix(mat.reshape(-1, 3, 3)).reshape(n, f, 4)
        self._pos_w, self._quat_w = pos[:, 0], self._frame_quat_w[:, 0]
        self._invalidate_cache()


class NativeTerrainEntity(TerrainEntity):
    def _add_env_origin_sites(self):
        # Visualization only: adding N sites to each of N native worlds makes
        # site pose storage and transfers quadratic. Keep env_origins itself.
        pass


class MetalScene(Scene):
    _add_terrain = _with_globals(Scene._add_terrain, TerrainEntity=NativeTerrainEntity)

    def initialize(self, mj_model, model, data):
        self._default_env_origins = torch.zeros((self._cfg.num_envs, 3), device="mps")
        for entity in self._entities.values():
            _entity_initialize(entity, mj_model, model, data, "mps")
        for sensor in self._sensors.values():
            if isinstance(sensor, TerrainHeightSensor):
                sensor.__class__ = MetalTerrainHeight
            elif isinstance(sensor, RayCastSensor):
                raise NotImplementedError(
                    "Only terrain height rays are qualified for native flat training"
                )
            sensor.initialize(mj_model, model, data, "mps")
        self._sensor_context = None


_original_init = _with_globals(
    ManagerBasedRlEnv.__init__, Scene=MetalScene, Simulation=MetalSimulation
)


class MetalEnv(ManagerBasedRlEnv):
    def __init__(self, cfg, device="mps", physics="mps", **kwargs):
        initialize = _original_init
        if physics == "cpu":
            from .cpu_simulation import CpuSimulation

            initialize = _with_globals(
                ManagerBasedRlEnv.__init__, Scene=MetalScene, Simulation=CpuSimulation
            )
        elif physics in ("metal", "unified_metal"):
            import sys
            from pathlib import Path
            p = Path("/Users/zixiao/workspace/microduck/unified-metal")
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
            from src.metal_simulation_adapter import UnifiedMetalSimulation

            initialize = _with_globals(
                ManagerBasedRlEnv.__init__, Scene=MetalScene, Simulation=UnifiedMetalSimulation
            )
        elif physics != "mps":
            raise ValueError(f"Unknown physics backend: {physics}")
        initialize(self, cfg, device, **kwargs)
        self.sim._sensors = [
            s for s in self.scene._sensors.values() if isinstance(s, MetalTerrainHeight)
        ]
