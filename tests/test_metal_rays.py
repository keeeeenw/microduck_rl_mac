"""GPU flat terrain rays compared with native MuJoCo reference intersections."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("jax")
mujoco = pytest.importorskip("mujoco")
from mjlab.sensor.builtin_sensor import ObjRef
from mjlab.sensor.raycast_sensor import RingPatternCfg
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensorCfg
from mjlab_microduck.native_gpu.environment import MetalTerrainHeight
from mjlab_microduck.native_gpu.simulation import ModelView


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires native MPS")
@pytest.mark.parametrize("x,z", [(0.0, 0.2), (0.0, 1.2), (0.0, -0.1), (0.6, 0.2)])
def test_height_rays_match_native_plane(x, z):
    model = mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
      <geom type="plane" size=".5 .5 .1" group="0"/>
      <site name="foot" pos="{x} 0 {z}" size=".01"/>
      </worldbody></mujoco>''')
    reference = mujoco.MjData(model)
    mujoco.mj_forward(model, reference)
    data = SimpleNamespace(
        **{
            name: torch.tensor(
                getattr(reference, name), dtype=torch.float32, device="mps"
            ).unsqueeze(0)
            for name in ("geom_xpos", "geom_xmat", "site_xpos", "site_xmat")
        }
    )
    cfg = TerrainHeightSensorCfg(
        name="height",
        frame=ObjRef(type="site", name="foot"),
        pattern=RingPatternCfg.single_ring(radius=0.04, num_samples=2),
        ray_alignment="yaw",
        max_distance=1.0,
        include_geom_groups=(0,),
        exclude_parent_body=False,
    )
    sensor = MetalTerrainHeight(cfg)
    sensor.initialize(model, ModelView(model, 1), data, "mps")
    expected = []
    offsets = sensor._local_offsets.cpu().numpy()
    for offset in offsets:
        geom = np.zeros(1, dtype=np.int32)
        distance = mujoco.mj_ray(
            model,
            reference,
            np.array([x, 0.0, z]) + offset,
            np.array([0.0, 0.0, -1.0]),
            None,
            1,
            -1,
            geom,
        )
        expected.append(distance if distance <= 1.0 else -1.0)
    np.testing.assert_allclose(
        sensor.data.distances.cpu().numpy()[0], expected, atol=1e-6
    )
