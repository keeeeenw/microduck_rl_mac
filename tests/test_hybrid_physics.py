"""Shared-model CPU worlds must preserve independent state and randomization."""

import copy
import numpy as np
import pytest
import torch
import mujoco

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires native MPS"
)


def test_world_randomization_and_reset_match_independent_mujoco():
    from mjlab.sim import SimulationCfg
    from mjlab_microduck.native_gpu.cpu_simulation import CpuSimulation

    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <geom type="plane" size="0 0 .1"/>
      <body pos="0 0 .12"><freejoint/><geom type="box" size=".1 .1 .1" mass="1"/>
      <body pos="0 0 .2"><joint name="hinge"/><geom type="sphere" size=".05" mass=".2"/></body>
      </body></worldbody><actuator><motor joint="hinge"/></actuator></mujoco>""")
    sim = CpuSimulation(2, SimulationCfg(), model, "mps")
    fields = (
        "body_mass",
        "body_inertia",
        "body_ipos",
        "dof_armature",
        "dof_frictionloss",
    )
    sim.expand_model_fields(fields)
    sim.model.body_mass[1] *= 1.3
    sim.model.body_inertia[1] *= 1.3
    sim.model.body_ipos[1, 1, 0] += 0.02
    sim.model.dof_frictionloss[:, 6] = torch.tensor([0.01, 0.03], device="mps")
    sim.recompute_constants(None)
    refs = []
    for i in range(2):
        m = copy.copy(model)
        m.body_sameframe[:] = 0
        m.body_simple[:] = 0
        m.dof_simplenum[:] = 0
        for name in fields:
            getattr(m, name)[:] = getattr(sim.model, name)[i].cpu().numpy()
        d = mujoco.MjData(m)
        mujoco.mj_setConst(m, d)
        mujoco.mj_resetData(m, d)
        refs.append((m, d))
    sim.data.ctrl[:, 0] = torch.tensor([0.1, -0.2], device="mps")
    for _ in range(8):
        for i, (m, d) in enumerate(refs):
            for name in sim._input_fields:
                getattr(d, name)[:] = getattr(sim.data, name)[i].cpu().numpy()
            mujoco.mj_step(m, d)
        sim.step()
        for i, (m, d) in enumerate(refs):
            np.testing.assert_allclose(sim.data.qpos[i].cpu(), d.qpos, atol=1e-6)
            np.testing.assert_allclose(sim.data.qvel[i].cpu(), d.qvel, atol=1e-6)
    untouched = sim.data.qpos[1].clone()
    sim.reset(torch.tensor([0], device="mps"))
    torch.testing.assert_close(sim.data.qpos[1], untouched, atol=0, rtol=0)
    np.testing.assert_allclose(sim.data.qpos[0].cpu(), model.qpos0, atol=1e-6)
    snapshot = sim.capture_physics()
    sim.step()
    sim.restore_physics(snapshot)
    torch.testing.assert_close(sim.data.qpos[1], untouched, atol=0, rtol=0)


def test_constraint_view_grows_without_truncating_or_retaining_stale_rows():
    from types import SimpleNamespace
    from mjlab.sim import SimulationCfg
    from mjlab_microduck.native_gpu.cpu_simulation import CpuSimulation

    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><freejoint/><geom type="sphere" size=".1"/></body></worldbody></mujoco>'
    )
    sim = CpuSimulation(1, SimulationCfg(), model, "mps")
    d = sim._worlds[0]
    rows = 257
    world = SimpleNamespace(
        **{name: getattr(d, name) for name in sim._output_fields},
        nefc=rows,
        efc_type=np.arange(rows, dtype=np.int32),
        efc_id=np.arange(rows, dtype=np.int32),
        efc_force=np.arange(rows, dtype=np.float64) / 2,
    )
    sim._worlds = [world]
    sim._sync_out()
    assert sim._constraint_capacity == 512
    for name in ("type", "id", "force"):
        actual = getattr(sim.data.efc, name)[0].cpu().numpy()
        np.testing.assert_array_equal(actual[:rows], getattr(world, "efc_" + name))
        np.testing.assert_array_equal(actual[rows:], 0)
    world.nefc = 2
    for name in ("type", "id", "force"):
        setattr(world, "efc_" + name, getattr(world, "efc_" + name)[:2])
    sim._sync_out()
    for name in ("type", "id", "force"):
        np.testing.assert_array_equal(getattr(sim.data.efc, name)[0, 2:].cpu(), 0)


@pytest.mark.parametrize("initial_capacity", [128, 8192])
def test_native_checkpoint_restores_constraint_capacity(initial_capacity):
    from types import SimpleNamespace
    from mjlab.sim import SimulationCfg
    from mjlab_microduck.native_gpu.cpu_simulation import CpuSimulation
    from mjlab_microduck.native_gpu.checkpoint import capture, restore

    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body pos="0 0 1"><freejoint/><geom type="sphere" size=".1"/></body></worldbody></mujoco>'
    )
    sim = CpuSimulation(1, SimulationCfg(), model, "mps")
    sim.ensure_constraint_capacity(4096)  # legacy checkpoint layout
    sim.step()
    state = capture(SimpleNamespace(sim=sim))
    fresh = CpuSimulation(1, SimulationCfg(), copy.copy(model), "mps")
    fresh.ensure_constraint_capacity(initial_capacity)
    restore(SimpleNamespace(sim=fresh), state)
    assert fresh._constraint_capacity == 4096
    assert fresh.data.efc.force.shape == (1, 4096)
    torch.testing.assert_close(fresh.data.qpos, sim.data.qpos)


def test_native_scene_omits_only_visual_origin_sites():
    import dataclasses
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg
    from mjlab.scene import Scene
    from mjlab_microduck.native_gpu.environment import MetalScene

    cfg = load_env_cfg("Mjlab-Velocity-Flat-MicroDuck").scene
    cfg.num_envs = 16
    original = Scene(dataclasses.replace(cfg), device="cpu")
    native = MetalScene(dataclasses.replace(cfg), device="cpu")
    reference = original.compile()
    compact = native.compile()
    assert reference.nsite - compact.nsite == 16
    assert compact.nsite == 7
    torch.testing.assert_close(native.env_origins, original.env_origins)
    for field in ("nq", "nv", "nu", "nbody", "ngeom"):
        assert getattr(reference, field) == getattr(compact, field)
    for field in (
        "qpos0",
        "body_mass",
        "body_inertia",
        "geom_friction",
        "actuator_gear",
    ):
        np.testing.assert_array_equal(
            getattr(reference, field), getattr(compact, field)
        )
    a, b = mujoco.MjData(reference), mujoco.MjData(compact)
    for _ in range(20):
        mujoco.mj_step(reference, a)
        mujoco.mj_step(compact, b)
    np.testing.assert_array_equal(a.qpos, b.qpos)
    np.testing.assert_array_equal(a.qvel, b.qvel)
    np.testing.assert_array_equal(a.sensordata, b.sensordata)
    for i in range(compact.nsite):
        name = mujoco.mj_id2name(compact, mujoco.mjtObj.mjOBJ_SITE, i)
        j = mujoco.mj_name2id(reference, mujoco.mjtObj.mjOBJ_SITE, name)
        np.testing.assert_array_equal(b.site_xpos[i], a.site_xpos[j])
        np.testing.assert_array_equal(b.site_xmat[i], a.site_xmat[j])
