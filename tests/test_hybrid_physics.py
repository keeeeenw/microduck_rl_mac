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
