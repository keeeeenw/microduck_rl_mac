"""Randomized inertia constants must match MuJoCo's own reference recomputation."""

import copy
import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
mujoco = pytest.importorskip("mujoco")
pytest.importorskip("mujoco.mjx")


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires native MPS")
def test_randomized_constants_match_reference():
    from mjlab.sim import SimulationCfg
    from mjlab_microduck.native_gpu.simulation import MetalSimulation

    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <body pos="0 0 1"><freejoint/><geom type="box" size=".1 .2 .3" mass="1"/>
        <body pos="0 0 .4"><joint name="hinge"/><geom type="capsule" size=".05 .1" mass=".2"/></body>
      </body></worldbody><actuator><motor joint="hinge"/></actuator></mujoco>""")
    sim = MetalSimulation(2, SimulationCfg(), model, "mps")
    fields = ("body_mass", "body_inertia", "body_ipos", "dof_armature")
    sim.expand_model_fields(fields)
    scale = torch.tensor([0.7, 1.3], device="mps")[:, None]
    sim.model.body_mass[:] *= scale
    sim.model.body_inertia[:] *= scale[:, :, None]
    sim.model.body_ipos[:, 1:, 0] += 0.02
    sim.model.dof_armature[:, 6:] += 0.002
    sim.recompute_constants(None)
    for i in range(2):
        ref = copy.copy(model)
        for field in fields:
            getattr(ref, field)[:] = getattr(sim.model, field)[i].cpu().numpy()
        # These compile-time fast paths assume the original zero COM. Native
        # mj_setConst does not invalidate them after programmatic COM changes.
        ref.body_sameframe[:] = 0
        ref.body_simple[:] = 0
        ref.dof_simplenum[:] = 0
        mujoco.mj_setConst(ref, mujoco.MjData(ref))
        for field in ("body_subtreemass", "body_invweight0", "dof_invweight0"):
            np.testing.assert_allclose(
                getattr(sim.model, field)[i].cpu(),
                getattr(ref, field),
                rtol=3e-4,
                atol=1e-4,
            )
        np.testing.assert_allclose(
            np.asarray(sim._device_fields["stat_meaninertia"][i]),
            ref.stat.meaninertia,
            rtol=3e-4,
        )
