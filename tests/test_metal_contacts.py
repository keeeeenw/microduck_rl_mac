"""Body/subtree matching and netforce must agree with native MuJoCo sensors."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
mujoco = pytest.importorskip("mujoco")
mjx = pytest.importorskip("mujoco.mjx")

from mjlab_microduck.native_gpu.contacts import ContactSensors


@pytest.fixture(scope="module", autouse=True)
def native_solver():
    if jax.default_backend() == "mps":
        from mjlab_microduck.native_gpu.linalg import register_mps_solver_lowerings

        register_mps_solver_lowerings()


@pytest.mark.parametrize(
    "primary,secondary",
    [("body", "body"), ("xbody", "body"), ("geom", "body"), ("body", "geom")],
)
def test_contact_sensors_match_mujoco(primary, secondary):
    spec = mujoco.MjSpec.from_string("""<mujoco>
      <option timestep="0.005" integrator="implicitfast"/>
      <worldbody>
        <body name="ground"><geom name="floor" type="plane" size="1 1 .1"/></body>
        <body name="root" pos="0 0 .09"><freejoint/>
          <body name="box"><geom name="cube" type="box" size=".1 .1 .1" mass="1"/></body>
        </body>
      </worldbody>
    </mujoco>""")
    kinds = {
        "body": mujoco.mjtObj.mjOBJ_BODY,
        "xbody": mujoco.mjtObj.mjOBJ_XBODY,
        "geom": mujoco.mjtObj.mjOBJ_GEOM,
    }
    for name, data in (("found", 1), ("force", 2), ("found_without_reduction", 1)):
        spec.add_sensor(
            name=name,
            type=mujoco.mjtSensor.mjSENS_CONTACT,
            objtype=kinds[primary],
            objname={"geom": "cube", "body": "box", "xbody": "root"}[primary],
            reftype=kinds[secondary],
            refname="floor" if secondary == "geom" else "ground",
            intprm=[data, 0 if name == "found_without_reduction" else 3, 1],
        )
    model = spec.compile()
    original_types = model.sensor_objtype.copy()
    adapter = ContactSensors(model)
    mx = adapter.put_model(model)
    dx = mjx.make_data(mx, impl="jax")
    result = jax.jit(adapter.forward)(mx, dx)
    ref = mujoco.MjData(model)
    mujoco.mj_forward(model, ref)
    np.testing.assert_array_equal(model.sensor_objtype, original_types)
    np.testing.assert_allclose(
        np.asarray(result.sensordata), ref.sensordata, atol=1e-3, rtol=1e-4
    )
    airborne = dx.replace(qpos=dx.qpos.at[2].set(1.0))
    result = jax.jit(adapter.forward)(mx, airborne)
    np.testing.assert_allclose(np.asarray(result.sensordata), 0.0, atol=1e-6)
