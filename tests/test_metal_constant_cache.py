from types import SimpleNamespace
import copy

import mujoco
import numpy as np
import pytest
import torch

import pytest
pytest.importorskip("jax")
pytest.importorskip("mujoco.mjx")

from mjlab_microduck.native_gpu.metal.metal_simulation_adapter import UnifiedMetalSimulation


XML = '<mujoco><worldbody><body name="root"><freejoint/><geom type="sphere" size="0.1" mass="1"/></body></worldbody></mujoco>'
INPUTS = ('body_mass', 'body_ipos', 'body_inertia', 'body_iquat', 'dof_armature')
OUTPUTS = ('body_subtreemass', 'dof_invweight0', 'body_invweight0')


def make_sim(batch=3):
    model = mujoco.MjModel.from_xml_string(XML)
    fields = {name: torch.tensor(np.repeat(np.asarray(getattr(model, name))[None], batch, axis=0).copy(), dtype=torch.float32) for name in INPUTS + OUTPUTS}
    sim = object.__new__(UnifiedMetalSimulation)
    sim.mj_model = model
    sim.num_envs = batch
    sim.device = 'cpu'
    sim.model = SimpleNamespace(**fields)
    sim.expanded_fields = set()
    sim.reset_transfer = SimpleNamespace(bytes=0, seconds=0.0)
    sim._constant_inputs_cache = None
    sim._constant_outputs_cache = None
    sim._fk_cache_signature = None
    sim.constants_recomputed_envs = 0
    sim.constants_cache_skips = 0
    return sim


def reference(sim):
    expected = {name: [] for name in OUTPUTS}
    for i in range(sim.num_envs):
        m = copy.copy(sim.mj_model)
        d = mujoco.MjData(m)
        for name in INPUTS:
            getattr(m, name)[:] = getattr(sim.model, name)[i].numpy()
        mujoco.mj_setConst(m, d)
        for name in OUTPUTS:
            expected[name].append(np.asarray(getattr(m, name)).copy())
    return {name: np.stack(values) for name, values in expected.items()}


def assert_reference(sim):
    for name, expected in reference(sim).items():
        np.testing.assert_allclose(getattr(sim.model, name).numpy(), expected, rtol=2e-5, atol=2e-5)


def test_real_recompute_full_noop_changed_rows_and_explicit_ids():
    sim = make_sim()
    sim.recompute_constants(None)
    assert tuple(sim.model.body_invweight0.shape) == (3, sim.mj_model.nbody, 2)
    assert_reference(sim)
    before = {name: getattr(sim.model, name).clone() for name in OUTPUTS}
    sim.recompute_constants(None)
    assert sim.constants_cache_skips == 1
    for name in OUTPUTS:
        torch.testing.assert_close(getattr(sim.model, name), before[name])
    sim.model.body_mass[1, 1] *= 1.3
    sim.recompute_constants(None)
    assert_reference(sim)
    for name in OUTPUTS:
        torch.testing.assert_close(getattr(sim.model, name)[0], before[name][0])
        torch.testing.assert_close(getattr(sim.model, name)[2], before[name][2])
    sim.model.body_mass[[0, 2], 1] *= 1.2
    sim.recompute_constants(None, env_ids=[2, 0, 2])
    assert_reference(sim)
    sim.recompute_constants(None, env_ids=[])
    with pytest.raises(IndexError):
        sim.recompute_constants(None, env_ids=[3])


def test_recompute_failure_invalidates_cache_and_retry(monkeypatch):
    sim = make_sim()
    original = mujoco.mj_setConst
    calls = 0
    def fail_once(m, d):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError('injected failure')
        return original(m, d)
    monkeypatch.setattr(mujoco, 'mj_setConst', fail_once)
    with pytest.raises(RuntimeError):
        sim.recompute_constants(None)
    assert sim._constant_inputs_cache is None
    sim.recompute_constants(None)
    assert_reference(sim)


def test_real_restore_invalidates_constants_before_a_b_a_sequence():
    sim = make_sim()
    data_names = ('time', 'qpos', 'qvel', 'qacc_warmstart', 'ctrl', 'qfrc_applied', 'xfrc_applied', 'qfrc_bias')
    sim.data = SimpleNamespace(**{name: torch.zeros((3, 1)) for name in data_names})
    sim.forward = lambda: None
    sim.recompute_constants(None)
    saved_a = {'backend': 'metal', **{name: getattr(sim.data, name).clone() for name in data_names}, **{name: getattr(sim.model, name).clone() for name in INPUTS + OUTPUTS}}
    sim.model.body_mass[:, 1] *= 1.4
    sim.recompute_constants(None)
    saved_b = {'backend': 'metal', **{name: getattr(sim.data, name).clone() for name in data_names}, **{name: getattr(sim.model, name).clone() for name in INPUTS + OUTPUTS}}
    sim.restore_physics(saved_a)
    sim.recompute_constants(None)  # cache A
    assert sim._constant_inputs_cache is not None
    sim.restore_physics(saved_b)  # external checkpoint B while cache described A
    assert sim._constant_inputs_cache is None
    sim.model.body_mass.copy_(saved_a['body_mass'])
    sim.recompute_constants(None)
    assert_reference(sim)
    sim.restore_physics(saved_a)
    sim.recompute_constants(None)
    assert_reference(sim)


def test_partial_restore_failure_invalidates_preexisting_cache():
    sim = make_sim()
    names = ('time', 'qpos', 'qvel', 'qacc_warmstart', 'ctrl', 'qfrc_applied', 'xfrc_applied', 'qfrc_bias')
    sim.data = SimpleNamespace(**{name: torch.zeros((3, 1)) for name in names})
    sim.forward = lambda: None
    sim.recompute_constants(None)
    assert sim._constant_inputs_cache is not None
    sim._fk_cache_signature = ('previous', 1)
    saved = {'backend': 'metal', **{name: getattr(sim.data, name).clone() for name in names}}
    saved['body_mass'] = sim.model.body_mass.clone() * 1.2
    saved['body_ipos'] = torch.zeros((2, 2, 2))  # fails after mass copy
    with pytest.raises(RuntimeError):
        sim.restore_physics(saved)
    assert sim._constant_inputs_cache is None
    assert sim._constant_outputs_cache is None
    assert sim._fk_cache_signature is None
    sim.recompute_constants(None)
    assert_reference(sim)
