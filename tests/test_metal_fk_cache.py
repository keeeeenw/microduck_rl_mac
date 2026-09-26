from types import SimpleNamespace
import torch

import pytest
pytest.importorskip("jax")
pytest.importorskip("mujoco.mjx")

from mjlab_microduck.native_gpu.metal.metal_simulation_adapter import UnifiedMetalSimulation


class Slice:
    def __init__(self):
        self.calls = 0
    def compute_forward_kinematics(self, qpos):
        self.calls += 1


def test_real_torch_mutations_replacement_and_reset_invalidation():
    sim = object.__new__(UnifiedMetalSimulation)
    sim.data = SimpleNamespace(qpos=torch.zeros(2, 3))
    sim.slice = Slice()
    sim._fk_cache_signature = None
    sim._compute_fk_for_current_state()
    sim._compute_fk_for_current_state()
    assert sim.slice.calls == 1
    sim.data.qpos.copy_(torch.ones(2, 3))
    sim._compute_fk_for_current_state()
    assert sim.slice.calls == 2
    sim.data.qpos[0, 0] = 3
    sim._compute_fk_for_current_state()
    assert sim.slice.calls == 3
    sim.data.qpos = sim.data.qpos.clone()
    sim._compute_fk_for_current_state()
    assert sim.slice.calls == 4
    sim._invalidate_fk_cache()  # reset/restore and public operation boundary
    sim._compute_fk_for_current_state()
    assert sim.slice.calls == 5


def test_inference_tensor_without_version_never_reuses_fk():
    sim = object.__new__(UnifiedMetalSimulation)
    with torch.inference_mode():
        sim.data = SimpleNamespace(qpos=torch.zeros(2, 3))
    sim.slice = Slice()
    sim._fk_cache_signature = None
    sim._compute_fk_for_current_state()
    sim._compute_fk_for_current_state()
    assert sim.slice.calls == 2


def test_public_forward_invalidates_shared_slice_buffer_after_internal_writer():
    sim = object.__new__(UnifiedMetalSimulation)
    sim.data = SimpleNamespace(qpos=torch.zeros(2, 3), qvel=torch.zeros(2, 3), ctrl=torch.zeros(2, 1))
    sim.slice = Slice()
    sim._fk_cache_signature = None
    sim._compute_fk_for_current_state()
    assert sim.slice.calls == 1
    sim._get_model_inputs = lambda: {}
    sim._detect_self_contacts = lambda: None
    sim._sync_out = lambda out: None
    sim.slice.forward_autonomous = lambda *args, **kwargs: SimpleNamespace(contact_overflow=torch.zeros(2), solver_status=torch.zeros(2))
    sim.forward()
    assert sim._fk_cache_signature is None
    sim._compute_fk_for_current_state()
    assert sim.slice.calls == 2


def test_real_reset_and_restore_invalidate_fk_signature(monkeypatch):
    from mjlab_microduck.native_gpu.metal import metal_simulation_adapter as adapter
    original_as_tensor = torch.as_tensor
    def cpu_as_tensor(data, *args, **kwargs):
        if kwargs.get('device') == 'mps':
            kwargs['device'] = 'cpu'
        return original_as_tensor(data, *args, **kwargs)
    monkeypatch.setattr(adapter.torch, 'as_tensor', cpu_as_tensor)
    sim = object.__new__(UnifiedMetalSimulation)
    sim.mj_model = SimpleNamespace(qpos0=torch.zeros(3).numpy())
    names = ('time', 'qpos', 'qvel', 'qacc_warmstart', 'ctrl', 'qfrc_applied', 'xfrc_applied', 'qfrc_bias')
    sim.data = SimpleNamespace(**{name: torch.ones(2, 3) for name in names})
    sim.device = 'cpu'
    sim.model = SimpleNamespace()
    sim.forward = lambda: None
    sim._fk_cache_signature = ('old', 1)
    sim.reset(torch.tensor([0]))
    assert sim._fk_cache_signature is None
    sim._fk_cache_signature = ('old', 2)
    saved = {'backend': 'metal', **{name: getattr(sim.data, name).clone() for name in names}}
    sim.restore_physics(saved)
    assert sim._fk_cache_signature is None
