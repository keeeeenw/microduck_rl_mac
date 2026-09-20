import random
from types import SimpleNamespace
import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires native MPS")
def test_environment_snapshot_restores_tensors_counters_and_rng():
    from mjlab_microduck.native_gpu.checkpoint import capture, restore

    sim = SimpleNamespace(
        _state={"qpos": jax.numpy.ones((2, 7))},
        _versions={"x": 1},
        _device_fields={"stat_meaninertia": jax.numpy.ones(2)},
        _sync_out=lambda: None,
    )
    env = SimpleNamespace(
        sim=sim,
        episode_length_buf=torch.arange(2, device="mps"),
        common_step_counter=24,
        command={"value": torch.ones((2, 3), device="mps")},
        settings=(0.1, 0.2),
    )
    env.extras = {"log": {"reward": torch.ones(1, device="mps")}}
    state = capture(env)
    env.extras["log"]["reward"] = torch.tensor(0.0, device="mps")
    expected = torch.rand(8, device="mps")
    expected_py = random.random()
    expected_np = np.random.rand()
    env.episode_length_buf += 10
    env.command["value"].zero_()
    env.common_step_counter = 999
    sim._state = {"qpos": jax.numpy.zeros((2, 7))}
    restore(env, state)
    assert env.common_step_counter == 24
    torch.testing.assert_close(env.episode_length_buf, torch.arange(2, device="mps"))
    torch.testing.assert_close(env.command["value"], torch.ones((2, 3), device="mps"))
    torch.testing.assert_close(torch.rand(8, device="mps"), expected, rtol=0, atol=0)
    assert random.random() == expected_py
    assert np.random.rand() == expected_np
    np.testing.assert_array_equal(np.asarray(sim._state["qpos"]), 1.0)
