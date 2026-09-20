"""MPS learner component qualification with synthetic data, NOT a walking smoke test."""

import copy
import dataclasses
import os

import pytest
import torch
from tensordict import TensorDict


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires native MPS")
def test_original_ppo_updates_and_restores_on_mps(tmp_path):
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"
    import mjlab_microduck.tasks  # noqa: F401 - populate task registry.
    from mjlab.tasks.registry import load_rl_cfg
    from rsl_rl.algorithms import PPO
    from rsl_rl.models import MLPModel
    from rsl_rl.storage import RolloutStorage

    cfg = dataclasses.asdict(load_rl_cfg("Mjlab-Velocity-Flat-MicroDuck"))
    torch.manual_seed(cfg["seed"])
    n = 64

    def observation():
        # Both inputs are synthetic fixtures. This does not reproduce critic
        # privileged observations or claim an integrated task rollout.
        return TensorDict(
            {name: torch.randn(n, 61, device="mps") for name in ("actor", "critic")},
            batch_size=[n],
            device="mps",
        )

    obs = observation()

    def learner():
        models = []
        for name, width in [("actor", 14), ("critic", 1)]:
            options = copy.deepcopy(cfg[name])
            assert options.pop("class_name") == "MLPModel"
            assert options.pop("cnn_cfg") is None
            assert options.pop("rnn_type") is None
            options.pop("rnn_hidden_dim")
            options.pop("rnn_num_layers")
            models.append(
                MLPModel(obs, cfg["obs_groups"], name, width, **options).to("mps")
            )
        storage = RolloutStorage(
            "rl", n, cfg["num_steps_per_env"], obs, (14,), device="mps"
        )
        options = copy.deepcopy(cfg["algorithm"])
        assert options.pop("class_name") == "PPO"
        assert not options.pop("share_cnn_encoders")
        return PPO(*models, storage, device="mps", **options)

    algo = learner()
    before = [p.detach().clone() for p in algo.actor.parameters()]
    for _ in range(5):
        with torch.inference_mode():
            for _ in range(cfg["num_steps_per_env"]):
                actions = algo.act(obs)
                assert actions.device.type == "mps"
                rewards = -actions.square().mean(-1)
                obs = observation()
                algo.process_env_step(
                    obs, rewards, torch.zeros(n, dtype=torch.bool, device="mps"), {}
                )
            algo.compute_returns(obs)
        losses = algo.update()
        assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
    assert any(
        not torch.equal(old, new) for old, new in zip(before, algo.actor.parameters())
    )
    assert all(p.device.type == "mps" for p in algo.actor.parameters())
    assert all(p.device.type == "mps" for p in algo.critic.parameters())
    # Moment tensors must stay on GPU; Adam's scalar step counter may be on host.
    for state in algo.optimizer.state.values():
        assert state["exp_avg"].device.type == "mps"
        assert state["exp_avg_sq"].device.type == "mps"
    path = tmp_path / "component-fixture.pt"
    torch.save(algo.save(), path)
    restored = learner()
    restored.load(
        torch.load(path, map_location="mps", weights_only=True), None, strict=True
    )
    with torch.inference_mode():
        torch.testing.assert_close(restored.actor(obs), algo.actor(obs), rtol=0, atol=0)
        torch.testing.assert_close(
            restored.critic(obs), algo.critic(obs), rtol=0, atol=0
        )
    assert (
        restored.optimizer.state_dict()["param_groups"]
        == algo.optimizer.state_dict()["param_groups"]
    )
    for a, b in zip(restored.optimizer.state.values(), algo.optimizer.state.values()):
        torch.testing.assert_close(a["exp_avg"], b["exp_avg"], rtol=0, atol=0)
        torch.testing.assert_close(a["exp_avg_sq"], b["exp_avg_sq"], rtol=0, atol=0)
