"""Native environment continuation state, separate from portable actor export."""

import random
from types import SimpleNamespace

import jax
import numpy as np
import torch


def _children(value):
    if isinstance(value, dict):
        return [(("key", key), child) for key, child in value.items()]
    if isinstance(value, (list, tuple)):
        return [(("index", i), child) for i, child in enumerate(value)]
    module = type(value).__module__
    if module.startswith(("mjlab", "bam")) or isinstance(value, SimpleNamespace):
        return [
            (("attr", name), child)
            for name, child in vars(value).items()
            if name
            not in (
                "_cached_data",
                "_cache_valid",
                "nan_guard",
                "_compiled",
                "_versions",
            )
        ]
    return []


def _simple(value):
    return isinstance(value, (str, int, float, bool, type(None))) or (
        isinstance(value, tuple) and all(_simple(x) for x in value)
    )


def capture(env):
    leaves = []
    seen = set()

    def visit(value, path):
        if isinstance(value, torch.Tensor):
            leaves.append((path, "tensor", value.detach().cpu().clone()))
            return
        if _simple(value):
            leaves.append((path, "value", value))
            return
        if id(value) in seen:
            return
        seen.add(id(value))
        for part, child in _children(value):
            visit(child, path + (part,))

    visit(env, ())
    arrays = jax.tree.leaves(env.sim._state)
    return {
        "leaves": leaves,
        "physics": [torch.from_numpy(np.asarray(a).copy()) for a in arrays],
        "stat_meaninertia": torch.from_numpy(
            np.asarray(env.sim._device_fields["stat_meaninertia"]).copy()
        ),
        "random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_random": torch.get_rng_state(),
        "mps_random": torch.mps.get_rng_state(),
    }


def _get(value, part):
    kind, key = part
    return getattr(value, key) if kind == "attr" else value[key]


def _set(value, part, child):
    kind, key = part
    if kind == "attr":
        setattr(value, key, child)
    elif isinstance(value, tuple):
        if _get(value, part) != child:
            raise ValueError("Immutable checkpoint configuration differs")
    else:
        value[key] = child


def restore(env, state):
    for path, kind, saved in state["leaves"]:
        # Episode logging tensors change rank between reset and aggregation.
        # They are diagnostics, not continuation state.
        if path and path[0] == ("attr", "extras"):
            continue
        parent = env
        for part in path[:-1]:
            parent = _get(parent, part)
        if kind == "tensor":
            try:
                current = _get(parent, path[-1])
            except (KeyError, AttributeError):
                current = None
            if current is None:
                _set(parent, path[-1], saved.to("mps"))
            elif current.shape != saved.shape:
                raise ValueError(f"Checkpoint tensor shape mismatch at {path}")
            else:
                current.copy_(saved.to(current.device))
        else:
            try:
                unchanged = _get(parent, path[-1]) == saved
            except (KeyError, AttributeError):
                unchanged = False
            if not unchanged:
                _set(parent, path[-1], saved)
    current, tree = jax.tree.flatten(env.sim._state)
    if len(current) != len(state["physics"]):
        raise ValueError("Checkpoint physics layout differs")
    for a, b in zip(current, state["physics"]):
        if a.shape != tuple(b.shape):
            raise ValueError("Checkpoint physics shape differs")
    env.sim._state = jax.tree.unflatten(
        tree, [jax.numpy.asarray(a.numpy()) for a in state["physics"]]
    )
    env.sim._versions.clear()
    env.sim._device_fields = {
        "stat_meaninertia": jax.numpy.asarray(state["stat_meaninertia"].numpy())
    }
    env.sim._sync_out()
    if hasattr(env, "scene"):
        for sensor in env.scene._sensors.values():
            sensor._invalidate_cache()
    random.setstate(state["random"])
    np.random.set_state(state["numpy_random"])
    torch.set_rng_state(state["torch_random"])
    torch.mps.set_rng_state(state["mps_random"])
