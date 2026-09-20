"""Unit tests for Deliverable 2: Two-Way Shared-Buffer Metal Probe."""

import pytest
import sys
from pathlib import Path

SRC_DIR = Path(__file__).parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from metal_probe import MetalSharedBufferProbe


@pytest.fixture(scope="module")
def probe():
    return MetalSharedBufferProbe()


def test_two_way_ordering_cycle(probe):
    """Verify bidirectional execution: Torch -> Metal -> Torch -> Metal -> Torch."""
    res = probe.run_two_way_ordering_cycle(num_envs=64, n_dofs=20, iterations=10)
    assert res["status"] == "passed"
    assert res["final_qpos_mean"] > 0.0
    assert res["final_qvel_mean"] > 0.0


def test_contiguous_slice_offset(probe):
    """Verify contiguous offset slices (e.g. x[10:20]) mutate only the target range."""
    res = probe.test_slicing_and_offsets()
    assert res["contiguous_offset_verified"] is True


def test_non_contiguous_rejection(probe):
    """Verify that non-contiguous slices (e.g. x[:, :14]) and transposed views are rejected."""
    res = probe.test_slicing_and_offsets()
    assert res["non_contiguous_slice_rejected"] is True
    assert res["transposed_tensor_rejected"] is True


def test_memory_stability_1000_steps(probe):
    """Verify bounded memory plateau over repeated batches."""
    res = probe.run_memory_stability_test(num_steps=1000, num_envs=64, n_dofs=20)
    assert res["status"] == "passed"
    assert res["alloc_delta_bytes"] <= 1024
    assert res["driver_delta_bytes"] <= 1024 * 1024  # within 1 MB plateau
