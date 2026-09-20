"""Unit tests for Deliverable 2: Two-Way Shared-Buffer Metal Probe."""

import pytest
import sys
from pathlib import Path
import torch

SRC_DIR = Path(__file__).parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from metal_probe import MetalSharedBufferProbe


@pytest.fixture(scope="module")
def probe():
    return MetalSharedBufferProbe()


def test_two_way_ordering_cycle_parity(probe):
    """Verify bidirectional execution against exact deterministic CPU mathematical reference."""
    res = probe.run_two_way_ordering_cycle(num_envs=64, n_dofs=20, iterations=5)
    assert res["status"] == "passed"
    assert res["max_qpos_error"] < 1e-5
    assert res["max_qvel_error"] < 1e-5
    assert res["max_obs_error"] < 1e-5
    assert res["max_actions_error"] < 1e-5
    assert res["max_grad_error"] < 1e-4


def test_two_way_ordering_omission_fails(probe):
    """Verify that omitting an intermediate Metal mutation is caught as an error."""
    res = probe.run_two_way_ordering_cycle(num_envs=64, n_dofs=20, iterations=1, test_mutation_omission=True)
    assert res["status"] == "omission_detected"
    assert res["diff_obs"] > 0.1
    assert res["diff_actions"] > 0.1
    assert res["diff_grad"] > 0.1


def test_pointer_identity_preserved(probe):
    """Verify that tensor memory address (data_ptr) is preserved across Metal and Torch ops."""
    qpos = torch.randn((64, 20), device="mps", dtype=torch.float32)
    ptr_initial = qpos.data_ptr()
    
    probe.mgr.launch("probe_inplace_scale", qpos, 2.0, threads=64*20)
    assert qpos.data_ptr() == ptr_initial, "In-place Metal kernel must not change data_ptr"
    
    # Slice offset pointer verification
    slice_10_20 = qpos[:, 5:15]
    assert slice_10_20.data_ptr() == ptr_initial + 5 * 4, "Slice data_ptr must equal base + offset * sizeof(float)"


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
