import pytest

from mjlab_microduck.native_gpu.probe import audit_hlo


def test_rng_gpu_custom_call_is_allowed():
    assert audit_hlo("stablehlo.custom_call @mps.threefry2x32()")["custom_calls"] == [
        "mps.threefry2x32"
    ]


@pytest.mark.parametrize(
    "operation",
    [
        "stablehlo.cholesky",
        "stablehlo.triangular_solve",
        "stablehlo.custom_call @mps.svd()",
        'stablehlo.custom_call @"unknown.operation"()',
    ],
)
def test_unqualified_backend_operation_is_rejected(operation):
    with pytest.raises(RuntimeError):
        audit_hlo(operation)
