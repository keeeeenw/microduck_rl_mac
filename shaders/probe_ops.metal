#include <metal_stdlib>
using namespace metal;

// Kernel 1: In-place scalar scaling on contiguous device buffer
kernel void probe_inplace_scale(
    device float* data [[buffer(0)]],
    constant float& scale [[buffer(1)]],
    uint idx [[thread_position_in_grid]]
) {
    data[idx] *= scale;
}

// Kernel 2: Fused Multiply-Add between device buffers
kernel void probe_fma(
    device const float* in_a [[buffer(0)]],
    device const float* in_b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant float& factor [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    out[idx] = in_a[idx] * factor + in_b[idx];
}

// Kernel 3: Substep state update (simulating physical integration)
kernel void probe_substep_integrate(
    device float* qpos [[buffer(0)]],
    device float* qvel [[buffer(1)]],
    device const float* ctrl [[buffer(2)]],
    constant float& dt [[buffer(3)]],
    constant float& damping [[buffer(4)]],
    uint idx [[thread_position_in_grid]]
) {
    float v = qvel[idx] + (ctrl[idx] - damping * qvel[idx]) * dt;
    float q = qpos[idx] + v * dt;
    qvel[idx] = v;
    qpos[idx] = q;
}

// Kernel 4: Indirect gather-scatter with integer indexing
kernel void probe_gather(
    device const float* src [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device float* dst [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    int src_idx = indices[idx];
    dst[idx] = src[src_idx];
}
