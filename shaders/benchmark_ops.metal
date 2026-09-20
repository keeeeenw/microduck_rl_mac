#include <metal_stdlib>
using namespace metal;

// Representative Stage 1: Forward Kinematics Transform
// Evaluates link transforms from parent frames and joint angles
kernel void stage1_kinematics(
    device const float* qpos [[buffer(0)]],
    device float* body_xpos [[buffer(1)]],
    device float* body_xquat [[buffer(2)]],
    constant uint& num_envs [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint env_id = gid.y;
    uint body_id = gid.x;
    if (env_id >= num_envs || body_id >= 17) return;

    uint idx = env_id * 17 * 3 + body_id * 3;
    uint q_idx = env_id * 21 + min(body_id, 20u);
    
    // Simulate link translation and rotation update
    float q = qpos[q_idx];
    body_xpos[idx + 0] = cos(q) * 0.1f;
    body_xpos[idx + 1] = sin(q) * 0.1f;
    body_xpos[idx + 2] = 0.25f + body_id * 0.02f;

    uint quat_idx = env_id * 17 * 4 + body_id * 4;
    body_xquat[quat_idx + 0] = 1.0f;
    body_xquat[quat_idx + 1] = 0.0f;
    body_xquat[quat_idx + 2] = sin(q * 0.5f);
    body_xquat[quat_idx + 3] = cos(q * 0.5f);
}

// Representative Stage 2: Broadphase Bounding Sphere Rejection
// Tests candidate pairs; writes active contact count and flags
kernel void stage2_broadphase(
    device const float* body_xpos [[buffer(0)]],
    device int* active_contacts [[buffer(1)]],
    constant uint& num_envs [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint env_id = gid.y;
    uint pair_id = gid.x;
    if (env_id >= num_envs || pair_id >= 35) return;

    // Simulate bounding sphere distance test
    uint b1_idx = env_id * 17 * 3 + (pair_id % 17) * 3;
    uint b2_idx = env_id * 17 * 3 + ((pair_id + 3) % 17) * 3;
    
    float dx = body_xpos[b1_idx + 0] - body_xpos[b2_idx + 0];
    float dy = body_xpos[b1_idx + 1] - body_xpos[b2_idx + 1];
    float dz = body_xpos[b1_idx + 2] - body_xpos[b2_idx + 2];
    float dist2 = dx*dx + dy*dy + dz*dz;

    // Active if within radius threshold
    active_contacts[env_id * 35 + pair_id] = (dist2 < 0.04f) ? 1 : 0;
}

// Representative Stage 3: Actuator Dynamics
// Calculates joint torques with friction and voltage limits
kernel void stage3_actuation(
    device const float* qvel [[buffer(0)]],
    device const float* ctrl [[buffer(1)]],
    device float* qfrc [[buffer(2)]],
    constant uint& num_envs [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint env_id = gid.y;
    uint dof_id = gid.x;
    if (env_id >= num_envs || dof_id >= 14) return;

    uint idx = env_id * 14 + dof_id;
    float v = qvel[env_id * 20 + 6 + dof_id];
    float target = ctrl[idx];
    
    // PD + Coulomb friction simulation
    float tau = 200.0f * (target - 0.0f) - 5.0f * v;
    float friction = (v > 0.01f) ? 0.05f : ((v < -0.01f) ? -0.05f : 0.0f);
    qfrc[idx] = tau - friction;
}

// Representative Stage 4: Integrator Step
// Integrates velocities and positions
kernel void stage4_integrate(
    device float* qpos [[buffer(0)]],
    device float* qvel [[buffer(1)]],
    device const float* qfrc [[buffer(2)]],
    constant float& dt [[buffer(3)]],
    constant uint& num_envs [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint env_id = gid.y;
    uint dof_id = gid.x;
    if (env_id >= num_envs || dof_id >= 20) return;

    uint v_idx = env_id * 20 + dof_id;
    uint q_idx = env_id * 21 + (dof_id >= 6 ? dof_id + 1 : dof_id);
    
    float force = (dof_id >= 6) ? qfrc[env_id * 14 + (dof_id - 6)] : 0.0f;
    float v = qvel[v_idx] + (force - 0.1f * qvel[v_idx]) * dt;
    qvel[v_idx] = v;
    qpos[q_idx] += v * dt;
}

// Representative Stage 5: Sensor Projection
// Computes IMU projected gravity and sensor outputs
kernel void stage5_sensors(
    device const float* body_xquat [[buffer(0)]],
    device float* sensordata [[buffer(1)]],
    constant uint& num_envs [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= num_envs) return;

    // Projected gravity from base quaternion
    uint q_idx = idx * 17 * 4; // base body quaternion
    float w = body_xquat[q_idx + 0];
    float x = body_xquat[q_idx + 1];
    float y = body_xquat[q_idx + 2];
    float z = body_xquat[q_idx + 3];

    // R^T * [0, 0, -1]
    sensordata[idx * 11 + 0] = 2.0f * (x * z - w * y);
    sensordata[idx * 11 + 1] = 2.0f * (y * z + w * x);
    sensordata[idx * 11 + 2] = 1.0f - 2.0f * (x * x + y * y);
}
