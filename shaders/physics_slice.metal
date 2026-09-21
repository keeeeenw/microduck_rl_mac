#include <metal_stdlib>
using namespace metal;

// -----------------------------------------------------------------------------
// Constant structures and parameters
// -----------------------------------------------------------------------------

struct BodyConstants {
    int parent_id;
    int joint_type; // 0: free, 3: hinge
    int qpos_adr;
    int dof_adr;
    packed_float3 body_pos;
    packed_float4 body_quat;
    packed_float3 body_ipos;
    packed_float4 body_iquat;
    packed_float3 jnt_axis;
    float mass;
    packed_float3 inertia;
};

struct DofConstants {
    int dof_parentid;
    int dof_bodyid;
    float dof_armature;
};

struct GeomConstants {
    int body_id;
    packed_float3 geom_pos;
    packed_float4 geom_quat;
};

struct vec10 {
    float Ixx, Iyy, Izz;
    float Ixy, Ixz, Iyz;
    float mx, my, mz;
    float m;
};

inline vec10 crb_add(vec10 a, vec10 b) {
    vec10 r;
    r.Ixx = a.Ixx + b.Ixx;
    r.Iyy = a.Iyy + b.Iyy;
    r.Izz = a.Izz + b.Izz;
    r.Ixy = a.Ixy + b.Ixy;
    r.Ixz = a.Ixz + b.Ixz;
    r.Iyz = a.Iyz + b.Iyz;
    r.mx = a.mx + b.mx;
    r.my = a.my + b.my;
    r.mz = a.mz + b.mz;
    r.m = a.m + b.m;
    return r;
}

struct spatial_vec {
    float3 w; // angular
    float3 v; // linear

    spatial_vec() : w(float3(0)), v(float3(0)) {}
    spatial_vec(float3 w_, float3 v_) : w(w_), v(v_) {}
};

inline spatial_vec operator+(spatial_vec a, spatial_vec b) {
    return spatial_vec(a.w + b.w, a.v + b.v);
}

inline spatial_vec operator*(spatial_vec a, float s) {
    return spatial_vec(a.w * s, a.v * s);
}

inline float spatial_dot(spatial_vec a, spatial_vec b) {
    return dot(a.w, b.w) + dot(a.v, b.v);
}

inline spatial_vec inert_vec(vec10 i, spatial_vec v) {
    spatial_vec res;
    res.w.x = i.Ixx * v.w.x + i.Ixy * v.w.y + i.Ixz * v.w.z - i.mz * v.v.y + i.my * v.v.z;
    res.w.y = i.Ixy * v.w.x + i.Iyy * v.w.y + i.Iyz * v.w.z + i.mz * v.v.x - i.mx * v.v.z;
    res.w.z = i.Ixz * v.w.x + i.Iyz * v.w.y + i.Izz * v.w.z - i.my * v.v.x + i.mx * v.v.y;
    res.v.x = i.mz * v.w.y - i.my * v.w.z + i.m * v.v.x;
    res.v.y = i.mx * v.w.z - i.mz * v.w.x + i.m * v.v.y;
    res.v.z = i.my * v.w.x - i.mx * v.w.y + i.m * v.v.z;
    return res;
}

inline spatial_vec motion_cross(spatial_vec u, spatial_vec v) {
    spatial_vec res;
    res.w = cross(u.w, v.w);
    res.v = cross(u.v, v.w) + cross(u.w, v.v);
    return res;
}

inline spatial_vec motion_cross_force(spatial_vec v, spatial_vec f) {
    spatial_vec res;
    res.w = cross(v.w, f.w) + cross(v.v, f.v);
    res.v = cross(v.w, f.v);
    return res;
}

// -----------------------------------------------------------------------------
// Math helper functions
// -----------------------------------------------------------------------------

inline float4 quat_mul(float4 q1, float4 q2) {
    return float4(
        q1.x * q2.x - q1.y * q2.y - q1.z * q2.z - q1.w * q2.w,
        q1.x * q2.y + q1.y * q2.x + q1.z * q2.w - q1.w * q2.z,
        q1.x * q2.z - q1.y * q2.w + q1.z * q2.x + q1.w * q2.y,
        q1.x * q2.w + q1.y * q2.z - q1.z * q2.y + q1.w * q2.x
    );
}

inline float3 quat_rot(float4 q, float3 v) {
    float3 q_vec = float3(q.y, q.z, q.w);
    float3 t = 2.0f * cross(q_vec, v);
    return v + q.x * t + cross(q_vec, t);
}

inline float3x3 quat_to_mat(float4 q) {
    float w = q.x, x = q.y, y = q.z, z = q.w;
    return float3x3(
        float3(1.0f - 2.0f*(y*y + z*z), 2.0f*(x*y + w*z), 2.0f*(x*z - w*y)), // col 0
        float3(2.0f*(x*y - w*z), 1.0f - 2.0f*(x*x + z*z), 2.0f*(y*z + w*x)), // col 1
        float3(2.0f*(x*z + w*y), 2.0f*(y*z - w*x), 1.0f - 2.0f*(x*x + y*y))  // col 2
    );
}

inline float3x3 load_row_major_mat3(device const float* ptr) {
    return float3x3(
        float3(ptr[0], ptr[3], ptr[6]), // col 0
        float3(ptr[1], ptr[4], ptr[7]), // col 1
        float3(ptr[2], ptr[5], ptr[8])  // col 2
    );
}

// -----------------------------------------------------------------------------
// 1. Hierarchical Forward Kinematics Kernel
// -----------------------------------------------------------------------------

kernel void kernel_forward_kinematics(
    constant BodyConstants* bodies [[buffer(0)]],      // 17 bodies
    constant GeomConstants* foot_geoms [[buffer(1)]],  // 2 foot geoms (left, right)
    device const float* qpos_batch [[buffer(2)]],      // (B, 21)
    device float* xpos_out [[buffer(3)]],              // (B, 17, 3)
    device float* xmat_out [[buffer(4)]],              // (B, 17, 9) row-major
    device float* geom_xpos_out [[buffer(5)]],         // (B, 2, 3)
    device float* geom_xmat_out [[buffer(6)]],         // (B, 2, 9) row-major
    uint tid [[thread_position_in_grid]]
) {
    uint b_idx = tid;
    device const float* qpos = qpos_batch + b_idx * 21;
    device float* b_xpos = xpos_out + b_idx * 17 * 3;
    device float* b_xmat = xmat_out + b_idx * 17 * 9;
    device float* g_xpos = geom_xpos_out + b_idx * 2 * 3;
    device float* g_xmat = geom_xmat_out + b_idx * 2 * 9;

    float3 xpos[17];
    float4 xquat[17];

    // Body 0: world
    xpos[0] = float3(0.0f);
    xquat[0] = float4(1.0f, 0.0f, 0.0f, 0.0f);

    // Body 1: terrain
    xpos[1] = float3(0.0f);
    xquat[1] = float4(1.0f, 0.0f, 0.0f, 0.0f);

    // Body 2: trunk_base (freejoint)
    xpos[2] = float3(qpos[0], qpos[1], qpos[2]);
    float4 q_raw = float4(qpos[3], qpos[4], qpos[5], qpos[6]);
    xquat[2] = normalize(q_raw);

    // Bodies 3..16 in topological order
    for (int i = 3; i < 17; ++i) {
        int pid = bodies[i].parent_id;
        float3 pos_rel = float3(bodies[i].body_pos);
        float4 quat_rel = float4(bodies[i].body_quat);
        float3 axis = float3(bodies[i].jnt_axis);
        int qadr = bodies[i].qpos_adr;
        float angle = qpos[qadr];

        float half_a = angle * 0.5f;
        float4 q_jnt = float4(cos(half_a), axis * sin(half_a));
        float4 q_b = quat_mul(quat_rel, q_jnt);

        xquat[i] = normalize(quat_mul(xquat[pid], q_b));
        xpos[i] = xpos[pid] + quat_rot(xquat[pid], pos_rel);
    }

    // Write body xpos and row-major xmat
    for (int i = 0; i < 17; ++i) {
        b_xpos[i * 3 + 0] = xpos[i].x;
        b_xpos[i * 3 + 1] = xpos[i].y;
        b_xpos[i * 3 + 2] = xpos[i].z;

        float3x3 m = quat_to_mat(xquat[i]);
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                b_xmat[i * 9 + r * 3 + c] = m[c][r]; // row r, col c
            }
        }
    }

    // Compute foot geom frames (0: left foot, 1: right foot)
    for (int g = 0; g < 2; ++g) {
        int bid = foot_geoms[g].body_id;
        float3 g_rel_pos = float3(foot_geoms[g].geom_pos);
        float4 g_rel_quat = float4(foot_geoms[g].geom_quat);

        float3 pos = xpos[bid] + quat_rot(xquat[bid], g_rel_pos);
        float4 quat = normalize(quat_mul(xquat[bid], g_rel_quat));
        float3x3 m = quat_to_mat(quat);

        g_xpos[g * 3 + 0] = pos.x;
        g_xpos[g * 3 + 1] = pos.y;
        g_xpos[g * 3 + 2] = pos.z;

        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                g_xmat[g * 9 + r * 3 + c] = m[c][r]; // row r, col c
            }
        }
    }
}

// -----------------------------------------------------------------------------
// 1b. Native Articulated Dynamics Kernel (CRBA + RNE + Per-World Randomization)
// -----------------------------------------------------------------------------

kernel void kernel_articulated_dynamics(
    constant BodyConstants* bodies [[buffer(0)]],        // 17 bodies
    constant DofConstants* dofs [[buffer(1)]],           // 20 dofs
    device const float* qpos_batch [[buffer(2)]],        // (B, 21)
    device const float* qvel_batch [[buffer(3)]],        // (B, 20)
    device const float* per_world_mass [[buffer(4)]],    // (B, 17) optional
    device const float* per_world_ipos [[buffer(5)]],    // (B, 17, 3) optional
    device const float* per_world_armature [[buffer(6)]],// (B, 20) optional
    constant int& flags [[buffer(7)]],                   // bit 0: mass, bit 1: ipos, bit 2: armature
    device float* M_eff_out [[buffer(8)]],               // (B, 20, 20)
    device float* qfrc_bias_out [[buffer(9)]],           // (B, 20)
    device float* xpos_out [[buffer(10)]],               // (B, 17, 3)
    device float* xmat_out [[buffer(11)]],               // (B, 17, 9) row-major
    device float* xipos_out [[buffer(12)]],              // (B, 17, 3)
    device float* ximat_out [[buffer(13)]],              // (B, 17, 9) row-major
    device float* subtree_com_out [[buffer(14)]],        // (B, 3)
    uint tid [[thread_position_in_grid]]
) {
    uint b_idx = tid;
    device const float* qpos = qpos_batch + b_idx * 21;
    device const float* qvel = qvel_batch + b_idx * 20;
    device float* M_out = M_eff_out + b_idx * 400;
    device float* bias_out = qfrc_bias_out + b_idx * 20;

    // Per-world parameters
    float mass[17];
    float3 ipos[17];
    float armature[20];

    for (int i = 0; i < 17; ++i) {
        mass[i] = (flags & 1) ? per_world_mass[b_idx * 17 + i] : bodies[i].mass;
        if (flags & 2) {
            ipos[i] = float3(
                per_world_ipos[(b_idx * 17 + i) * 3 + 0],
                per_world_ipos[(b_idx * 17 + i) * 3 + 1],
                per_world_ipos[(b_idx * 17 + i) * 3 + 2]
            );
        } else {
            ipos[i] = float3(bodies[i].body_ipos);
        }
    }
    for (int d = 0; d < 20; ++d) {
        armature[d] = (flags & 4) ? per_world_armature[b_idx * 20 + d] : dofs[d].dof_armature;
    }

    // Kinematics arrays
    float3 xpos[17];
    float4 xquat[17];
    float3x3 xmat[17];
    float3 xipos[17];
    float3x3 ximat[17];
    float3 xaxis[20];
    float3 xanchor[20];

    xpos[0] = float3(0.0f);
    xquat[0] = float4(1.0f, 0.0f, 0.0f, 0.0f);
    xmat[0] = float3x3(1.0f);
    xipos[0] = float3(0.0f);
    ximat[0] = float3x3(1.0f);

    xpos[1] = float3(0.0f);
    xquat[1] = float4(1.0f, 0.0f, 0.0f, 0.0f);
    xmat[1] = float3x3(1.0f);
    xipos[1] = float3(0.0f);
    ximat[1] = float3x3(1.0f);

    // Body 2 (trunk_base, freejoint)
    xpos[2] = float3(qpos[0], qpos[1], qpos[2]);
    xquat[2] = normalize(float4(qpos[3], qpos[4], qpos[5], qpos[6]));
    xmat[2] = quat_to_mat(xquat[2]);
    xipos[2] = xpos[2] + quat_rot(xquat[2], ipos[2]);
    ximat[2] = quat_to_mat(normalize(quat_mul(xquat[2], float4(bodies[2].body_iquat))));

    for (int k = 0; k < 6; ++k) {
        xanchor[k] = xpos[2];
    }

    // Bodies 3..16 (topological order)
    for (int i = 3; i < 17; ++i) {
        int pid = bodies[i].parent_id;
        float3 pos_rel = float3(bodies[i].body_pos);
        float4 quat_rel = float4(bodies[i].body_quat);
        float3 axis = float3(bodies[i].jnt_axis);
        int qadr = bodies[i].qpos_adr;
        int dof = bodies[i].dof_adr;
        float angle = qpos[qadr];

        float half_a = angle * 0.5f;
        float4 q_jnt = float4(cos(half_a), axis * sin(half_a));
        float4 q_b = quat_mul(quat_rel, q_jnt);

        xquat[i] = normalize(quat_mul(xquat[pid], q_b));
        xanchor[dof] = xpos[pid] + quat_rot(xquat[pid], pos_rel);
        xpos[i] = xanchor[dof]; // jnt_pos = 0
        xmat[i] = quat_to_mat(xquat[i]);
        xipos[i] = xpos[i] + quat_rot(xquat[i], ipos[i]);
        ximat[i] = quat_to_mat(normalize(quat_mul(xquat[i], float4(bodies[i].body_iquat))));
        xaxis[dof] = quat_rot(xquat[i], axis);
    }

    // Subtree Center of Mass (robot root is body 2)
    float tot_mass = 0.0f;
    float3 com_num = float3(0.0f);
    for (int i = 2; i < 17; ++i) {
        tot_mass += mass[i];
        com_num += mass[i] * xipos[i];
    }
    float3 subtree_com = (tot_mass > 0.0f) ? (com_num / tot_mass) : float3(0.0f);

    // Spatial Inertias (cinert) in subtree CoM frame
    vec10 cinert[17];
    for (int i = 2; i < 17; ++i) {
        float3 dif = xipos[i] - subtree_com;
        float3x3 mat = ximat[i];
        float3 inert = float3(bodies[i].inertia);
        float3x3 diag_inert = float3x3(
            float3(inert.x, 0, 0),
            float3(0, inert.y, 0),
            float3(0, 0, inert.z)
        );
        float3x3 tmp = mat * diag_inert * transpose(mat);

        vec10 ci;
        ci.Ixx = tmp[0][0] + mass[i] * (dif.y * dif.y + dif.z * dif.z);
        ci.Iyy = tmp[1][1] + mass[i] * (dif.x * dif.x + dif.z * dif.z);
        ci.Izz = tmp[2][2] + mass[i] * (dif.x * dif.x + dif.y * dif.y);
        ci.Ixy = tmp[0][1] - mass[i] * dif.x * dif.y;
        ci.Ixz = tmp[0][2] - mass[i] * dif.x * dif.z;
        ci.Iyz = tmp[1][2] - mass[i] * dif.y * dif.z;
        ci.mx = mass[i] * dif.x;
        ci.my = mass[i] * dif.y;
        ci.mz = mass[i] * dif.z;
        ci.m = mass[i];
        cinert[i] = ci;
    }

    // Spatial Motion DOFs (cdof) in subtree CoM frame
    spatial_vec cdof[20];
    cdof[0] = spatial_vec(float3(0), float3(1, 0, 0));
    cdof[1] = spatial_vec(float3(0), float3(0, 1, 0));
    cdof[2] = spatial_vec(float3(0), float3(0, 0, 1));

    float3 offset_root = subtree_com - xanchor[0];
    cdof[3] = spatial_vec(xmat[2][0], cross(xmat[2][0], offset_root));
    cdof[4] = spatial_vec(xmat[2][1], cross(xmat[2][1], offset_root));
    cdof[5] = spatial_vec(xmat[2][2], cross(xmat[2][2], offset_root));

    for (int d = 6; d < 20; ++d) {
        float3 offset = subtree_com - xanchor[d];
        float3 ax = xaxis[d];
        cdof[d] = spatial_vec(ax, cross(ax, offset));
    }

    // CRBA: Composite Rigid Body Inertias
    vec10 crb[17];
    for (int b = 2; b < 17; ++b) {
        crb[b] = cinert[b];
    }
    for (int b = 16; b >= 3; --b) {
        int pid = bodies[b].parent_id;
        crb[pid] = crb_add(crb[pid], crb[b]);
    }

    // Form M(q) including configured armature
    for (int i = 0; i < 400; ++i) {
        M_out[i] = 0.0f;
    }

    for (int i = 0; i < 20; ++i) {
        int bid = dofs[i].dof_bodyid;
        spatial_vec buf = inert_vec(crb[bid], cdof[i]);
        M_out[i * 20 + i] = armature[i] + spatial_dot(cdof[i], buf);

        int j = dofs[i].dof_parentid;
        while (j >= 0) {
            float val = spatial_dot(cdof[j], buf);
            M_out[i * 20 + j] = val;
            M_out[j * 20 + i] = val;
            j = dofs[j].dof_parentid;
        }
    }

    // RNE: Coriolis, centrifugal, gravity bias forces
    spatial_vec cvel[17];
    cvel[0] = spatial_vec(float3(0), float3(0));
    cvel[1] = spatial_vec(float3(0), float3(0));
    cvel[2] = spatial_vec(float3(0), float3(0));
    for (int k = 0; k < 6; ++k) {
        cvel[2] = cvel[2] + cdof[k] * qvel[k];
    }
    for (int b = 3; b < 17; ++b) {
        int pid = bodies[b].parent_id;
        int dof = bodies[b].dof_adr;
        cvel[b] = cvel[pid] + cdof[dof] * qvel[dof];
    }

    spatial_vec cdof_dot[20];
    cdof_dot[0] = spatial_vec(float3(0), float3(0));
    cdof_dot[1] = spatial_vec(float3(0), float3(0));
    cdof_dot[2] = spatial_vec(float3(0), float3(0));
    for (int k = 3; k < 6; ++k) {
        cdof_dot[k] = motion_cross(cvel[2], cdof[k]);
    }
    for (int d = 6; d < 20; ++d) {
        int b = dofs[d].dof_bodyid;
        cdof_dot[d] = motion_cross(cvel[b], cdof[d]);
    }

    spatial_vec cacc[17];
    cacc[0] = spatial_vec(float3(0), float3(0, 0, 9.81f)); // -gravity
    cacc[1] = cacc[0];
    cacc[2] = cacc[0];
    for (int k = 0; k < 6; ++k) {
        cacc[2] = cacc[2] + cdof_dot[k] * qvel[k];
    }
    for (int b = 3; b < 17; ++b) {
        int pid = bodies[b].parent_id;
        int dof = bodies[b].dof_adr;
        cacc[b] = cacc[pid] + cdof_dot[dof] * qvel[dof];
    }

    spatial_vec cfrc[17];
    cfrc[0] = spatial_vec(float3(0), float3(0));
    cfrc[1] = spatial_vec(float3(0), float3(0));
    for (int b = 2; b < 17; ++b) {
        spatial_vec iv_acc = inert_vec(cinert[b], cacc[b]);
        spatial_vec iv_vel = inert_vec(cinert[b], cvel[b]);
        cfrc[b] = iv_acc + motion_cross_force(cvel[b], iv_vel);
    }

    for (int b = 16; b >= 3; --b) {
        int pid = bodies[b].parent_id;
        cfrc[pid] = cfrc[pid] + cfrc[b];
    }

    for (int d = 0; d < 20; ++d) {
        int b = dofs[d].dof_bodyid;
        bias_out[d] = spatial_dot(cdof[d], cfrc[b]);
    }

    // Write auxiliary kinematics outputs to device buffers
    device float* b_xpos = xpos_out + b_idx * 17 * 3;
    device float* b_xmat = xmat_out + b_idx * 17 * 9;
    device float* b_xipos = xipos_out + b_idx * 17 * 3;
    device float* b_ximat = ximat_out + b_idx * 17 * 9;
    device float* b_com = subtree_com_out + b_idx * 3;

    b_com[0] = subtree_com.x;
    b_com[1] = subtree_com.y;
    b_com[2] = subtree_com.z;

    for (int i = 0; i < 17; ++i) {
        b_xpos[i * 3 + 0] = xpos[i].x;
        b_xpos[i * 3 + 1] = xpos[i].y;
        b_xpos[i * 3 + 2] = xpos[i].z;

        b_xipos[i * 3 + 0] = xipos[i].x;
        b_xipos[i * 3 + 1] = xipos[i].y;
        b_xipos[i * 3 + 2] = xipos[i].z;

        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                b_xmat[i * 9 + r * 3 + c] = xmat[i][c][r]; // row-major
                b_ximat[i * 9 + r * 3 + c] = ximat[i][c][r]; // row-major
            }
        }
    }
}

// -----------------------------------------------------------------------------
// 2. CAD Sole Contact Manifold Kernel
// -----------------------------------------------------------------------------

kernel void kernel_cad_contact_manifold(
    device const float* geom_xpos_batch [[buffer(0)]],      // (B, 2, 3)
    device const float* geom_xmat_batch [[buffer(1)]],      // (B, 2, 9) row-major
    constant const float* left_mesh_verts [[buffer(2)]],    // (N_L, 3)
    constant const float* right_mesh_verts [[buffer(3)]],   // (N_R, 3)
    constant int& num_left_verts [[buffer(4)]],             // N_L
    constant int& num_right_verts [[buffer(5)]],            // N_R
    device float* contact_pos_out [[buffer(6)]],            // (B, nconmax, 3)
    device float* contact_dist_out [[buffer(7)]],           // (B, nconmax)
    device float* contact_normal_out [[buffer(8)]],         // (B, nconmax, 3)
    device int* contact_body_out [[buffer(9)]],             // (B, nconmax) body ID for contact
    device int* ncon_out [[buffer(10)]],                    // (B,)
    device int* overflow_flag_out [[buffer(11)]],           // (B,)
    constant int& nconmax [[buffer(12)]],                   // capacity
    uint tid [[thread_position_in_grid]]
) {
    uint b_idx = tid;
    device const float* g_xpos = geom_xpos_batch + b_idx * 2 * 3;
    device const float* g_xmat = geom_xmat_batch + b_idx * 2 * 9;

    device float* out_pos = contact_pos_out + b_idx * nconmax * 3;
    device float* out_dist = contact_dist_out + b_idx * nconmax;
    device float* out_norm = contact_normal_out + b_idx * nconmax * 3;
    device int* out_body = contact_body_out + b_idx * nconmax;

    int total_contacts = 0;
    int overflow = 0;

    // Process left foot (geom 0, body 7) and right foot (geom 1, body 16)
    for (int foot = 0; foot < 2; ++foot) {
        float3 pos_g = float3(g_xpos[foot * 3 + 0], g_xpos[foot * 3 + 1], g_xpos[foot * 3 + 2]);
        float3x3 mat_g = load_row_major_mat3(g_xmat + foot * 9);

        constant const float* verts = (foot == 0) ? left_mesh_verts : right_mesh_verts;
        int nverts = (foot == 0) ? num_left_verts : num_right_verts;
        int body_id = (foot == 0) ? 7 : 16;

        // 1. Find deepest penetrating vertex (point a)
        float min_z = 1e6f;
        int idx_a = -1;
        float3 w_a = float3(0.0f);

        for (int v = 0; v < nverts; ++v) {
            float3 local_v = float3(verts[v * 3 + 0], verts[v * 3 + 1], verts[v * 3 + 2]);
            float3 w_v = pos_g + mat_g * local_v;
            if (w_v.z < min_z) {
                min_z = w_v.z;
                idx_a = v;
                w_a = w_v;
            }
        }

        // If foot does not penetrate ground plane (z=0), skip
        if (min_z >= 0.0f || idx_a < 0) {
            continue;
        }

        // Support threshold (1 mm above deepest)
        float threshold = min_z + 1e-3f;

        // 2. Find vertex b furthest from a in xy
        float max_d_ab = -1e6f;
        int idx_b = -1;
        float3 w_b = w_a;

        for (int v = 0; v < nverts; ++v) {
            float3 local_v = float3(verts[v * 3 + 0], verts[v * 3 + 1], verts[v * 3 + 2]);
            float3 w_v = pos_g + mat_g * local_v;
            if (w_v.z <= threshold) {
                float d2 = (w_v.x - w_a.x)*(w_v.x - w_a.x) + (w_v.y - w_a.y)*(w_v.y - w_a.y);
                if (d2 > max_d_ab) {
                    max_d_ab = d2;
                    idx_b = v;
                    w_b = w_v;
                }
            }
        }

        // 3. Find vertex c furthest from line a-b
        float2 ab = float2(w_b.x - w_a.x, w_b.y - w_a.y);
        float ab_len = length(ab);
        float max_d_c = -1e6f;
        int idx_c = -1;
        float3 w_c = w_a;

        if (ab_len > 1e-4f) {
            float2 ab_unit = ab / ab_len;
            float2 perp = float2(-ab_unit.y, ab_unit.x);
            for (int v = 0; v < nverts; ++v) {
                float3 local_v = float3(verts[v * 3 + 0], verts[v * 3 + 1], verts[v * 3 + 2]);
                float3 w_v = pos_g + mat_g * local_v;
                if (w_v.z <= threshold) {
                    float dist_line = abs(dot(float2(w_v.x - w_a.x, w_v.y - w_a.y), perp));
                    if (dist_line > max_d_c) {
                        max_d_c = dist_line;
                        idx_c = v;
                        w_c = w_v;
                    }
                }
            }
        }

        // Assemble unique contact points for this foot (up to 3)
        float3 foot_pts[3];
        float foot_dists[3];
        int num_foot_con = 0;

        foot_pts[num_foot_con] = float3(w_a.x, w_a.y, w_a.z * 0.5f);
        foot_dists[num_foot_con] = w_a.z;
        num_foot_con++;

        if (idx_b >= 0 && idx_b != idx_a && max_d_ab > 1e-6f) {
            foot_pts[num_foot_con] = float3(w_b.x, w_b.y, w_b.z * 0.5f);
            foot_dists[num_foot_con] = w_b.z;
            num_foot_con++;
        }

        if (idx_c >= 0 && idx_c != idx_a && idx_c != idx_b && max_d_c > 1e-4f) {
            foot_pts[num_foot_con] = float3(w_c.x, w_c.y, w_c.z * 0.5f);
            foot_dists[num_foot_con] = w_c.z;
            num_foot_con++;
        }

        // Store into batch buffer with explicit capacity check
        for (int c = 0; c < num_foot_con; ++c) {
            if (total_contacts < nconmax) {
                out_pos[total_contacts * 3 + 0] = foot_pts[c].x;
                out_pos[total_contacts * 3 + 1] = foot_pts[c].y;
                out_pos[total_contacts * 3 + 2] = foot_pts[c].z;

                out_dist[total_contacts] = foot_dists[c];

                out_norm[total_contacts * 3 + 0] = 0.0f;
                out_norm[total_contacts * 3 + 1] = 0.0f;
                out_norm[total_contacts * 3 + 2] = 1.0f;

                out_body[total_contacts] = body_id;
                total_contacts++;
            } else {
                overflow = 1;
            }
        }
    }

    ncon_out[b_idx] = total_contacts;
    overflow_flag_out[b_idx] = overflow;
}

// -----------------------------------------------------------------------------
// 3. Constrained Solve Kernel
// -----------------------------------------------------------------------------

kernel void kernel_constrained_solve(
    device const float* M_inv_batch [[buffer(0)]],          // (B, 20, 20)
    device const float* qfrc_bias_batch [[buffer(1)]],      // (B, 20)
    device const float* contact_pos_batch [[buffer(2)]],    // (B, nconmax, 3)
    device const float* contact_dist_batch [[buffer(3)]],   // (B, nconmax)
    device const int* contact_body_batch [[buffer(4)]],     // (B, nconmax)
    device const int* ncon_batch [[buffer(5)]],             // (B,)
    device const float* body_xpos_batch [[buffer(6)]],      // (B, 17, 3)
    device const float* body_xmat_batch [[buffer(7)]],      // (B, 17, 9) row-major
    constant BodyConstants* bodies [[buffer(8)]],           // 17 bodies
    constant float& friction_coef [[buffer(9)]],            // mu
    constant int& nconmax [[buffer(10)]],                   // capacity
    device const float* qvel_batch [[buffer(11)]],          // (B, 20)
    device float* qacc_out [[buffer(12)]],                  // (B, 20)
    device float* qfrc_constraint_out [[buffer(13)]],       // (B, 20)
    uint tid [[thread_position_in_grid]]
) {
    uint b_idx = tid;
    int ncon = ncon_batch[b_idx];
    if (ncon <= 0) {
        device const float* M_inv = M_inv_batch + b_idx * 400;
        device const float* bias = qfrc_bias_batch + b_idx * 20;
        device float* qacc = qacc_out + b_idx * 20;
        device float* qfrc_c = qfrc_constraint_out + b_idx * 20;

        for (int i = 0; i < 20; ++i) {
            qfrc_c[i] = 0.0f;
            float sum = 0.0f;
            for (int j = 0; j < 20; ++j) {
                sum += M_inv[i * 20 + j] * (-bias[j]);
            }
            qacc[i] = sum;
        }
        return;
    }

    int nefc = ncon * 4;
    if (nefc > 32) nefc = 32;

    device const float* M_inv = M_inv_batch + b_idx * 400;
    device const float* bias = qfrc_bias_batch + b_idx * 20;
    device const float* c_pos = contact_pos_batch + b_idx * nconmax * 3;
    device const float* c_dist = contact_dist_batch + b_idx * nconmax;
    device const int* c_body = contact_body_batch + b_idx * nconmax;
    device const float* b_xpos = body_xpos_batch + b_idx * 17 * 3;
    device const float* b_xmat = body_xmat_batch + b_idx * 17 * 9;
    device const float* qvel = qvel_batch + b_idx * 20;
    device float* qacc = qacc_out + b_idx * 20;
    device float* qfrc_c = qfrc_constraint_out + b_idx * 20;

    // 1. Assemble contact Jacobian J (nefc x 20)
    float J[32][20];
    float aref[32];
    float D_diag[32];

    float3 base_pos = float3(b_xpos[2 * 3 + 0], b_xpos[2 * 3 + 1], b_xpos[2 * 3 + 2]);

    for (int c = 0; c < ncon && c * 4 < 32; ++c) {
        int efc_adr = c * 4;
        float3 p = float3(c_pos[c * 3 + 0], c_pos[c * 3 + 1], c_pos[c * 3 + 2]);
        float dist = c_dist[c];
        int body_id = c_body[c];

        float J_p[3][20];
        for (int r = 0; r < 3; ++r) {
            for (int col = 0; col < 20; ++col) {
                J_p[r][col] = 0.0f;
            }
        }

        // Freejoint linear DOFs (0, 1, 2)
        J_p[0][0] = 1.0f;
        J_p[1][1] = 1.0f;
        J_p[2][2] = 1.0f;

        // Freejoint angular DOFs (3, 4, 5): -r x
        float3 r = p - base_pos;
        J_p[0][3] = 0.0f;    J_p[0][4] = r.z;    J_p[0][5] = -r.y;
        J_p[1][3] = -r.z;   J_p[1][4] = 0.0f;   J_p[1][5] = r.x;
        J_p[2][3] = r.y;    J_p[2][4] = -r.x;   J_p[2][5] = 0.0f;

        // Ancestor hinge joints
        int curr_b = body_id;
        while (curr_b > 2) {
            int dof_adr = bodies[curr_b].dof_adr;
            float3 local_axis = float3(bodies[curr_b].jnt_axis);
            float3x3 mat_b = load_row_major_mat3(b_xmat + curr_b * 9);
            float3 world_axis = mat_b * local_axis;
            float3 anchor = float3(b_xpos[curr_b * 3 + 0], b_xpos[curr_b * 3 + 1], b_xpos[curr_b * 3 + 2]);
            float3 r_j = p - anchor;
            float3 col = cross(world_axis, r_j);

            J_p[0][dof_adr] = col.x;
            J_p[1][dof_adr] = col.y;
            J_p[2][dof_adr] = col.z;

            curr_b = bodies[curr_b].parent_id;
        }

        // Pyramidal cone rows (condim=3)
        // Normal = z, tangent1 = y, tangent2 = -x
        float mu = friction_coef;
        for (int col = 0; col < 20; ++col) {
            J[efc_adr + 0][col] = J_p[2][col] + mu * J_p[1][col];
            J[efc_adr + 1][col] = J_p[2][col] - mu * J_p[1][col];
            J[efc_adr + 2][col] = J_p[2][col] - mu * J_p[0][col];
            J[efc_adr + 3][col] = J_p[2][col] + mu * J_p[0][col];
        }

        float timeconst = 0.02f;
        float dampratio = 1.0f;
        float dmax = 0.95f;
        float width = 0.001f;

        float k = 1.0f / (dmax * dmax * timeconst * timeconst * dampratio * dampratio);
        float b_damp = 2.0f / (dmax * timeconst);
        float imp_x = abs(dist) / width;
        float imp = (imp_x > 1.0f) ? dmax : 0.9f;

        float invweight0 = 33.333333f;
        float invweight_pyr = (invweight0 + mu * mu * invweight0) * 2.0f * mu * mu;
        float D_val = 1.0f / (invweight_pyr * (1.0f - imp) / imp);

        for (int r = 0; r < 4; ++r) {
            float vel_r = 0.0f;
            for (int col = 0; col < 20; ++col) {
                vel_r += J[efc_adr + r][col] * qvel[col];
            }
            aref[efc_adr + r] = -k * imp * dist - b_damp * vel_r;
            D_diag[efc_adr + r] = D_val;
        }
    }

    // 2. Unconstrained acceleration qacc_0 = -M_inv * bias
    float qacc_0[20];
    for (int i = 0; i < 20; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < 20; ++j) {
            sum += M_inv[i * 20 + j] * (-bias[j]);
        }
        qacc_0[i] = sum;
    }

    // 3. Free constraint acceleration a_0 = J * qacc_0 - aref
    float a_0[32];
    for (int i = 0; i < nefc; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < 20; ++j) {
            sum += J[i][j] * qacc_0[j];
        }
        a_0[i] = sum - aref[i];
    }

    // 4. Delassus matrix A = J * M_inv * J^T + diag(1/D)
    float A[32][32];
    float J_Minv[32][20];

    for (int i = 0; i < nefc; ++i) {
        for (int j = 0; j < 20; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < 20; ++k) {
                sum += J[i][k] * M_inv[k * 20 + j];
            }
            J_Minv[i][j] = sum;
        }
    }

    for (int i = 0; i < nefc; ++i) {
        for (int j = 0; j < nefc; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < 20; ++k) {
                sum += J_Minv[i][k] * J[j][k];
            }
            if (i == j) {
                sum += 1.0f / D_diag[i];
            }
            A[i][j] = sum;
        }
    }

    // 5. Projected Gauss-Seidel (PGS) solve for lambda >= 0
    float lambda[32];
    for (int i = 0; i < nefc; ++i) {
        lambda[i] = 0.0f;
    }

    for (int iter = 0; iter < 100; ++iter) {
        for (int i = 0; i < nefc; ++i) {
            float row_dot = 0.0f;
            for (int j = 0; j < nefc; ++j) {
                row_dot += A[i][j] * lambda[j];
            }
            float delta = -(a_0[i] + row_dot) / A[i][i];
            lambda[i] = max(0.0f, lambda[i] + delta);
        }
    }

    // 6. Compute constraint force qfrc_constraint = J^T * lambda
    for (int j = 0; j < 20; ++j) {
        float sum = 0.0f;
        for (int i = 0; i < nefc; ++i) {
            sum += J[i][j] * lambda[i];
        }
        qfrc_c[j] = sum;
    }

    // 7. Solved acceleration qacc = qacc_0 + M_inv * qfrc_constraint
    for (int i = 0; i < 20; ++i) {
        float sum = qacc_0[i];
        for (int j = 0; j < 20; ++j) {
            sum += M_inv[i * 20 + j] * qfrc_c[j];
        }
        qacc[i] = sum;
    }
}

// -----------------------------------------------------------------------------
// 4. Native Cholesky Factorization and Multi-RHS Linear Solve Kernel
// -----------------------------------------------------------------------------

kernel void kernel_cholesky_solve(
    device const float* M_batch [[buffer(0)]],      // (B, 20, 20)
    device const float* B_batch [[buffer(1)]],      // (B, 20, K)
    constant int& K [[buffer(2)]],                  // number of RHS columns
    device float* L_out [[buffer(3)]],              // (B, 20, 20)
    device float* X_out [[buffer(4)]],              // (B, 20, K)
    device int* status_out [[buffer(5)]],           // (B,) 0: success, -1: pivot/NaN failure
    uint tid [[thread_position_in_grid]]
) {
    uint b_idx = tid;
    device const float* M = M_batch + b_idx * 400;
    device const float* B = B_batch + b_idx * 20 * K;
    device float* L = L_out + b_idx * 400;
    device float* X = X_out + b_idx * 20 * K;

    float L_loc[20][20];
    for (int i = 0; i < 20; ++i) {
        for (int j = 0; j < 20; ++j) {
            L_loc[i][j] = 0.0f;
        }
    }

    status_out[b_idx] = 0;

    // Cholesky factorization: M = L * L^T
    for (int j = 0; j < 20; ++j) {
        float sum_sq = 0.0f;
        for (int k = 0; k < j; ++k) {
            sum_sq += L_loc[j][k] * L_loc[j][k];
        }
        float s = M[j * 20 + j] - sum_sq;
        if (s <= 0.0f || isnan(s)) {
            status_out[b_idx] = -1; // non-positive pivot or NaN detected
            return;
        }
        float diag = sqrt(s);
        L_loc[j][j] = diag;
        float inv_diag = 1.0f / diag;

        for (int i = j + 1; i < 20; ++i) {
            float sum_prod = 0.0f;
            for (int k = 0; k < j; ++k) {
                sum_prod += L_loc[i][k] * L_loc[j][k];
            }
            L_loc[i][j] = (M[i * 20 + j] - sum_prod) * inv_diag;
        }
    }

    // Write factor L to device output
    for (int i = 0; i < 20; ++i) {
        for (int j = 0; j < 20; ++j) {
            L[i * 20 + j] = L_loc[i][j];
        }
    }

    // Solve M X = B for each column c in [0, K-1]
    float Y_loc[20];
    for (int c = 0; c < K; ++c) {
        // Forward solve: L Y = B
        for (int i = 0; i < 20; ++i) {
            float s = B[i * K + c];
            for (int k = 0; k < i; ++k) {
                s -= L_loc[i][k] * Y_loc[k];
            }
            Y_loc[i] = s / L_loc[i][i];
        }

        // Back solve: L^T X = Y
        for (int i = 19; i >= 0; --i) {
            float s = Y_loc[i];
            for (int k = i + 1; k < 20; ++k) {
                s -= L_loc[k][i] * X[k * K + c];
            }
            X[i * K + c] = s / L_loc[i][i];
        }
    }
}
