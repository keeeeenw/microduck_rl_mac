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
    packed_float3 jnt_axis;
    float mass;
    packed_float3 inertia;
};

struct GeomConstants {
    int body_id;
    packed_float3 geom_pos;
    packed_float4 geom_quat;
};

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
