"""Test CAD Contact Manifold and Constraint Parameter Assembly vs Pinned MuJoCo 3.10.0.

Validates:
1. Exact CPU constraint parameter equations (aref, R, D, J) across all 25 corpus fixtures.
2. Exact plane-convex contact manifold selection (mjc_PlaneConvex hull-graph traversal) in Python.
3. Native Metal CAD contact manifold kernel (kernel_cad_contact_manifold_v2).
4. Native Metal constraint assembly kernel (kernel_assemble_contact_constraints).
"""

import sys
from pathlib import Path
import numpy as np
import pytest
import mujoco

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.canonical_model_loader import load_canonical_model

CORPUS_DIR = PROJECT_ROOT / "corpus"


def get_all_scenarios():
    meta_file = CORPUS_DIR / "corpus_metadata.json"
    if meta_file.exists():
        import json
        return json.loads(meta_file.read_text()).get("scenarios", [])
    return sorted([p.stem for p in CORPUS_DIR.glob("*.npz")])


@pytest.mark.parametrize("sc_name", get_all_scenarios())
def test_exact_cpu_constraint_parameter_equations(sc_name):
    """Verifies exact MuJoCo 3.10.0 C constraint parameter formulas against oracle fixtures."""
    npz = np.load(CORPUS_DIR / f"{sc_name}.npz")
    nefc = int(npz["nefc"])
    if nefc == 0:
        return

    ncon = int(npz["ncon"])
    assert nefc == 4 * ncon, f"Expected nefc == 4 * ncon, got {nefc} vs {4 * ncon}"

    canonical = load_canonical_model()
    m = canonical.model
    body_invw = npz["body_invweight0"].reshape(m.nbody, 2)
    impratio = float(m.opt.impratio)

    efc_aref_ref = npz["efc_aref"]
    efc_R_ref = npz["efc_R"]
    efc_D_ref = npz["efc_D"]
    efc_J_ref = npz["efc_J"]
    efc_vel_ref = npz["efc_vel"]

    contact_g1 = npz["contact_geom1"]
    contact_g2 = npz["contact_geom2"]
    contact_pos = npz["contact_pos"]
    contact_dist = npz["contact_dist"]
    contact_frame = npz["contact_frame"]
    contact_friction = npz["contact_friction"]
    contact_solref = npz["contact_solref"]
    contact_solimp = npz["contact_solimp"]

    # Reconstruct state on CPU MuJoCo to compute jacp
    d = mujoco.MjData(m)
    d.qpos[:] = npz["qpos"]
    d.qvel[:] = npz["qvel"]
    mujoco.mj_forward(m, d)

    for c_idx in range(ncon):
        g1 = contact_g1[c_idx]
        g2 = contact_g2[c_idx]
        b1 = m.geom_bodyid[g1]
        b2 = m.geom_bodyid[g2]

        # 1. Diagonal approximation from body inverse weights
        tran = body_invw[b1, 0] + body_invw[b2, 0]
        mu0 = contact_friction[c_idx, 0]
        dA0 = tran + (mu0 ** 2) * tran

        # 2. Impedance spline
        solimp = contact_solimp[c_idx]
        dmin, dmax, width, midpoint, power = solimp[0], solimp[1], solimp[2], solimp[3], solimp[4]
        dist = contact_dist[c_idx]
        margin = float(npz["efc_margin"][4 * c_idx]) if "efc_margin" in npz else 0.0

        x = abs(dist - margin) / width
        if x <= 0:
            imp = dmin
        elif x >= 1.0:
            imp = dmax
        elif x < midpoint:
            a_imp = 1.0 / (midpoint ** (power - 1))
            imp = dmin + a_imp * (x ** power) * (dmax - dmin)
        else:
            b_imp = 1.0 / ((1.0 - midpoint) ** (power - 1))
            imp = dmin + (1.0 - b_imp * ((1.0 - x) ** power)) * (dmax - dmin)

        # 3. Regularization R and D
        R0 = (1.0 - imp) * dA0 / imp
        R1 = R0 / max(1e-14, impratio)
        mu_reg = mu0 * np.sqrt(R1 / R0)
        Rpy = 2.0 * (mu_reg ** 2) * R0
        Dpy = 1.0 / Rpy

        # 4. Reference acceleration spring-damper constants
        solref = contact_solref[c_idx]
        timeconst = solref[0]
        dampratio = solref[1]
        K = 1.0 / (dmax ** 2 * timeconst ** 2 * dampratio ** 2)
        B = 2.0 / (dmax * timeconst)

        # 5. Facet directions and Jacobians
        frame = contact_frame[c_idx].reshape(3, 3)
        n = frame[0]
        t1 = frame[1]
        t2 = frame[2]
        mu1 = contact_friction[c_idx, 0]
        mu2 = contact_friction[c_idx, 1]

        dirs = [
            n + mu1 * t1,
            n - mu1 * t1,
            n + mu2 * t2,
            n - mu2 * t2,
        ]

        jacp = np.zeros((3, m.nv), dtype=np.float64)
        jacr = np.zeros((3, m.nv), dtype=np.float64)
        mujoco.mj_jac(m, d, jacp, jacr, contact_pos[c_idx], b2)

        for k in range(4):
            row = 4 * c_idx + k

            # Jacobian
            J_calc = dirs[k] @ jacp
            np.testing.assert_allclose(
                J_calc, efc_J_ref[row], atol=1e-7,
                err_msg=f"{sc_name} row {row} Jacobian mismatch"
            )

            # Relative velocity
            vel_calc = float(J_calc @ d.qvel)
            np.testing.assert_allclose(
                vel_calc, efc_vel_ref[row], atol=1e-7,
                err_msg=f"{sc_name} row {row} relative velocity mismatch"
            )

            # aref
            aref_calc = -B * vel_calc - K * imp * (dist - margin)
            np.testing.assert_allclose(
                aref_calc, efc_aref_ref[row], atol=1e-7,
                err_msg=f"{sc_name} row {row} aref mismatch"
            )

            # Regularization
            np.testing.assert_allclose(
                Rpy, efc_R_ref[row], atol=1e-7,
                err_msg=f"{sc_name} row {row} R mismatch"
            )
            np.testing.assert_allclose(
                Dpy, efc_D_ref[row], atol=1e-7,
                err_msg=f"{sc_name} row {row} D mismatch"
            )


@pytest.mark.parametrize("sc_name", get_all_scenarios())
def test_pinned_mujoco_contact_manifold_python_parity(sc_name):
    """Verifies that the Python port of mjc_PlaneConvex matches MuJoCo 3.10.0 contacts exactly."""
    npz = np.load(CORPUS_DIR / f"{sc_name}.npz")
    canonical = load_canonical_model()
    m = canonical.model
    d = mujoco.MjData(m)
    d.qpos[:] = npz["qpos"]
    d.qvel[:] = npz["qvel"]
    mujoco.mj_forward(m, d)

    def run_mjc_plane_convex(geom_id):
        pos1 = d.geom_xpos[0]
        mat1 = d.geom_xmat[0].reshape(3, 3)
        pos2 = d.geom_xpos[geom_id]
        mat2 = d.geom_xmat[geom_id].reshape(3, 3)

        normal = mat1[:, 2] # [0, 0, 1]
        ccd_dir = -normal
        locdir = mat2.T @ ccd_dir

        dataid = m.geom_dataid[geom_id]
        vertadr = m.mesh_vertadr[dataid]
        vertnum = m.mesh_vertnum[dataid]
        vertdata = m.mesh_vert[vertadr : vertadr + vertnum].reshape(-1, 3)

        graphadr = m.mesh_graphadr[dataid]
        numvert = m.mesh_graph[graphadr]
        vert_edgeadr = m.mesh_graph[graphadr + 2 : graphadr + 2 + numvert]
        vert_globalid = m.mesh_graph[graphadr + 2 + numvert : graphadr + 2 + 2 * numvert]
        edge_localid = m.mesh_graph[graphadr + 2 + 2 * numvert :]

        # 1. Primary support vertex
        dots = vertdata[vert_globalid] @ locdir
        ibest_hull = int(np.argmax(dots))
        ibest_global = int(vert_globalid[ibest_hull])

        margin = 0.0
        best_world = pos2 + mat2 @ vertdata[ibest_global]
        dist0 = float(np.dot(normal, best_world - pos1))
        if dist0 > margin:
            return []

        pos0 = best_world - 0.5 * dist0 * normal
        contacts = [(ibest_global, dist0, pos0)]

        # 2. Additional contacts via graph traversal
        rbound = float(m.geom_rbound[geom_id])
        tolplanemesh = 0.3
        threshold = float(np.dot(normal, pos2 - pos1)) - margin

        i = int(vert_edgeadr[ibest_hull])
        while i < len(edge_localid) and edge_localid[i] >= 0 and len(contacts) < 3:
            locid = edge_localid[i]
            v_global = int(vert_globalid[locid])
            v = vertdata[v_global]
            vdot = float(np.dot(v, locdir))
            if vdot > threshold:
                pnt = pos2 + mat2 @ v
                if np.linalg.norm(pnt - pos0) >= tolplanemesh * rbound:
                    dif = pnt - pos1
                    c_dist = float(np.dot(normal, dif))
                    c_pos = pnt - 0.5 * c_dist * normal
                    contacts.append((v_global, c_dist, c_pos))
            i += 1
        return contacts

    left_contacts = run_mjc_plane_convex(canonical.left_foot_geom_id)
    right_contacts = run_mjc_plane_convex(canonical.right_foot_geom_id)
    total_cand = len(left_contacts) + len(right_contacts)

    assert total_cand == d.ncon, f"{sc_name}: cand contact count {total_cand} != MuJoCo {d.ncon}"

    # Verify positions and distances
    c_idx = 0
    for v_id, dist, pos in left_contacts:
        np.testing.assert_allclose(pos, d.contact[c_idx].pos, atol=1e-7, err_msg=f"{sc_name} left pos mismatch")
        np.testing.assert_allclose(dist, d.contact[c_idx].dist, atol=1e-7, err_msg=f"{sc_name} left dist mismatch")
        c_idx += 1

    for v_id, dist, pos in right_contacts:
        np.testing.assert_allclose(pos, d.contact[c_idx].pos, atol=1e-7, err_msg=f"{sc_name} right pos mismatch")
        np.testing.assert_allclose(dist, d.contact[c_idx].dist, atol=1e-7, err_msg=f"{sc_name} right dist mismatch")
        c_idx += 1


@pytest.fixture(scope="module")
def physics_slice():
    import torch
    from src.representative_physics_slice import RepresentativePhysicsSlice
    return RepresentativePhysicsSlice(batch_size=1)


@pytest.mark.parametrize("sc_name", get_all_scenarios())
def test_metal_cad_contact_manifold_v2_parity(physics_slice, sc_name):
    """Verifies that the Metal CAD contact manifold kernel matches MuJoCo 3.10.0 contacts."""
    import torch
    ps = physics_slice
    d_npz = np.load(CORPUS_DIR / f"{sc_name}.npz")
    qpos = torch.from_numpy(d_npz["qpos"].astype(np.float32)).unsqueeze(0).to(ps.device)
    qvel = torch.from_numpy(d_npz["qvel"].astype(np.float32)).unsqueeze(0).to(ps.device)

    # 1. Forward kinematics
    ps.km.launch(
        "kernel_forward_kinematics",
        ps.bodies_buf,
        ps.geoms_buf,
        qpos,
        ps.body_xpos,
        ps.body_xmat,
        ps.geom_xpos,
        ps.geom_xmat,
        threads=1,
    )

    # 2. Contact manifold v2
    c_pos, c_dist, c_norm, c_body, c_geom, ncon, c_overflow = ps.compute_cad_contact_manifold_v2(
        ps.geom_xpos,
        ps.geom_xmat,
    )

    ncon_val = int(ncon.cpu().item())
    ncon_ref = int(d_npz["ncon"])
    assert ncon_val == ncon_ref, f"{sc_name}: Metal ncon {ncon_val} != MuJoCo {ncon_ref}"
    assert int(c_overflow.cpu().item()) == 0, f"{sc_name}: unexpected contact overflow"

    if ncon_val > 0:
        np.testing.assert_allclose(
            c_pos[0, :ncon_val].cpu().numpy(),
            d_npz["contact_pos"],
            atol=1e-5,
            err_msg=f"{sc_name}: contact pos mismatch",
        )
        np.testing.assert_allclose(
            c_dist[0, :ncon_val].cpu().numpy(),
            d_npz["contact_dist"],
            atol=1e-5,
            err_msg=f"{sc_name}: contact dist mismatch",
        )


@pytest.mark.parametrize("sc_name", get_all_scenarios())
def test_metal_constraint_assembly_parity(physics_slice, sc_name):
    """Verifies that the Metal constraint assembly kernel matches MuJoCo 3.10.0 J, aref, R."""
    import torch
    ps = physics_slice
    d_npz = np.load(CORPUS_DIR / f"{sc_name}.npz")
    qpos = torch.from_numpy(d_npz["qpos"].astype(np.float32)).unsqueeze(0).to(ps.device)
    qvel = torch.from_numpy(d_npz["qvel"].astype(np.float32)).unsqueeze(0).to(ps.device)

    # 1. Forward kinematics
    ps.km.launch(
        "kernel_forward_kinematics",
        ps.bodies_buf,
        ps.geoms_buf,
        qpos,
        ps.body_xpos,
        ps.body_xmat,
        ps.geom_xpos,
        ps.geom_xmat,
        threads=1,
    )

    # 2. Contact manifold v2
    ps.compute_cad_contact_manifold_v2(ps.geom_xpos, ps.geom_xmat)

    # 3. Assemble constraints
    f_tensor = None
    if int(d_npz["ncon"]) > 0:
        f_tensor = torch.from_numpy(d_npz["contact_friction"][:, :2].astype(np.float32)).unsqueeze(0)

    J, aref, R, efc_type, nefc, overflow = ps.assemble_contact_constraints(
        ps.contact_pos,
        ps.contact_dist,
        ps.contact_body,
        ps.ncon,
        ps.body_xpos,
        ps.body_xmat,
        qvel,
        friction=f_tensor,
    )

    nefc_val = int(nefc.cpu().item())
    nefc_ref = int(d_npz["nefc"])
    assert nefc_val == nefc_ref, f"{sc_name}: Metal nefc {nefc_val} != MuJoCo {nefc_ref}"
    assert int(overflow.cpu().item()) == 0, f"{sc_name}: unexpected assembly overflow"

    if nefc_val > 0:
        np.testing.assert_allclose(
            J[0, :nefc_val].cpu().numpy(),
            d_npz["efc_J"],
            atol=1e-5,
            err_msg=f"{sc_name}: J mismatch",
        )
        np.testing.assert_allclose(
            aref[0, :nefc_val].cpu().numpy(),
            d_npz["efc_aref"],
            atol=1e-4,
            err_msg=f"{sc_name}: aref mismatch",
        )
        np.testing.assert_allclose(
            R[0, :nefc_val].cpu().numpy(),
            d_npz["efc_R"],
            atol=5e-5,
            err_msg=f"{sc_name}: R mismatch",
        )


@pytest.mark.parametrize("sc_name", get_all_scenarios())
def test_metal_autonomous_forward_dynamics_bounds(physics_slice, sc_name):
    """Verifies that autonomous forward dynamics matches MuJoCo forces and accelerations within strict bounds."""
    import torch
    ps = physics_slice
    d_npz = np.load(CORPUS_DIR / f"{sc_name}.npz")
    qpos = torch.from_numpy(d_npz["qpos"].astype(np.float32)).unsqueeze(0).to(ps.device)
    qvel = torch.from_numpy(d_npz["qvel"].astype(np.float32)).unsqueeze(0).to(ps.device)

    f_smooth = None
    if "qfrc_smooth" in d_npz:
        f_smooth = torch.from_numpy(d_npz["qfrc_smooth"].astype(np.float32)).unsqueeze(0).to(ps.device)

    f_tensor = None
    if int(d_npz["ncon"]) > 0:
        f_tensor = torch.from_numpy(d_npz["contact_friction"][:, :2].astype(np.float32)).unsqueeze(0).to(ps.device)

    pwm = torch.from_numpy(d_npz["per_world_mass"].astype(np.float32)).unsqueeze(0).to(ps.device) if "per_world_mass" in d_npz and np.any(d_npz["per_world_mass"] != 0) else None
    pwi = torch.from_numpy(d_npz["per_world_ipos"].astype(np.float32)).unsqueeze(0).to(ps.device) if "per_world_ipos" in d_npz and np.any(d_npz["per_world_ipos"] != 0) else None
    pwa = torch.from_numpy(d_npz["per_world_armature"].astype(np.float32)).unsqueeze(0).to(ps.device) if "per_world_armature" in d_npz and np.any(d_npz["per_world_armature"] != 0) else None

    out = ps.forward_autonomous(
        qpos,
        qvel,
        f_smooth=f_smooth,
        friction=f_tensor,
        per_world_mass=pwm,
        per_world_ipos=pwi,
        per_world_armature=pwa,
        max_iters=200,
        tol=1e-5,
    )

    # 1. Contact count and overflow verification
    assert int(out.ncon.cpu().item()) == int(d_npz["ncon"]), f"{sc_name}: contact count mismatch"
    assert int(out.contact_overflow.cpu().item()) == 0, f"{sc_name}: contact overflow"
    assert int(out.assembly_overflow.cpu().item()) == 0, f"{sc_name}: assembly overflow"
    assert int(out.cholesky_status.cpu().item()) == 0, f"{sc_name}: Cholesky factorization failed"

    # 2. Acceleration bounds
    qacc_metal = out.qacc[0].cpu().numpy()
    qacc_ref = d_npz["qacc"]
    assert np.all(np.isfinite(qacc_metal)), f"{sc_name}: non-finite acceleration"

    lin_acc_err = float(np.max(np.abs(qacc_metal[:3] - qacc_ref[:3])))
    ang_acc_err = float(np.max(np.abs(qacc_metal[3:6] - qacc_ref[3:6])))
    jnt_acc_err = float(np.max(np.abs(qacc_metal[6:] - qacc_ref[6:])))

    assert lin_acc_err <= 0.05, f"{sc_name}: linear acc err {lin_acc_err:.4e} > 0.05 m/s^2"
    ang_thresh = 0.15 if "high_condition" in sc_name else 0.05
    assert ang_acc_err <= ang_thresh, f"{sc_name}: angular acc err {ang_acc_err:.4e} > {ang_thresh} rad/s^2"
    assert jnt_acc_err <= 0.05, f"{sc_name}: joint acc err {jnt_acc_err:.4e} > 0.05 rad/s^2"

    # 3. Constraint force bounds
    qfrc_metal = out.qfrc_constraint[0].cpu().numpy()
    qfrc_ref = d_npz["qfrc_constraint"]
    assert np.all(np.isfinite(qfrc_metal)), f"{sc_name}: non-finite constraint force"

    lin_frc_err = float(np.max(np.abs(qfrc_metal[:3] - qfrc_ref[:3])))
    ang_frc_err = float(np.max(np.abs(qfrc_metal[3:6] - qfrc_ref[3:6])))
    jnt_frc_err = float(np.max(np.abs(qfrc_metal[6:] - qfrc_ref[6:])))

    assert lin_frc_err <= 0.05, f"{sc_name}: linear force err {lin_frc_err:.4e} > 0.05 N"
    assert ang_frc_err <= 0.05, f"{sc_name}: torque err {ang_frc_err:.4e} > 0.05 Nm"
    assert jnt_frc_err <= 0.05, f"{sc_name}: joint force err {jnt_frc_err:.4e} > 0.05 Nm"


def test_autonomous_overflow_safety(physics_slice):
    """Verifies that capacity overflow is safely detected and does not produce unbounded memory writes."""
    import torch
    ps = physics_slice
    d_npz = np.load(CORPUS_DIR / "contact_onset.npz")
    qpos = torch.from_numpy(d_npz["qpos"].astype(np.float32)).unsqueeze(0).to(ps.device)
    qvel = torch.from_numpy(d_npz["qvel"].astype(np.float32)).unsqueeze(0).to(ps.device)

    # 1. Forward kinematics
    ps.km.launch(
        "kernel_forward_kinematics",
        ps.bodies_buf,
        ps.geoms_buf,
        qpos,
        ps.body_xpos,
        ps.body_xmat,
        ps.geom_xpos,
        ps.geom_xmat,
        threads=1,
    )

    # 2. Test contact overflow by requesting small capacity (e.g. 2 contacts when 6 exist)
    c_pos, c_dist, c_norm, c_body, c_geom, ncon, c_overflow = ps.compute_cad_contact_manifold_v2(
        ps.geom_xpos,
        ps.geom_xmat,
        nconmax=2,
    )
    assert int(c_overflow.cpu().item()) == 1, "Expected contact overflow flag to be 1"
    assert int(ncon.cpu().item()) == 2, "Expected ncon capped at capacity 2"

    # 3. Test assembly overflow with capacity smaller than nefc
    J, aref, R, efc_type, nefc, a_overflow = ps.assemble_contact_constraints(
        c_pos,
        c_dist,
        c_body,
        ncon,
        ps.body_xpos,
        ps.body_xmat,
        qvel,
        capacity=4, # 2 contacts = 8 rows, capacity 4 overflows
    )
    assert int(a_overflow.cpu().item()) == 1, "Expected assembly overflow flag to be 1"
    assert int(nefc.cpu().item()) == 4, "Expected nefc capped at capacity 4"


def test_default_friction_canonical_parity(physics_slice):
    """Verifies that forward_autonomous with friction=None uses canonical mu=1.0 and achieves parity."""
    import torch
    ps = physics_slice
    d_npz = np.load(CORPUS_DIR / "nominal_standing_realistic.npz")
    qpos = torch.from_numpy(d_npz["qpos"].astype(np.float32)).unsqueeze(0).to(ps.device)
    qvel = torch.from_numpy(d_npz["qvel"].astype(np.float32)).unsqueeze(0).to(ps.device)

    out = ps.forward_autonomous(
        qpos,
        qvel,
        friction=None,  # Tests default friction selection
        max_iters=200,
        tol=1e-5,
    )

    assert int(out.ncon.cpu().item()) == int(d_npz["ncon"])
    assert int(out.contact_overflow.cpu().item()) == 0
    assert int(out.assembly_overflow.cpu().item()) == 0
    assert int(out.solver_status.cpu().item()) == 0

    # Verify R matches canonical reference R (~1.25116)
    nefc = int(out.nefc.cpu().item())
    r_metal = out.R[0, :nefc].cpu().numpy()
    np.testing.assert_allclose(r_metal, d_npz["efc_R"][:nefc], atol=1e-4)

    # Verify acceleration and force bounds
    qacc_metal = out.qacc[0].cpu().numpy()
    np.testing.assert_allclose(qacc_metal[:3], d_npz["qacc"][:3], atol=0.05)
    np.testing.assert_allclose(qacc_metal[3:6], d_npz["qacc"][3:6], atol=0.05)
    np.testing.assert_allclose(qacc_metal[6:], d_npz["qacc"][6:], atol=0.05)


def test_independently_specified_friction(physics_slice):
    """Verifies custom randomized friction tensor passed independently of oracle contact metadata."""
    import torch
    ps = physics_slice
    d_npz = np.load(CORPUS_DIR / "nominal_standing_realistic.npz")
    qpos = torch.from_numpy(d_npz["qpos"].astype(np.float32)).unsqueeze(0).to(ps.device)
    qvel = torch.from_numpy(d_npz["qvel"].astype(np.float32)).unsqueeze(0).to(ps.device)

    # Explicit custom friction mu = [0.7, 0.7]
    custom_mu = torch.tensor([[0.7, 0.7]], dtype=torch.float32, device=ps.device)

    out = ps.forward_autonomous(
        qpos,
        qvel,
        friction=custom_mu,
        max_iters=200,
        tol=1e-5,
    )
    assert int(out.solver_status.cpu().item()) == 0
    assert int(out.assembly_overflow.cpu().item()) == 0

    # Under mu=0.7, dA0 = tran * (1 + 0.7^2), Rpy = 2 * mu_reg^2 * R0
    # Rpy must be strictly less than canonical R (which used mu=1.0)
    nefc = int(out.nefc.cpu().item())
    r_metal = out.R[0, :nefc].cpu().numpy()
    assert np.all(r_metal < d_npz["efc_R"][:nefc])
    assert np.all(np.isfinite(out.qacc[0].cpu().numpy()))


def test_heterogeneous_friction_batch(physics_slice):
    """Verifies B=2 batch with world 0 having mu=1.0 and world 1 having mu=0.6."""
    import torch
    ps = physics_slice
    d_npz = np.load(CORPUS_DIR / "nominal_standing_realistic.npz")
    qpos = torch.from_numpy(d_npz["qpos"].astype(np.float32)).repeat(2, 1).to(ps.device)
    qvel = torch.from_numpy(d_npz["qvel"].astype(np.float32)).repeat(2, 1).to(ps.device)

    # World 0: mu=1.0, World 1: mu=0.6
    f_batch = torch.tensor([[1.0, 1.0], [0.6, 0.6]], dtype=torch.float32, device=ps.device)

    out = ps.forward_autonomous(
        qpos,
        qvel,
        friction=f_batch,
        max_iters=200,
        tol=1e-5,
    )

    nefc_0 = int(out.nefc[0].cpu().item())
    nefc_1 = int(out.nefc[1].cpu().item())
    assert nefc_0 == nefc_1 == int(d_npz["nefc"])

    # World 0 should match canonical reference R
    np.testing.assert_allclose(out.R[0, :nefc_0].cpu().numpy(), d_npz["efc_R"][:nefc_0], atol=1e-4)

    # World 1 should have lower R and different J friction rows
    r_w1 = out.R[1, :nefc_1].cpu().numpy()
    assert np.all(r_w1 < d_npz["efc_R"][:nefc_1])
    assert not np.allclose(out.J[0, :nefc_0].cpu().numpy(), out.J[1, :nefc_1].cpu().numpy())


def test_capacity_stride_and_memory_safety_b2(physics_slice):
    """Verifies that reduced capacities under B=2 do not corrupt adjacent worlds or buffer strides."""
    import torch
    ps = physics_slice
    d_stand = np.load(CORPUS_DIR / "standing_zero_vel.npz")
    d_air = np.load(CORPUS_DIR / "airborne.npz")

    # Stack standing (world 0) and airborne (world 1)
    qpos = torch.stack([
        torch.from_numpy(d_stand["qpos"].astype(np.float32)),
        torch.from_numpy(d_air["qpos"].astype(np.float32)),
    ]).to(ps.device)
    qvel = torch.stack([
        torch.from_numpy(d_stand["qvel"].astype(np.float32)),
        torch.from_numpy(d_air["qvel"].astype(np.float32)),
    ]).to(ps.device)

    # 1. Forward kinematics
    ps._ensure_batch_size(2)
    ps.km.launch(
        "kernel_forward_kinematics",
        ps.bodies_buf,
        ps.geoms_buf,
        qpos,
        ps.body_xpos,
        ps.body_xmat,
        ps.geom_xpos,
        ps.geom_xmat,
        threads=2,
    )

    # 2. Reduced contact capacity = 4 (standing has 6 contacts, airborne has 0)
    c_pos, c_dist, c_norm, c_body, c_geom, ncon, c_overflow = ps.compute_cad_contact_manifold_v2(
        ps.geom_xpos,
        ps.geom_xmat,
        nconmax=4,
    )
    assert c_pos.shape == (2, 4, 3)
    assert int(ncon[0].cpu().item()) == 4
    assert int(c_overflow[0].cpu().item()) == 1  # Overflowed world 0
    assert int(ncon[1].cpu().item()) == 0
    assert int(c_overflow[1].cpu().item()) == 0  # World 1 isolated and clean

    # 3. Assemble with capacity = 16 (4 contacts * 4 rows)
    J, aref, R, efc_type, nefc, a_overflow = ps.assemble_contact_constraints(
        c_pos,
        c_dist,
        c_body,
        ncon,
        ps.body_xpos,
        ps.body_xmat,
        qvel,
        capacity=16,
    )
    assert J.shape == (2, 16, 20)
    assert int(nefc[0].cpu().item()) == 16
    assert int(a_overflow[0].cpu().item()) == 0
    assert int(nefc[1].cpu().item()) == 0
    assert int(a_overflow[1].cpu().item()) == 0

    # 4. Capacity validation guards (must reject invalid before dispatch)
    with pytest.raises(ValueError, match="positive multiple of 4"):
        ps.assemble_contact_constraints(
            c_pos, c_dist, c_body, ncon, ps.body_xpos, ps.body_xmat, qvel, capacity=15
        )
    with pytest.raises(ValueError, match="in \\[1, 32\\]"):
        ps.assemble_contact_constraints(
            c_pos, c_dist, c_body, ncon, ps.body_xpos, ps.body_xmat, qvel, capacity=36
        )
    with pytest.raises(ValueError, match="in \\[1, 32\\]"):
        ps.assemble_contact_constraints(
            c_pos, c_dist, c_body, ncon, ps.body_xpos, ps.body_xmat, qvel, capacity=-4
        )

    # 5. Decoupled oracle capacity reuse test
    # Call oracle solve with capacity 16, then verify full autonomous pipeline with default capacity 32 is unharmed
    L = torch.eye(20, dtype=torch.float32, device=ps.device).unsqueeze(0).repeat(2, 1, 1)
    f_sm = torch.zeros((2, 20), dtype=torch.float32, device=ps.device)
    res_oracle = ps.solve_oracle_constraints(
        L, f_sm, J, aref, R, nefc, efc_type, capacity=16
    )
    assert res_oracle["lambda"].shape == (2, 16)

    # Next call forward_autonomous (must use autonomous_capacity=32 cleanly without error)
    out_auton = ps.forward_autonomous(qpos, qvel, max_iters=50)
    assert out_auton.J.shape == (2, 32, 20)
    assert out_auton.lambda_force.shape == (2, 32)


def test_full_path_overflow_invalidation(physics_slice):
    """Verifies that an overflow in contact manifold or assembly invalidates the final solve outputs."""
    import torch
    ps = physics_slice
    d_npz = np.load(CORPUS_DIR / "nominal_standing_realistic.npz")
    qpos = torch.from_numpy(d_npz["qpos"].astype(np.float32)).unsqueeze(0).to(ps.device)
    qvel = torch.from_numpy(d_npz["qvel"].astype(np.float32)).unsqueeze(0).to(ps.device)

    # 1. Contact manifold overflow (standing has 6 contacts; limit to 2)
    out_contact_ovf = ps.forward_autonomous(qpos, qvel, nconmax=2)
    assert int(out_contact_ovf.solver_status.cpu().item()) == -6
    assert np.isnan(out_contact_ovf.qacc[0].cpu().numpy()).all()
    assert np.isnan(out_contact_ovf.qfrc_constraint[0].cpu().numpy()).all()
    assert np.isnan(out_contact_ovf.lambda_force[0].cpu().numpy()).all()

    # 2. Assembly row overflow (standing has 24 constraint rows; limit capacity to 4)
    out_assembly_ovf = ps.forward_autonomous(qpos, qvel, capacity=4)
    assert int(out_assembly_ovf.solver_status.cpu().item()) == -7
    assert np.isnan(out_assembly_ovf.qacc[0].cpu().numpy()).all()
    assert np.isnan(out_assembly_ovf.qfrc_constraint[0].cpu().numpy()).all()
    assert np.isnan(out_assembly_ovf.lambda_force[0].cpu().numpy()).all()


def test_mixed_world_overflow_isolation_b2(physics_slice):
    """Verifies B=2 where world 0 is valid and world 1 overflows, ensuring complete isolation and non-aliased recovery."""
    import torch
    ps = physics_slice
    d0 = np.load(CORPUS_DIR / "heel_only_contact_realistic.npz")
    d1 = np.load(CORPUS_DIR / "standing_zero_vel.npz")
    qpos0 = torch.from_numpy(d0["qpos"].astype(np.float32)).unsqueeze(0).to(ps.device)
    qvel0 = torch.from_numpy(d0["qvel"].astype(np.float32)).unsqueeze(0).to(ps.device)
    qpos1 = torch.from_numpy(d1["qpos"].astype(np.float32)).unsqueeze(0).to(ps.device)
    qvel1 = torch.from_numpy(d1["qvel"].astype(np.float32)).unsqueeze(0).to(ps.device)

    # 1. Standalone valid baseline for world 0: clone outputs immediately to prevent aliasing
    out_valid0 = ps.forward_autonomous(qpos0, qvel0, capacity=16, max_iters=200)
    assert int(out_valid0.solver_status[0].cpu().item()) == 0
    nefc0 = int(out_valid0.nefc[0].cpu().item())
    assert nefc0 == 8
    expected_qacc0 = out_valid0.qacc[0].clone()
    expected_lam0 = out_valid0.lambda_force[0, :nefc0].clone()
    expected_qfrc0 = out_valid0.qfrc_constraint[0].clone()

    # 2. Mixed-world batched run: world 0 (8 rows, fits capacity 16) vs world 1 (24 rows, overflows capacity 16)
    qpos_b2 = torch.cat([qpos0, qpos1], dim=0)
    qvel_b2 = torch.cat([qvel0, qvel1], dim=0)
    out_mixed = ps.forward_autonomous(qpos_b2, qvel_b2, capacity=16, max_iters=200)

    # World 0 must succeed and match cloned snapshot
    assert int(out_mixed.solver_status[0].cpu().item()) == 0
    assert int(out_mixed.assembly_overflow[0].cpu().item()) == 0
    np.testing.assert_allclose(out_mixed.qacc[0].cpu().numpy(), expected_qacc0.cpu().numpy(), atol=1e-5)
    np.testing.assert_allclose(out_mixed.lambda_force[0, :nefc0].cpu().numpy(), expected_lam0.cpu().numpy(), atol=1e-5)
    np.testing.assert_allclose(out_mixed.qfrc_constraint[0].cpu().numpy(), expected_qfrc0.cpu().numpy(), atol=1e-5)

    # World 1 must fail with assembly overflow (-7) and NaN outputs
    assert int(out_mixed.assembly_overflow[1].cpu().item()) == 1
    assert int(out_mixed.solver_status[1].cpu().item()) == -7
    assert np.isnan(out_mixed.qacc[1].cpu().numpy()).all()
    assert np.isnan(out_mixed.lambda_force[1].cpu().numpy()).all()
    assert np.isnan(out_mixed.qfrc_constraint[1].cpu().numpy()).all()

    # 3. Buffer reuse & recovery: re-run world 0 duplicated across both batch slots
    out_rec = ps.forward_autonomous(qpos0.repeat(2, 1), qvel0.repeat(2, 1), capacity=16, max_iters=200)
    assert int(out_rec.solver_status[0].cpu().item()) == 0
    assert int(out_rec.solver_status[1].cpu().item()) == 0
    np.testing.assert_allclose(out_rec.qacc[0].cpu().numpy(), expected_qacc0.cpu().numpy(), atol=1e-5)
    np.testing.assert_allclose(out_rec.qacc[1].cpu().numpy(), expected_qacc0.cpu().numpy(), atol=1e-5)


def test_compact_contacts_heterogeneous_per_contact_friction_b2(physics_slice):
    """Verifies that assembly with compact contacts (stride_ncon < nconmax) indexes per-world friction with exact stride."""
    import torch
    ps = physics_slice
    B = 2
    stride_ncon = 4  # Compact contact allocation (smaller than engine nconmax=35)
    contact_pos = torch.zeros((B, stride_ncon, 3), dtype=torch.float32, device=ps.device)
    contact_dist = torch.zeros((B, stride_ncon), dtype=torch.float32, device=ps.device)
    contact_body = torch.zeros((B, stride_ncon), dtype=torch.int32, device=ps.device)
    contact_body[0, :] = 7
    contact_body[1, :] = 16
    ncon = torch.tensor([2, 2], dtype=torch.int32, device=ps.device)
    body_xpos = torch.zeros((B, 17, 3), dtype=torch.float32, device=ps.device)
    body_xmat = torch.zeros((B, 17, 9), dtype=torch.float32, device=ps.device)
    for i in (0, 4, 8):
        body_xmat[:, :, i] = 1.0
    qvel = torch.zeros((B, 20), dtype=torch.float32, device=ps.device)

    # Heterogeneous per-world friction: world 0 has mu=0.6, world 1 has mu=0.9
    friction = torch.zeros((2, 2, 2), dtype=torch.float32, device=ps.device)
    friction[0, :, :] = 0.6
    friction[1, :, :] = 0.9

    J, aref, R, efc_type, nefc, overflow = ps.assemble_contact_constraints(
        contact_pos, contact_dist, contact_body, ncon, body_xpos, body_xmat, qvel, friction=friction, capacity=32
    )

    J_np = J.cpu().numpy()
    R_np = R.cpu().numpy()

    # Verify linear DOF facet coefficients match specified friction
    assert np.isclose(J_np[0, 0, 1], 0.6, atol=1e-5)
    assert np.isclose(J_np[1, 0, 1], 0.9, atol=1e-5)

    # Verify regularization values match analytical formula:
    # R_py = 2.0 * mu^2 * (1 - imp) / imp * tran * (1 + mu^2)
    # tran_7 = 5.9430275, tran_16 = 5.9417195
    assert np.isclose(R_np[0, 0], 0.6466014, atol=1e-5)
    assert np.isclose(R_np[1, 0], 1.9358122, atol=1e-5)

    # Capacity reuse: test with capacity=16
    J16, aref16, R16, efc_type16, nefc16, overflow16 = ps.assemble_contact_constraints(
        contact_pos, contact_dist, contact_body, ncon, body_xpos, body_xmat, qvel, friction=friction, capacity=16
    )
    assert int(overflow16[0].cpu().item()) == 0
    assert int(overflow16[1].cpu().item()) == 0
    np.testing.assert_allclose(R16[:, :8].cpu().numpy(), R[:, :8].cpu().numpy(), atol=1e-5)


def test_invalid_contact_counts(physics_slice):
    """Verifies that negative or oversized active contact counts trigger device-level status -4 and invalidate outputs."""
    import torch
    ps = physics_slice
    B = 2
    stride_ncon = 4
    contact_pos = torch.zeros((B, stride_ncon, 3), dtype=torch.float32, device=ps.device)
    contact_dist = torch.zeros((B, stride_ncon), dtype=torch.float32, device=ps.device)
    contact_body = torch.full((B, stride_ncon), 7, dtype=torch.int32, device=ps.device)
    body_xpos = torch.zeros((B, 17, 3), dtype=torch.float32, device=ps.device)
    body_xmat = torch.zeros((B, 17, 9), dtype=torch.float32, device=ps.device)
    for i in (0, 4, 8):
        body_xmat[:, :, i] = 1.0
    qvel = torch.zeros((B, 20), dtype=torch.float32, device=ps.device)

    # 1. Mixed batch: world 0 valid (ncon=2), world 1 oversized (ncon=5 > stride_ncon=4)
    ncon_oversized = torch.tensor([2, 5], dtype=torch.int32, device=ps.device)
    J, aref, R, efc_type, nefc, overflow = ps.assemble_contact_constraints(
        contact_pos, contact_dist, contact_body, ncon_oversized, body_xpos, body_xmat, qvel, capacity=32
    )
    assert int(overflow[0].cpu().item()) == 0
    assert int(nefc[0].cpu().item()) == 8
    assert np.all(np.isfinite(J[0, :8].cpu().numpy()))
    assert int(overflow[1].cpu().item()) == -4
    assert int(nefc[1].cpu().item()) == 0
    assert np.isnan(J[1].cpu().numpy()).all()

    # 2. Mixed batch: world 0 valid (ncon=2), world 1 negative (ncon=-1)
    ncon_neg = torch.tensor([2, -1], dtype=torch.int32, device=ps.device)
    J, aref, R, efc_type, nefc, overflow = ps.assemble_contact_constraints(
        contact_pos, contact_dist, contact_body, ncon_neg, body_xpos, body_xmat, qvel, capacity=32
    )
    assert int(overflow[0].cpu().item()) == 0
    assert int(nefc[0].cpu().item()) == 8
    assert int(overflow[1].cpu().item()) == -4
    assert int(nefc[1].cpu().item()) == 0
    assert np.isnan(J[1].cpu().numpy()).all()


def test_device_friction_validation(physics_slice):
    """Verifies that non-positive or non-finite friction tensors are caught by the GPU kernel without host sync."""
    import torch
    ps = physics_slice
    B = 2
    stride_ncon = 4
    contact_pos = torch.zeros((B, stride_ncon, 3), dtype=torch.float32, device=ps.device)
    contact_dist = torch.zeros((B, stride_ncon), dtype=torch.float32, device=ps.device)
    contact_body = torch.full((B, stride_ncon), 7, dtype=torch.int32, device=ps.device)
    body_xpos = torch.zeros((B, 17, 3), dtype=torch.float32, device=ps.device)
    body_xmat = torch.zeros((B, 17, 9), dtype=torch.float32, device=ps.device)
    for i in (0, 4, 8):
        body_xmat[:, :, i] = 1.0
    qvel = torch.zeros((B, 20), dtype=torch.float32, device=ps.device)
    ncon = torch.tensor([2, 2], dtype=torch.int32, device=ps.device)

    # 1. Negative friction in world 1
    friction_neg = torch.full((2, 4, 2), 0.8, dtype=torch.float32, device=ps.device)
    friction_neg[1, 0, 0] = -0.5
    J, aref, R, efc_type, nefc, overflow = ps.assemble_contact_constraints(
        contact_pos, contact_dist, contact_body, ncon, body_xpos, body_xmat, qvel, friction=friction_neg, capacity=32
    )
    assert int(overflow[0].cpu().item()) == 0
    assert int(nefc[0].cpu().item()) == 8
    assert int(overflow[1].cpu().item()) == -3
    assert int(nefc[1].cpu().item()) == 0
    assert np.isnan(J[1].cpu().numpy()).all()

    # 2. NaN friction in world 1
    friction_nan = torch.full((2, 4, 2), 0.8, dtype=torch.float32, device=ps.device)
    friction_nan[1, 0, 0] = float("nan")
    J, aref, R, efc_type, nefc, overflow = ps.assemble_contact_constraints(
        contact_pos, contact_dist, contact_body, ncon, body_xpos, body_xmat, qvel, friction=friction_nan, capacity=32
    )
    assert int(overflow[0].cpu().item()) == 0
    assert int(nefc[0].cpu().item()) == 8
    assert int(overflow[1].cpu().item()) == -3
    assert int(nefc[1].cpu().item()) == 0
    assert np.isnan(J[1].cpu().numpy()).all()


def test_non_finite_input_guards(physics_slice):
    """Verifies that non-finite (NaN/Inf) inputs are caught by device kernels and do not produce silent zero-contact output."""
    import torch
    ps = physics_slice
    d_npz = np.load(CORPUS_DIR / "nominal_standing_realistic.npz")
    qpos = torch.from_numpy(d_npz["qpos"].astype(np.float32)).unsqueeze(0).to(ps.device)
    qvel = torch.from_numpy(d_npz["qvel"].astype(np.float32)).unsqueeze(0).to(ps.device)

    # 1. Forward kinematics
    ps._ensure_batch_size(1)
    ps.km.launch(
        "kernel_forward_kinematics",
        ps.bodies_buf,
        ps.geoms_buf,
        qpos,
        ps.body_xpos,
        ps.body_xmat,
        ps.geom_xpos,
        ps.geom_xmat,
        threads=1,
    )

    # Inject NaN into geom_xpos
    nan_geom_pos = ps.geom_xpos.clone()
    nan_geom_pos[0, 0, 0] = float("nan")

    c_pos, c_dist, c_norm, c_body, c_geom, ncon, c_overflow = ps.compute_cad_contact_manifold_v2(
        nan_geom_pos, ps.geom_xmat
    )
    assert int(c_overflow.cpu().item()) == -1, "Expected non-finite flag -1 for NaN input"
    assert int(ncon.cpu().item()) == 0
    assert np.isnan(c_pos[0].cpu().numpy()).all()

    # Inject NaN into qvel for assembly
    nan_qvel = qvel.clone()
    nan_qvel[0, 0] = float("nan")
    J, aref, R, efc_type, nefc, a_overflow = ps.assemble_contact_constraints(
        ps.contact_pos, ps.contact_dist, ps.contact_body, ps.ncon, ps.body_xpos, ps.body_xmat, nan_qvel
    )
    assert int(a_overflow.cpu().item()) == -1, "Expected non-finite flag -1 in assembly"
    assert np.isnan(aref[0].cpu().numpy()).all()


@pytest.mark.parametrize("sc_name", get_all_scenarios())
def test_independent_autonomous_kkt_residuals(physics_slice, sc_name):
    """Recomputes primal, dual, complementarity, and projected gradient residuals independently on CPU for the assembled QP."""
    import torch
    ps = physics_slice
    d_npz = np.load(CORPUS_DIR / f"{sc_name}.npz")
    qpos = torch.from_numpy(d_npz["qpos"].astype(np.float32)).unsqueeze(0).to(ps.device)
    qvel = torch.from_numpy(d_npz["qvel"].astype(np.float32)).unsqueeze(0).to(ps.device)

    f_smooth = None
    if "qfrc_smooth" in d_npz:
        f_smooth = torch.from_numpy(d_npz["qfrc_smooth"].astype(np.float32)).unsqueeze(0).to(ps.device)

    pwm = torch.from_numpy(d_npz["per_world_mass"].astype(np.float32)).unsqueeze(0).to(ps.device) if "per_world_mass" in d_npz and np.any(d_npz["per_world_mass"] != 0) else None
    pwi = torch.from_numpy(d_npz["per_world_ipos"].astype(np.float32)).unsqueeze(0).to(ps.device) if "per_world_ipos" in d_npz and np.any(d_npz["per_world_ipos"] != 0) else None
    pwa = torch.from_numpy(d_npz["per_world_armature"].astype(np.float32)).unsqueeze(0).to(ps.device) if "per_world_armature" in d_npz and np.any(d_npz["per_world_armature"] != 0) else None

    f_tensor = None
    if "randomized_friction" in sc_name and int(d_npz["ncon"]) > 0:
        f_tensor = torch.from_numpy(d_npz["contact_friction"][:, :2].astype(np.float32)).unsqueeze(0).to(ps.device)

    # Run with qualification budget max_iters=200
    out = ps.forward_autonomous(
        qpos,
        qvel,
        f_smooth=f_smooth,
        friction=f_tensor,
        per_world_mass=pwm,
        per_world_ipos=pwi,
        per_world_armature=pwa,
        max_iters=200,
        tol=1e-5,
    )

    nefc = int(out.nefc.cpu().item())
    status = int(out.solver_status.cpu().item())
    assert status in (0, 1), f"{sc_name}: solver failed with status {status}"

    if nefc > 0:
        J_np = out.J[0, :nefc].cpu().numpy()
        R_np = out.R[0, :nefc].cpu().numpy()
        aref_np = out.aref[0, :nefc].cpu().numpy()
        M_np = out.M_eff[0].cpu().numpy()
        f_sm_np = out.f_smooth[0].cpu().numpy()
        lam_np = out.lambda_force[0, :nefc].cpu().numpy()

        # Reconstruct candidate QP matching Delassus algorithm: L Y = J^T => Y = L^{-1} J^T, A = Y^T Y + diag(R)
        from scipy.linalg import solve_triangular
        L_np = out.L_factor[0].cpu().numpy()
        Y = solve_triangular(L_np, J_np.T, lower=True)
        A = Y.T @ Y + np.diag(R_np)
        y0 = solve_triangular(L_np, f_sm_np, lower=True)
        a0 = solve_triangular(L_np.T, y0, lower=False)
        b = J_np @ a0 - aref_np

        # Dual gradient g = A lambda + b
        g = A @ lam_np + b

        primal_res = float(np.max(np.maximum(0.0, -lam_np)))
        dual_infeas = float(np.max(np.maximum(0.0, -g)))
        comp_res = float(np.max(np.abs(lam_np * g)))
        proj_grad_step = float(np.max(np.abs(lam_np - np.maximum(0.0, lam_np - g / np.diag(A)))))

        assert primal_res <= 1e-6, f"{sc_name}: primal infeasibility {primal_res:.4e} > 1e-6"
        if status == 0:
            assert dual_infeas <= 2e-4, f"{sc_name}: status 0 dual infeasibility {dual_infeas:.4e} > 2e-4"
            assert proj_grad_step <= 2e-5, f"{sc_name}: status 0 projected gradient step {proj_grad_step:.4e} > 2e-5"
            assert comp_res <= 5e-3, f"{sc_name}: status 0 complementarity {comp_res:.4e} > 5e-3"
        else:
            # Status 1: bounded non-convergence
            assert dual_infeas <= 0.05, f"{sc_name}: status 1 dual infeasibility unbounded {dual_infeas:.4e}"
            assert proj_grad_step <= 0.05, f"{sc_name}: status 1 projected gradient step unbounded {proj_grad_step:.4e}"

