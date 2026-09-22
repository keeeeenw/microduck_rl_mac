import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from src.representative_physics_slice import RepresentativePhysicsSlice
from scipy.linalg import solve_triangular


def compute_kkt(out, nefc, max_iters):
    if nefc == 0:
        return {
            "stat": 0,
            "iters": 0,
            "label": "converged",
            "primal": 0.0,
            "dual": 0.0,
            "comp": 0.0,
            "proj": 0.0,
        }
    stat = int(out.solver_status[0].cpu().item())
    iters = int(out.actual_iters[0].cpu().item())
    if stat == 0:
        label = "converged"
    elif iters == max_iters:
        label = "exhausted"
    else:
        label = "stagnated"

    J_np = out.J[0, :nefc].cpu().numpy()
    R_np = out.R[0, :nefc].cpu().numpy()
    aref_np = out.aref[0, :nefc].cpu().numpy()
    L_np = out.L_factor[0].cpu().numpy()
    f_sm_np = out.f_smooth[0].cpu().numpy()
    lam_np = out.lambda_force[0, :nefc].cpu().numpy()

    Y = solve_triangular(L_np, J_np.T, lower=True)
    A = Y.T @ Y + np.diag(R_np)
    y0 = solve_triangular(L_np, f_sm_np, lower=True)
    a0 = solve_triangular(L_np.T, y0, lower=False)
    b = J_np @ a0 - aref_np
    g = A @ lam_np + b

    primal = float(np.max(np.maximum(0.0, -lam_np)))
    dual = float(np.max(np.maximum(0.0, -g)))
    comp = float(np.max(np.abs(lam_np * g)))
    proj = float(np.max(np.abs(lam_np - np.maximum(0.0, lam_np - g / np.diag(A)))))

    return {
        "stat": stat,
        "iters": iters,
        "label": label,
        "primal": primal,
        "dual": dual,
        "comp": comp,
        "proj": proj,
    }


def main():
    ps = RepresentativePhysicsSlice(batch_size=1)
    corpus_dir = Path("corpus")
    scenarios = sorted([f.stem for f in corpus_dir.glob("*.npz")])

    print("### Autonomous Forward Dynamics & Budget Comparison (100 vs 200 iters)")
    print("| Scenario | nefc | Stat(100) | Iter(100) | Label(100) | Stat(200) | Iter(200) | Label(200) | Dual Infeas (200) | Proj Res (200) | Max Acc Err | Max Frc Err |")
    print("|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    kkt_records = []

    for sc_name in scenarios:
        d_npz = np.load(corpus_dir / f"{sc_name}.npz")
        qpos = torch.from_numpy(d_npz["qpos"].astype(np.float32)).unsqueeze(0).to(ps.device)
        qvel = torch.from_numpy(d_npz["qvel"].astype(np.float32)).unsqueeze(0).to(ps.device)
        f_smooth = torch.from_numpy(d_npz["qfrc_smooth"].astype(np.float32)).unsqueeze(0).to(ps.device) if "qfrc_smooth" in d_npz else None
        pwm = torch.from_numpy(d_npz["per_world_mass"].astype(np.float32)).unsqueeze(0).to(ps.device) if "per_world_mass" in d_npz and np.any(d_npz["per_world_mass"] != 0) else None
        pwi = torch.from_numpy(d_npz["per_world_ipos"].astype(np.float32)).unsqueeze(0).to(ps.device) if "per_world_ipos" in d_npz and np.any(d_npz["per_world_ipos"] != 0) else None
        pwa = torch.from_numpy(d_npz["per_world_armature"].astype(np.float32)).unsqueeze(0).to(ps.device) if "per_world_armature" in d_npz and np.any(d_npz["per_world_armature"] != 0) else None

        # Scenario friction input: prepared once and supplied to both budgets
        f_tensor = None
        if "randomized_friction" in sc_name and int(d_npz["ncon"]) > 0:
            f_tensor = torch.from_numpy(d_npz["contact_friction"][:, :2].astype(np.float32)).unsqueeze(0).to(ps.device)

        # Budget 100
        out100 = ps.forward_autonomous(
            qpos, qvel, f_smooth=f_smooth, friction=f_tensor,
            per_world_mass=pwm, per_world_ipos=pwi, per_world_armature=pwa,
            max_iters=100, tol=1e-5
        )
        nefc = int(out100.nefc[0].cpu().item())
        kkt100 = compute_kkt(out100, nefc, 100)

        # Budget 200
        out200 = ps.forward_autonomous(
            qpos, qvel, f_smooth=f_smooth, friction=f_tensor,
            per_world_mass=pwm, per_world_ipos=pwi, per_world_armature=pwa,
            max_iters=200, tol=1e-5
        )
        kkt200 = compute_kkt(out200, nefc, 200)

        qacc_err = float(np.max(np.abs(out200.qacc[0].cpu().numpy() - d_npz["qacc"])))
        qfrc_err = float(np.max(np.abs(out200.qfrc_constraint[0].cpu().numpy() - d_npz["qfrc_constraint"])))

        kkt_records.append((sc_name, nefc, kkt100, kkt200, qacc_err, qfrc_err))

        print(
            f"| `{sc_name}` | {nefc} | {kkt100['stat']} | {kkt100['iters']} | {kkt100['label']} | "
            f"{kkt200['stat']} | {kkt200['iters']} | {kkt200['label']} | "
            f"{kkt200['dual']:.2e} | {kkt200['proj']:.2e} | {qacc_err:.2e} | {qfrc_err:.2e} |"
        )

    print("\n### Complete Candidate-QP KKT Residual Breakdown (Budgets 100 and 200)")
    print("| Scenario | Budget | Stat | Iter | Label | Primal Infeas | Dual Infeas | Complementarity | Proj Residual |")
    print("|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    for sc_name, nefc, kkt100, kkt200, _, _ in kkt_records:
        print(f"| `{sc_name}` | 100 | {kkt100['stat']} | {kkt100['iters']} | {kkt100['label']} | {kkt100['primal']:.2e} | {kkt100['dual']:.2e} | {kkt100['comp']:.2e} | {kkt100['proj']:.2e} |")
        print(f"| `{sc_name}` | 200 | {kkt200['stat']} | {kkt200['iters']} | {kkt200['label']} | {kkt200['primal']:.2e} | {kkt200['dual']:.2e} | {kkt200['comp']:.2e} | {kkt200['proj']:.2e} |")


if __name__ == "__main__":
    main()
