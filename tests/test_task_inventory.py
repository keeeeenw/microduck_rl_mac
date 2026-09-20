"""Unit test for Deliverable 1: Task Inventory Verification."""

import json
from pathlib import Path

CONFIGS_DIR = Path(__file__).parent.parent / "configs"
REPORTS_DIR = Path(__file__).parent.parent / "reports"


def test_task_inventory_json_exists():
    json_path = CONFIGS_DIR / "canonical_flat_task.json"
    assert json_path.exists(), "canonical_flat_task.json must exist"
    with open(json_path) as f:
        data = json.load(f)
    
    # Coordinate dimensions
    dims = data["dimensions"]
    assert dims["nq"] == 21, f"Expected nq=21, got {dims['nq']}"
    assert dims["nv"] == 20, f"Expected nv=20, got {dims['nv']}"
    assert dims["nu"] == 14, f"Expected nu=14, got {dims['nu']}"
    assert dims["nbody"] == 17, f"Expected nbody=17, got {dims['nbody']}"
    assert dims["ngeom"] == 76, f"Expected ngeom=76, got {dims['ngeom']}"

    # Numerical options
    sim = data["simulation_options"]
    assert sim["timestep"] == 0.005, f"Expected dt=0.005, got {sim['timestep']}"
    assert sim["decimation"] == 4, f"Expected decimation=4, got {sim['decimation']}"
    assert sim["integrator"] == "mjINT_IMPLICITFAST", f"Expected ImplicitFast, got {sim['integrator']}"
    assert sim["solver"] == "mjSOL_NEWTON", f"Expected Newton, got {sim['solver']}"
    assert sim["iterations"] == 10, f"Expected 10 iterations, got {sim['iterations']}"
    assert sim["ls_iterations"] == 20, f"Expected 20 ls_iterations, got {sim['ls_iterations']}"
    assert sim["cone"] == "mjCONE_PYRAMIDAL", f"Expected pyramidal cone, got {sim['cone']}"
    assert sim["nconmax"] == 35, f"Expected nconmax=35, got {sim['nconmax']}"

    # BAM actuator check
    acts = data["actuators"]
    assert len(acts) > 0, "Actuators list must not be empty"
    act0 = acts[0]
    assert act0["motor_name"] == "xl330"
    assert act0["model"] == "m6"
    assert act0["delay_min_lag"] == 3
    assert act0["delay_max_lag"] == 6
    assert "bam_params" in act0
    bp = act0["bam_params"]
    assert "dtheta_stribeck" in bp
    assert "friction_base" in bp
    assert "load_friction_motor" in bp


def test_task_inventory_markdown_exists():
    report_path = REPORTS_DIR / "task_contract_inventory.md"
    assert report_path.exists(), "task_contract_inventory.md must exist"
    content = report_path.read_text()
    assert "nq" in content
    assert "IMPLICITFAST" in content
    assert "FrictionDRBamActuator" in content
    assert "Curricula" in content
