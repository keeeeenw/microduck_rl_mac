# MicroDuck Flat Task Physical Contract & Inventory

Generated from live instantiated `make_microduck_velocity_env_cfg()` on MacBook-Pro-14.local.

## 1. Provenance & Model Verification

- **Live Compiled XML SHA-256**: `c0fa38aa02ca23c46cbe8b4a224abf7aa5ef4b25da113171f51e779294e92be5`
- **Versioned Snapshot SHA-256**: `50e4fdf1e4045e4face124f694a64f1ab2ed7dea11df50058f39dde943be4eaa`
- **Exact Hash Match**: **NO (see diff)**

## 2. Mechanical Coordinates & Dimensions

| Dimension | Symbol | Count | Description |
| --- | --- | --- | --- |
| Generalized Positions | `nq` | 21 | 7 floating base ($p_{xyz} + q_{wxyz}$) + 14 joints |
| Generalized Velocities | `nv` | 20 | 6 floating base ($v_{xyz} + \omega_{xyz}$) + 14 joints |
| Actuators | `nu` | 14 | 14 active motors (10 leg + 4 neck/head) |
| Bodies | `nbody` | 17 | 17 bodies (including `world`, `trunk_base`, legs, head) |
| Geometries | `ngeom` | 76 | 76 total geoms (6 collision-active) |
| Sites | `nsite` | 8 | Feet tracking, IMU, sensors |

## 3. Simulation & Numerical Configuration

| Parameter | Configured Value | Enforcement Notes |
| --- | --- | --- |
| Physics Timestep | `0.005 s` (5 ms) | Exactly one substep per `sim.step()` |
| Control Decimation | `4` (20 ms / 50 Hz) | Loop in `manager_based_rl_env.py` calls `step()` 4 times |
| Integrator | `mjINT_IMPLICITFAST` | Velocity derivative correction required before solve |
| Friction Cone | `mjCONE_PYRAMIDAL` | 4-sided pyramidal friction cone |
| Solver Method | `mjSOL_NEWTON` | Newton solver with line search |
| Solver Iterations | `10` | 10 Newton iterations |
| Line-Search Iterations | `20` | 20 iterations, ls_tolerance = 0.01 |
| Max Contacts (`nconmax`) | `35` | Bounded capacity; explicit overflow detection required |

## 4. Actuation: BAM M6 Actuator Model

- **Actuator Class**: `FrictionDRBamActuator` (runs in PyTorch on MPS per substep).
- **Substep Integration**: `apply_action()` runs every 5 ms inside the decimation loop.
### Actuator Group `0`: `FrictionDRBamActuatorCfg`
- Target Joints Pattern: `('^(?!passive_).*',)`
- Motor Name: `xl330`, Model: `m6`
- Command Delay: `3–6` substeps (15–30 ms)
- Firmware Stiffness `kp_fw`: `200.0`
- Voltage Range `vin_range`: `(6.5, 8.2)` V
- Voltage Sag Gain: `(0.0, 0.2)`
- Stribeck Threshold `dtheta_stribeck`: `2.890372` rad/s
- Exponent `alpha`: `8.683260`
- Base Coulomb Friction `friction_base`: `0.004771` Nm
- Stribeck Friction `friction_stribeck`: `0.004676` Nm
- Motor Load Friction: `0.266786`, Ext Load Friction: `0.000009`
- Motor Quad Friction: `0.009972`, Ext Quad Friction: `0.004903`
- Reflected Armature: `0.001808` kg m^2

## 5. Domain Randomization & Curricula

| Term | Target | Initial Range | Curriculum / Max |
| --- | --- | --- | --- |
| CoM Randomization | Trunk base body | $\pm 3$ mm | Ramped to $\pm 8$ mm via curriculum |
| Head CoM Randomization | 5 Head bodies | $\pm 3$ mm | Ramped via curriculum |
| Mass & Inertia | Trunk / limbs | $\pm 5\%$ | Non-accumulating multiplicative scale |
| Joint Friction Scale | BAM budget | $[0.9, 1.1]$ | Scales BAM friction loss per env |
| Joint Armature | Reflected rotor inertia | $\pm 10\%$ | Preserved in BAM dynamics |
| Pushes | Base linear velocity | $\pm 0.3$ m/s | Applied every 3–6 s |
| IMU Orientation | Mounting error | $\le 6^\circ$ | Zero-centered random-axis rotation |
| Encoder Bias | Joint position obs | $\pm 0.86^\circ$ | Constant per env offset |

## 6. Required Simulation Data Views (`sim.data`)

The simulation adapter must expose persistent PyTorch MPS views for all fields consumed by RL managers:

```python
fields = [
    'time',
    'qpos',
    'qvel',
    'qacc',
    'qacc_warmstart',
    'ctrl',
    'act',
    'qfrc_applied',
    'xfrc_applied',
    'qfrc_actuator',
    'qfrc_bias',
    'qfrc_constraint',
    'xpos',
    'xquat',
    'xmat',
    'xipos',
    'ximat',
    'geom_xpos',
    'geom_xmat',
    'site_xpos',
    'site_xmat',
    'cvel',
    'cacc',
    'subtree_com',
    'subtree_linvel',
    'subtree_angmom',
    'sensordata',
    'efc.force',
    'nefc',
]
```
