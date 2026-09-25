# Flat walking validation on Apple Silicon

Evaluated September 24, 2026, on an **Apple M1 Max, 24-core GPU, 32 GB unified
memory**. This validates a bounded forward-walking result from the Mac training
path, not complete velocity-task performance or physical robot deployment.

## Result

**The policy meets our basic simulated-walking bar:** sustained forward locomotion,
upright balance, and return to a stationary pose after a stop command. It does
**not** meet a general command-tracking bar: small movement commands often leave
it standing, speed is below the requested value, and heading can drift substantially.

The completed training run used CPU MuJoCo physics with Torch MPS policy inference,
task managers, BAM actuation and PPO: 4,096 environments, 24 rollout steps per
update, and 6,000 total updates (589,824,000 environment transitions). It resumed
the initial five-update checkpoint with full environment and learner state.
The final export passed the trainer's actor-versus-ONNX comparison.

Evaluation used the **final exported ONNX policy**, CPU MuJoCo 3.10.0 and ONNX
Runtime through the existing playback/BAM code. Thus training uses the Apple GPU;
this separate deployment-style playback evaluation uses CPU inference and physics.

Policy SHA-256:
`6f7909b2fd710b0c1af771f38a052874c11e973570b6128d622fd0d36d2fcb2b`.

## Measurements

There were **52 independent trials, totaling 20 simulated minutes, with no falls
under the stated rule**. Four trials per scenario paired actuator delays of
3, 4, 5 and 6 physics steps with seeds 0, 1, 2 and 3. These are controlled scenarios,
not 52 independent training seeds or a statistical reliability guarantee.

| Scenario | Duration per trial | Command | Measured result |
| --- | ---: | --- | --- |
| Sustained forward walking | 60 s | vx = 0.30 m/s | Mean forward speed 0.184 m/s; individual means 0.175–0.188 m/s; 4/4 upright |
| Forward walking | 20 s | vx = 0.30 m/s | Mean forward speed 0.184 m/s; 4/4 upright |
| Walk, then stop | 20 s | vx = 0.30, then 0 after 10 s | 4/4 stopped upright; absolute mean vx and vy each below 0.0001 m/s after settling |
| Forward + left turn | 20 s | vx = 0.30 m/s, wz = +0.50 rad/s | Mean vx 0.174 m/s, wz +0.498 rad/s; 4/4 upright |
| Forward + right turn | 20 s | vx = 0.30 m/s, wz = −0.50 rad/s | Mean vx 0.180 m/s, wz −0.347 rad/s; 4/4 upright |
| Idle | 20 s | zero twist | Remained nearly stationary; 4/4 upright |
| Low-speed forward | 20 s each | vx = 0.05 or 0.15 m/s | Essentially stationary; command tracking fails |
| Backward / lateral | 20 s each | vx = −0.10 or vy = ±0.10 m/s | Essentially stationary; command tracking fails |
| Turn in place | 20 s each | wz = ±0.50 rad/s | Essentially stationary; command tracking fails |

The 60-second forward trials kept root height above **112 mm** and trunk tilt
below **4.6°**. These demonstrate sustained balance and locomotion, but not straight
line accuracy: one trial averaged approximately 0.086 rad/s yaw with zero requested
turning. Gait oscillations also produce substantial instantaneous velocity error;
the table reports means, not tight per-step tracking. Per-trial RMSE is retained
in the JSON results.

## Protocol and limits

- Same 61-input policy layout and projected-gravity observation mode as playback.
- BAM M6 XL330 actuation, nominal 7.4 V, voltage-drop gain 0.1, minimum 6.0 V.
- Physics at 200 Hz; inference at 50 Hz; actuator delays of 15–30 ms advance at
  physics rate. No action smoothing was added.
- Standard playback ground-contact scene, flat floor, nominal model parameters.
  Seed 0 starts at the nominal standing pose. Other seeds add at most ±2° initial
  roll/pitch and ±0.01 rad joint perturbations. Seed and delay are paired, not
  independently swept.
- No pushes, observation-noise sweep, rough terrain, battery sweep, or full
  training-domain randomization. No automatic resets or hidden recovery episodes.
- A fall ends the trial: root height below 65 mm, trunk tilt above 60°, or a
  nonfinite action/state. Measurements are taken every 20 ms.
- Velocity means/RMSE exclude the first 2 seconds. Stop measurements also exclude
  the first 2 seconds after the stop command. vx/vy are body-frame linear velocity;
  wz is local root angular velocity. Position displacement alone can be misleading
  when the robot walks in an arc.
- One completed training seed was evaluated. Linux inference and real hardware
  remain unvalidated. ONNX portability is the intended deployment path, not a
  claim of tested Linux or sim2real equivalence.

## Reproduce

From the repository root, point `POLICY` to the completed run's `policy.onnx`:

```bash
POLICY=logs/native-gpu/hybrid-walk-4096-20260920/policy.onnx
.venv/bin/python scripts/evaluate_mac_walking.py --policy "$POLICY" \
  --output artifacts/evaluation/commands.json \
  --cases idle forward_slow forward forward_fast backward left right turn_left turn_right
.venv/bin/python scripts/evaluate_mac_walking.py --policy "$POLICY" \
  --output artifacts/evaluation/maneuvers.json \
  --cases walk_stop forward_turn_left forward_turn_right
.venv/bin/python scripts/evaluate_mac_walking.py --policy "$POLICY" \
  --output artifacts/evaluation/long.json --cases forward_fast --seconds 60 \
  --gif artifacts/evaluation/walking.gif
```

The export is not bundled in this repository; supply your own or the retained
completed-run artifact. GIF rendering requires Pillow and native macOS graphics
access; omit `--gif` for headless numerical evaluation.

Checked results: [command sweep](validation/flat-walking-6000.json),
[maneuvers](validation/flat-walking-6000-maneuvers.json),
[60-second walks](validation/flat-walking-6000-long.json).

The [README GIF](media/mac-flat-walking.gif) shows seconds 2–10 of the nominal
60-second forward trial, at real-time speed with a following camera, reduced to
12.5 frames/s for size. It is actual rendered simulation; no policy actions were
changed for the recording. The loop boundary is an edit, not a simulation reset.

## GPU physics + GPU learning progress

The experimental unified Metal effort has working rigid-body dynamics, CAD ground
contacts, two-body self-contact constraint assembly/response, short PPO/export
smokes, and a bounded split-run learner/environment continuation test. Self-contact
narrowphase still runs on CPU MuJoCo and supplies geometry to the Metal solver;
reset-time model-constant synchronization also uses the CPU.

This is currently **CPU narrowphase + Metal physics solve + MPS PPO**, not a fully
GPU-resident training loop. Remaining work includes native narrowphase, broader
task qualification, and end-to-end performance validation. Short smoke evidence is not evidence
of learned walking on that backend. No matched end-to-end speed advantage over
the CPU-physics baseline has been established. The walking results above apply
only to the completed CPU-physics/MPS-PPO run.
