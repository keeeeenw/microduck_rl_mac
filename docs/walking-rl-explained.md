# How walking training works

**This is a detailed explanation of existing methods, not a contribution of a new
RL algorithm or walking-training method by this project.** PPO and GAE are standard
algorithms, implemented here through RSL-RL. The MicroDuck task, reward recipe,
randomization and BAM actuator setup come from the upstream MicroDuck RL/mjlab
ecosystem. This project's work concerns Mac execution, integration and validation.

This guide follows the flat walking configuration inspected on 20 September 2026.
The numerical settings below describe that configuration, not universal PPO
requirements. Curricula change some settings during training. The reward examples
are important components, not an exhaustive specification of every task term.

## 1. The complete loop

$$
\text{observation}\rightarrow\text{policy action}\rightarrow
\text{motor model and physics}\rightarrow\text{reward}\rightarrow
\text{GAE}\rightarrow\text{PPO update}
$$

The environment defines desirable behavior through rewards. The critic predicts
future return. Advantage estimates compare outcomes with those predictions. PPO
uses those estimates to update the actor.

In the recommended hybrid Mac path:

| Work | Execution |
| --- | --- |
| MuJoCo dynamics, collisions, constraints and integration | CPU |
| Policy inference and BAM tensor calculations | Apple GPU, PyTorch MPS |
| Reward/observation/manager tensor calculations | MPS with Python orchestration |
| Rollout tensors, advantages and PPO optimization | MPS |

Rollout collection therefore spans CPU and GPU. CPU physics does not mean CPU PPO.
Unified physical memory does not remove this adapter's framework transfers.

## 2. Robot state and observations

Let $s_t$ denote the full environment state, including positions, velocities,
contacts, actuator history and randomized physical parameters. The policy sees an
observation derived from it:

$$
o_t=h(s_t,c_t,\text{sensor history})+\epsilon_t.
$$

Here $c_t$ is the command and $\epsilon_t$ represents observation noise. Configured
sensor delays and biases also participate in the observation pipeline.

MicroDuck has **14 actuated joints**: five per leg and four neck/head joints.
The flat model has 20 generalized velocity degrees of freedom (14 joints plus
six floating-base degrees of freedom) and 21 position coordinates because base
orientation uses four quaternion components.

The actor's **61 inputs** comprise:

$$
o_t=[\omega_t^{(3)},\ g_t^{body\,(3)},\
(q_t-q_{ref})^{(14)},\ \dot q_t^{(14)},\ a_{t-1}^{(14)},\ c_t^{(13)}].
$$

The command block contains twist (3), head pose (4), and body pose (6).
The critic receives a separate **76-dimensional** observation $o_t^V$ with
privileged simulation information. Actor and critic inputs are not interchangeable.

Observation normalization has the form

$$
\tilde o_t=\frac{o_t-\mu_o}{\sqrt{\sigma_o^2+\varepsilon}}.
$$

Running statistics are maintained during training. The deployment export must
include the actor's normalization.

## 3. Actor and action distribution

Actor and critic are separate MLPs with hidden widths **512, 256, 128** and ELU
activations. For the actor, write

$$
h_0=\tilde o_t,\qquad h_i=\operatorname{ELU}(W_i h_{i-1}+b_i),\quad i=1,2,3,
$$
$$
\mu_\theta(o_t)=W_4h_3+b_4.
$$

Training samples a 14-dimensional diagonal Gaussian:

$$
a_t\sim\pi_\theta(\cdot\mid o_t)
=\mathcal N(\mu_\theta(o_t),\operatorname{diag}(\sigma_\theta^2)).
$$

The exploration standard deviations are learned. Deterministic playback uses the
mean action. The position-action mapping is schematically

$$
q_t^{target}=q_{default}+Sa_t,
$$

where $S$ contains configured action scales. These are position targets, not
directly commanded physical torques.

## 4. BAM and physical rollout

Control runs at 50 Hz, with four 200 Hz physics substeps:

$$
\Delta t=0.020\ \mathrm{s},\qquad h=0.005\ \mathrm{s},\qquad \Delta t=4h.
$$

At physics substep $k$, BAM uses delayed commands, motor state and voltage to
produce torque and friction/damping parameters:

$$
(\tau_k,f_k,d_k)=\operatorname{BAM}(q_k,\dot q_k,
q^{target}_{k-\ell},\text{voltage},\text{motor history}).
$$

This is a schematic interface, not the complete BAM M6 equation set. BAM includes
motor voltage behavior and friction effects rather than assuming an ideal PD motor.

The generalized dynamics have the form

$$
M(q)\ddot q+b(q,\dot q)=\tau_{actuator}+\tau_{passive}
+\tau_{external}+J(q)^T\lambda.
$$

$M$ includes configured armature; $b$ contains gravity and velocity-dependent bias;
$J^T\lambda$ represents constraint forces. Here $q$ denotes generalized positions,
with appropriate quaternion handling for the base. MuJoCo solves constraints and
advances the state using **ImplicitFast**. This dynamics equation alone does not
specify that integrator.

Domain randomization samples physical/sensor parameters, schematically
$\xi\sim p_k(\xi)$, where curriculum stage $k$ may change the distribution.
The learning objective then averages over this variation as well as policy
sampling. Randomization must not accumulate incorrectly across resets.

## 5. Environment rewards

With the configured timestep scaling, the reward is

$$
r_t=\Delta t\sum_i w_i(t)\rho_i(s_t,a_t,c_t).
$$

Positive weights reward positive-valued terms; negative weights penalize costs.
Some upstream terms already carry a sign, so each term's implementation determines
the correct weight convention. Curricula can change $w_i$.

For body-frame velocities, important flat-task terms are:

**Linear velocity tracking**, weight 2.0:

$$
\rho_{linear}=\exp\left[-\frac{(v_x-v_x^*)^2+(v_y-v_y^*)^2+v_z^2}{0.1}\right].
$$

**Angular velocity tracking**, weight 2.0:

$$
\rho_{angular}=\exp\left[-\frac{(\omega_z-\omega_z^*)^2+\omega_x^2+\omega_y^2}{0.5}\right].
$$

The angular term rewards turning-command tracking while discouraging unwanted
roll/pitch angular motion.

**Upright posture**, weight 2.0 on flat terrain:

$$
\rho_{upright}=\exp\left[-\frac{(g_x^{body})^2+(g_y^{body})^2}{0.05}\right],
$$

where $g^{body}$ is the projected gravity direction.

**Action-change cost**:

$$
\rho_{action\ rate}=\|a_t-a_{t-1}\|_2^2.
$$

Its weight progresses from -0.1 to -1.0 as the curriculum increases smoothing.

**Air-time reward**, weight 3.0:

$$
\rho_{air}=\mathbf1_{\text{movement commanded}}
\sum_f\mathbf1_{0.125<T_f^{air}<0.300}.
$$

This counts feet whose current airborne duration lies in the configured window.
Movement gating uses commanded planar speed plus absolute commanded yaw rate,
with threshold 0.01 in this configuration.

Other terms address leg posture, clearance/swing height, slipping, joint limits,
self-collisions, body rotation/angular momentum, and head/body command tracking.
Consult the task configuration for the full sum and curricula. **Advantage is not
part of the environment reward definition.**

## 6. Return and critic

The policy objective is expected discounted return:

$$
J(\theta)=\mathbb E_{\pi_\theta,\xi}
\left[\sum_{t=0}^{T-1}\gamma^t r_t\right],\qquad \gamma=0.99.
$$

The critic predicts remaining discounted reward:

$$
V_\phi(o_t^V)\approx\mathbb E\left[\sum_{l\ge0}\gamma^l r_{t+l}\mid o_t^V\right].
$$

It provides a baseline for judging outcomes, not a new definition of good walking.

## 7. Temporal-difference residual and GAE

For a nonterminal transition, the TD residual is

$$
\delta_t=r_t+\gamma V_{old}(o_{t+1}^V)-V_{old}(o_t^V).
$$

The implemented backward recurrence uses a boundary mask $m_t=1-done_t$:

$$
\delta_t=\bar r_t+\gamma m_t V_{next}-V_{old}(o_t^V),
\qquad
\hat A_t=\delta_t+\gamma\lambda_{GAE}m_t\hat A_{t+1},
$$

with $\lambda_{GAE}=0.95$. Here $\bar r_t$ is the stored reward after any timeout
bootstrap adjustment. The installed RSL-RL implementation adds
$\gamma V_{old}(o_t^V)\mathbf1_{timeout}$ before storing the transition; the done
mask then stops the recurrence across reset boundaries. This is distinct from
treating time-limit truncations as physical terminal failures. At the end of a
nonterminal rollout segment, the last critic value supplies the bootstrap.

Without boundaries, GAE expands to

$$
\hat A_t=\sum_{l\ge0}(\gamma\lambda_{GAE})^l\delta_{t+l}.
$$

The critic target and normalized actor advantage are

$$
\hat R_t=\hat A_t+V_{old}(o_t^V),\qquad
\tilde A_t=\frac{\hat A_t-\operatorname{mean}(\hat A)}
{\operatorname{std}(\hat A)+\varepsilon}.
$$

Positive advantage means better than predicted. It does not require a positive
raw reward. Returns/advantages are fixed targets during the PPO update.

## 8. PPO actor objective

Stored actions were sampled from the old policy. Their probability ratio is

$$
\rho_t^\pi(\theta)=\frac{\pi_\theta(a_t\mid o_t)}{\pi_{old}(a_t\mid o_t)}
=\exp(\log\pi_\theta(a_t\mid o_t)-\log\pi_{old}(a_t\mid o_t)).
$$

For a diagonal Gaussian, the log probability is

$$
\log\pi_\theta(a\mid o)=-\frac12\sum_{j=1}^{14}
\left[\frac{(a_j-\mu_j)^2}{\sigma_j^2}+2\log\sigma_j+\log(2\pi)\right].
$$

The minimized clipped actor loss is

$$
L_{actor}=-\mathbb E_t\left[\min\left(
\rho_t^\pi\tilde A_t,
\operatorname{clip}(\rho_t^\pi,1-\epsilon,1+\epsilon)\tilde A_t
\right)\right],\qquad \epsilon=0.2.
$$

Clipping limits the incentive to move too far from the behavior policy. It is not
a hard bound on every probability ratio. Old action log probabilities are held
fixed while the new actor is optimized.

## 9. Value loss, entropy and total loss

The configured clipped value prediction and loss are

$$
V_t^{clip}=V_t^{old}+\operatorname{clip}(V_\phi(o_t^V)-V_t^{old},-\epsilon,\epsilon),
$$
$$
L_{value}=\mathbb E_t\left[\max\left(
(V_\phi(o_t^V)-\hat R_t)^2,(V_t^{clip}-\hat R_t)^2\right)\right].
$$

Gaussian differential entropy is

$$
H(\pi)=\sum_{j=1}^{14}\left[\log\sigma_j+\tfrac12\log(2\pi e)\right].
$$

The minimized combined objective is

$$
\boxed{L=L_{actor}+1.0L_{value}-0.01\mathbb E_t[H(\pi_\theta)]}.
$$

The entropy term encourages exploration. Differential entropy can be negative;
that alone is not a training error.

## 10. Gradients, Adam and rollout cadence

Compute gradients through actor and critic, then clip their combined norm:

$$
g=\nabla_{\theta,\phi}L,\qquad
g_c=g\min\left(1,\frac{1.0}{\|g\|_2+\varepsilon}\right).
$$

Adam maintains first and second moments (optimizer index $k$):

$$
m_k=\beta_1m_{k-1}+(1-\beta_1)g_c,\qquad
v_k=\beta_2v_{k-1}+(1-\beta_2)g_c^2,
$$
$$
\hat m_k=\frac{m_k}{1-\beta_1^k},\quad
\hat v_k=\frac{v_k}{1-\beta_2^k},\quad
\psi_{k+1}=\psi_k-\alpha_k\frac{\hat m_k}{\sqrt{\hat v_k}+\varepsilon},
$$

where $\psi$ collects actor/critic parameters and operations on $v_k$ are
elementwise. The learning rate is adaptive, with desired policy KL 0.01.
For diagonal Gaussians, the old-to-new KL used to assess movement has the form

$$
D_{KL}(\pi_{old}\|\pi_{new})=
\sum_j\left[\log\frac{\sigma_{new,j}}{\sigma_{old,j}}+
\frac{\sigma_{old,j}^2+(\mu_{old,j}-\mu_{new,j})^2}{2\sigma_{new,j}^2}-\frac12\right].
$$

For the documented 4,096-environment run, each update collects

$$
4096\times24=98{,}304\text{ transitions}.
$$

Five epochs with four minibatches per epoch produce 20 optimizer steps per update.
The same rollout is reused for those epochs, then fresh experience is collected.
This bounded reuse is part of on-policy PPO; it is not a long-lived off-policy
replay buffer.

**No gradients pass through MuJoCo or BAM in this pipeline.** Physics supplies
experience. Autograd differentiates the actor/critic loss, not the trajectory
through the simulator.

## 11. Deployment and interpretation

Deployment uses the actor, baked-in observation normalization and matching action
conventions. The critic, rewards, GAE and optimizer are training components and
are unnecessary for policy inference. A portable ONNX artifact does not by itself
validate Linux behavior or physical-robot transfer; those need separate checks.

Reward design specifies the behavior we want. Simulation and randomization specify
the situations the policy learns in. PPO/GAE provide the standard learning method.
Changing the CPU/GPU execution backend should preserve this contract.

## References and implementation sources

- Schulman et al., [Proximal Policy Optimization Algorithms (2017)](https://arxiv.org/abs/1707.06347).
- Schulman et al., [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438).
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl): actor–critic training implementation.
- [Upstream MicroDuck RL](https://github.com/pollen-robotics/microduck_rl): robot task and training recipe.
- [mjlab](https://github.com/mujocolab/mjlab): simulation/environment and manager framework.
- [BAM](https://github.com/Rhoban/bam): actuator modeling.
- [Walking task configuration](../src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py),
  [custom task terms](../src/mjlab_microduck/tasks/mdp.py), and
  [native training entry point](../src/mjlab_microduck/native_gpu/train.py).
- [Mac training and playback guide](mac-training.md).

This document is explanatory documentation of these existing methods. It does
not claim their algorithms, reward recipe or actuator model as original work of
the Mac project.
