"""Shared evaluator helper functions for state verification, fall detection, and trajectory metrics."""

import math
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch


def check_state_finite(
    qpos: torch.Tensor,
    qvel: torch.Tensor,
    actions: Optional[torch.Tensor] = None,
    obs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Evaluate whether all components of the physics and policy state are strictly finite.
    
    Args:
        qpos: Generalized positions (num_envs, nq)
        qvel: Generalized velocities (num_envs, nv)
        actions: Optional policy actions (num_envs, nu)
        obs: Optional actor observations (num_envs, nobs)
        
    Returns:
        1D boolean tensor of shape (num_envs,) where True indicates all checked values are finite.
    """
    is_finite = torch.isfinite(qpos).all(dim=-1) & torch.isfinite(qvel).all(dim=-1)
    if actions is not None:
        is_finite = is_finite & torch.isfinite(actions).all(dim=-1)
    if obs is not None:
        is_finite = is_finite & torch.isfinite(obs).all(dim=-1)
    return is_finite


def check_fall_rule(
    root_pos: torch.Tensor,
    tilt_deg: torch.Tensor,
    is_finite: torch.Tensor,
    min_height_threshold: float = 0.065,
    max_tilt_threshold: float = 60.0,
) -> torch.Tensor:
    """Evaluate explicit deployment playback fall rule.
    
    Fall occurs if:
    - Root height < min_height_threshold (0.065 m)
    - Trunk tilt > max_tilt_threshold (60.0 deg)
    - Any state element is non-finite (NaN or Inf)
    - Root height is NaN or tilt is NaN
    
    Args:
        root_pos: Root position tensor (num_envs, 3)
        tilt_deg: Trunk tilt angle in degrees (num_envs,)
        is_finite: Boolean tensor indicating whether state is finite (num_envs,)
        min_height_threshold: Height limit below which robot is considered fallen
        max_tilt_threshold: Tilt angle limit above which robot is considered fallen
        
    Returns:
        1D boolean tensor of shape (num_envs,) where True indicates a fall-rule violation.
    """
    height = root_pos[:, 2]
    nan_height = torch.isnan(height)
    nan_tilt = torch.isnan(tilt_deg)
    
    has_fallen = (
        (~is_finite)
        | nan_height
        | (height < min_height_threshold)
        | nan_tilt
        | (tilt_deg > max_tilt_threshold)
    )
    return has_fallen


def extract_yaw_from_quat(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Extract Euler yaw angle around +Z from MuJoCo (w, x, y, z) quaternion.
    
    yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
    """
    w = quat_wxyz[:, 0]
    x = quat_wxyz[:, 1]
    y = quat_wxyz[:, 2]
    z = quat_wxyz[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def unwrap_yaw_trajectory(yaw_trajectory: np.ndarray) -> np.ndarray:
    """Unwrap yaw angle time-series along the step dimension and return accumulated heading change.
    
    Args:
        yaw_trajectory: Array of shape (T, num_envs) representing yaw at each step in radians.
        
    Returns:
        1D array of shape (num_envs,) representing total accumulated heading change in radians.
    """
    if yaw_trajectory.ndim != 2:
        raise ValueError(f"Expected 2D array (T, num_envs), got shape {yaw_trajectory.shape}")
    unwrapped = np.unwrap(yaw_trajectory, axis=0)
    return unwrapped[-1] - unwrapped[0]


def compute_walk_to_stop_intervals(
    t_series: np.ndarray,
    vx_series: np.ndarray,
    target_walk_vx: float = 0.30,
    stop_time_s: float = 10.0,
    settle_time_s: Optional[float] = None,
    transient_cutoff_s: float = 2.0,
) -> Dict[str, Any]:
    """Separately summarize walking phase and settled stop phase for walk_to_stop.
    
    Args:
        t_series: Array of timestamps of shape (T,)
        vx_series: Array of forward body velocities of shape (T, num_envs)
        target_walk_vx: Commanded forward velocity during walking phase
        stop_time_s: Timestamp when zero-velocity command is issued
        settle_time_s: Timestamp after which robot should be fully settled/stopped (default stop_time_s + 2.0)
        transient_cutoff_s: Initial warmup transient to exclude from walking metrics
        
    Returns:
        Dictionary containing separated metrics for walking and settled stop intervals.
    """
    num_envs = vx_series.shape[1] if vx_series.ndim > 1 else 1
    total_duration = float(t_series[-1]) if len(t_series) > 0 else 0.0
    
    if settle_time_s is None:
        settle_time_s = min(stop_time_s + 2.0, total_duration)

    walk_start = min(transient_cutoff_s, stop_time_s / 2.0) if stop_time_s > 0 else 0.0
    # 1. Steady walking mask
    walk_mask = (t_series >= walk_start) & (t_series < stop_time_s)
    # 2. Settled stopped mask
    stop_mask = (t_series >= settle_time_s)
    
    walk_vx = vx_series[walk_mask]  # shape: (N_walk, num_envs)
    stop_vx = vx_series[stop_mask]  # shape: (N_stop, num_envs)
    
    if len(walk_vx) > 0:
        walk_mean_per_env = np.mean(walk_vx, axis=0)
        walk_rmse_per_env = np.sqrt(np.mean((walk_vx - target_walk_vx) ** 2, axis=0))
    else:
        walk_mean_per_env = np.zeros(num_envs)
        walk_rmse_per_env = np.zeros(num_envs)
        
    if len(stop_vx) > 0:
        stop_mean_per_env = np.mean(stop_vx, axis=0)
        stop_rms_per_env = np.sqrt(np.mean(stop_vx ** 2, axis=0))
        stop_max_abs_per_env = np.max(np.abs(stop_vx), axis=0)
    else:
        stop_mean_per_env = np.zeros(num_envs)
        stop_rms_per_env = np.zeros(num_envs)
        stop_max_abs_per_env = np.zeros(num_envs)
    
    return {
        "walking_phase": {
            "interval_s": [float(walk_start), float(stop_time_s)],
            "mean_forward_vel_mps": float(np.mean(walk_mean_per_env)),
            "per_env_forward_vel_mps": walk_mean_per_env.tolist(),
            "mean_tracking_rmse_vx": float(np.mean(walk_rmse_per_env)),
            "per_env_tracking_rmse_vx": walk_rmse_per_env.tolist(),
        },
        "settled_stop_phase": {
            "interval_s": [float(settle_time_s), total_duration],
            "mean_forward_vel_mps": float(np.mean(stop_mean_per_env)),
            "per_env_forward_vel_mps": stop_mean_per_env.tolist(),
            "rms_forward_vel_mps": float(np.mean(stop_rms_per_env)),
            "per_env_rms_forward_vel_mps": stop_rms_per_env.tolist(),
            "max_abs_forward_vel_mps": float(np.max(stop_max_abs_per_env)),
        }
    }

