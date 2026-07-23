"""IMU (inertial measurement unit) utilities.

Provides BMI270 serial line parsing, quaternion math for orientation
tracking, a dead-reckoned trajectory integrator, and a streaming
moving-average filter.
"""

from __future__ import annotations

import re
from collections import deque
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

G_MPS2 = 9.80665
FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
IMU_SAMPLE_PATTERN = re.compile(
    rf"accel\[g\]\s+x=\s*({FLOAT_PATTERN})\s+y=\s*({FLOAT_PATTERN})\s+z=\s*({FLOAT_PATTERN})"
    rf"\s+\|\s+gyro\[dps\]\s+x=\s*({FLOAT_PATTERN})\s+y=\s*({FLOAT_PATTERN})\s+z=\s*({FLOAT_PATTERN})"
)

CHANNEL_NAMES = ("ax", "ay", "az", "gx", "gy", "gz")

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_imu_line(line: str) -> Optional[tuple[float, float, float, float, float, float]]:
    """Parse one ESP32 BMI270 serial line into six float values.

    Returns ``(ax, ay, az, gx, gy, gz)`` — accelerometer in g and
    gyroscope in deg/s — or *None* if the line doesn't match.
    """
    match = IMU_SAMPLE_PATTERN.search(line)
    if not match:
        return None
    return tuple(float(v) for v in match.groups())


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------


def quat_normalized(q: np.ndarray) -> np.ndarray:
    """Return a unit-length quaternion ``(w, x, y, z)``."""
    norm = float(np.linalg.norm(q))
    return q / norm if norm > 1e-12 else q


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Multiply two quaternions ``(w, x, y, z)``."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=float,
    )


def quat_from_two_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the quaternion that rotates *source* onto *target*."""
    s = source / (np.linalg.norm(source) + 1e-12)
    t = target / (np.linalg.norm(target) + 1e-12)
    dot = float(np.dot(s, t))

    if dot > 0.999999:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    if dot < -0.999999:
        axis = np.cross(s, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(s, np.array([0.0, 1.0, 0.0]))
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        return np.array([0.0, axis[0], axis[1], axis[2]], dtype=float)

    axis = np.cross(s, t)
    return quat_normalized(np.array([1.0 + dot, axis[0], axis[1], axis[2]], dtype=float))


def quat_from_angular_velocity(omega_rad_s: np.ndarray, dt: float) -> np.ndarray:
    """Integrate angular velocity over *dt* seconds into a quaternion."""
    angle = float(np.linalg.norm(omega_rad_s) * dt)
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    axis = omega_rad_s / np.linalg.norm(omega_rad_s)
    ha = 0.5 * angle
    return np.array([np.cos(ha), *(axis * np.sin(ha))], dtype=float)


def quat_rotate(q: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate *vector* by quaternion *q*."""
    qv = q[1:]
    uv = np.cross(qv, vector)
    uuv = np.cross(qv, uv)
    return vector + 2.0 * (q[0] * uv + uuv)


# ---------------------------------------------------------------------------
# Trajectory reconstruction (dead reckoning)
# ---------------------------------------------------------------------------


def compute_trajectory(
    times: np.ndarray,
    accel_g: np.ndarray,
    gyro_dps: np.ndarray,
    *,
    gravity_axis: int = 2,
    gyro_bias: Optional[np.ndarray] = None,
    accel_deadband: float = 0.08,
    velocity_damping: float = 0.08,
    stationary_accel_threshold: float = 0.18,
    stationary_gyro_threshold: float = 2.0,
    max_dt: float = 0.2,
) -> dict:
    """Dead-reckoned 3D trajectory from IMU accelerometer + gyroscope data.

    Parameters
    ----------
    times : ndarray, shape (N,)
        Monotonic timestamps in seconds (*not* necessarily evenly spaced).
    accel_g : ndarray, shape (N, 3)
        Accelerometer readings in g (m/s² after scaling).
    gyro_dps : ndarray, shape (N, 3)
        Gyroscope readings in degrees per second.
    gravity_axis : int
        Which axis gravity is expected along (0=x, 1=y, 2=z).
    gyro_bias : ndarray, shape (3,) or None
        Static gyroscope bias to subtract.  Estimated from the first few
        samples if *None*.
    **kwargs
        Tuning parameters (see defaults above).

    Returns
    -------
    dict with keys ``position_m``, ``velocity_mps``, ``linear_accel_mps2``,
    ``orientation`` (quaternion history), ``gravity_norm_g``, ``gyro_bias_dps``.
    """
    n = len(times)
    if n < 3:
        raise ValueError(f"Need at least 3 samples, got {n}.")

    rel_time = times.astype(float) - float(times[0])

    # Estimate gravity direction from mean acceleration (first samples).
    baseline_count = min(30, n)
    mean_accel = np.mean(accel_g[:baseline_count], axis=0)
    gravity_norm = float(np.linalg.norm(mean_accel))
    if gravity_norm < 0.2:
        raise ValueError(
            f"Gravity magnitude too small ({gravity_norm:.3f} g) — "
            "check that the IMU is not in free-fall."
        )

    gyro_bias = (
        np.mean(gyro_dps[:baseline_count], axis=0)
        if gyro_bias is None
        else np.asarray(gyro_bias, dtype=float)
    )

    # Gravity vector in world frame.
    world_gravity = np.zeros(3, dtype=float)
    world_gravity[gravity_axis] = gravity_norm

    # Initial orientation: align measured gravity with world gravity.
    q_body_to_world = quat_from_two_vectors(mean_accel, world_gravity)

    position = np.zeros((n, 3), dtype=float)
    velocity = np.zeros(3, dtype=float)
    lin_accel = np.zeros((n, 3), dtype=float)
    orientation = np.zeros((n, 4), dtype=float)
    orientation[0] = q_body_to_world

    for i in range(1, n):
        dt = float(rel_time[i] - rel_time[i - 1])
        if dt <= 0.0 or dt > max_dt:
            orientation[i] = orientation[i - 1]
            continue

        # Gyroscope integration.
        gyro_corrected = gyro_dps[i] - gyro_bias
        delta_q = quat_from_angular_velocity(np.deg2rad(gyro_corrected), dt)
        q_body_to_world = quat_normalized(quat_multiply(q_body_to_world, delta_q))
        orientation[i] = q_body_to_world

        # Accelerometer: rotate to world frame, remove gravity.
        specific_force_world = quat_rotate(q_body_to_world, accel_g[i])
        linear = (specific_force_world - world_gravity) * G_MPS2

        if np.linalg.norm(linear) < accel_deadband:
            linear[:] = 0.0

        stationary = (
            np.linalg.norm(linear) < stationary_accel_threshold
            and np.linalg.norm(gyro_corrected) < stationary_gyro_threshold
        )

        old_vel = velocity.copy()
        if stationary:
            velocity[:] = 0.0
            linear[:] = 0.0
        else:
            velocity += linear * dt
            if velocity_damping > 0:
                velocity *= np.exp(-velocity_damping * dt)

        position[i] = position[i - 1] + 0.5 * (old_vel + velocity) * dt
        lin_accel[i] = linear

    return {
        "position_m": position,
        "velocity_mps": velocity,  # final velocity only
        "linear_accel_mps2": lin_accel,
        "orientation_quat": orientation,
        "gravity_norm_g": gravity_norm,
        "gyro_bias_dps": gyro_bias,
    }


# ---------------------------------------------------------------------------
# Moving-average smoothing
# ---------------------------------------------------------------------------


class MovingAverageFilter:
    """Streaming moving average over a fixed-size window.

    Parameters
    ----------
    window_size : int
        Number of samples to average over.
    """

    def __init__(self, window_size: int) -> None:
        self.window_size = window_size
        self.window: deque[float] = deque(maxlen=window_size)
        self.running_sum = 0.0

    def update(self, new_value: float) -> float:
        if len(self.window) == self.window_size:
            self.running_sum -= self.window[0]
        self.window.append(new_value)
        self.running_sum += new_value
        return self.running_sum / len(self.window)


def smooth_samples(
    samples: list[tuple[float, ...]],
    window_size: int,
    num_channels: int = 6,
) -> list[tuple[float, ...]]:
    """Apply a moving-average filter to each channel of a sample list.

    Parameters
    ----------
    samples : list of tuple
        Each tuple is ``(timestamp, ch0, ch1, ..., chN)``.
    window_size : int
        Number of samples to average over (1 = no smoothing).
    num_channels : int
        Number of value channels after the timestamp.

    Returns
    -------
    list of tuple
        Smoothed samples with the same structure.
    """
    if window_size <= 1:
        return list(samples)

    filters = [MovingAverageFilter(window_size) for _ in range(num_channels)]
    smoothed: list[tuple[float, ...]] = []
    for sample in samples:
        ts = sample[0]
        values = sample[1:]
        smoothed_values = tuple(
            filters[i].update(float(values[i])) for i in range(num_channels)
        )
        smoothed.append((ts, *smoothed_values))
    return smoothed


__all__ = [
    "G_MPS2",
    "IMU_SAMPLE_PATTERN",
    "CHANNEL_NAMES",
    "parse_imu_line",
    "quat_normalized",
    "quat_multiply",
    "quat_from_two_vectors",
    "quat_from_angular_velocity",
    "quat_rotate",
    "compute_trajectory",
    "MovingAverageFilter",
    "smooth_samples",
]
