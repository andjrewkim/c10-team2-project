from __future__ import annotations

import argparse
import os
import pickle
import signal
import time
from collections import deque
from pathlib import Path

import numpy as np

from src.sensors.imu_reader import ImuReader
from src.sensors.mmWave.mmwave_reader import MmWaveReader
from src.sensors.uwb_reader import UwbReader

SENSOR_REGISTRY = {
    "mmwave": MmWaveReader,
    "imu": ImuReader,
    "uwb": UwbReader,
}

SENSOR_FEATURE_COUNTS = {
    "mmwave": 10,
    "imu": 6,
    "uwb": 6,
}

# IMU processing parameters (set from CLI args before starting)
_gyro_gain: float = 1.0
_gyro_deadband: float = 0.0
_accel_gain: float = 1.0
# Whether to compute FFT spectral + correlation features (v3+ models)
_use_extra_features: bool = False

# High-pass filter state for gravity removal from accelerometer
# Tracks the slowly-changing gravity vector so we can isolate true linear acceleration
_accel_lp: np.ndarray | None = None  # [ax, ay, az] low-pass filtered (gravity estimate)


def extract_features_from_reading(reading: any, sensor_type: str) -> list[float]:
    if sensor_type == "mmwave":
        data = reading.data
        points = data.get("points", [])
        num_points = data.get("num_points", len(points))
        if not points:
            return [float(num_points), 0.0, 0.0, 0.0, 0.0, 0.0]
        xs = np.array([p.get("x", 0) for p in points])
        ys = np.array([p.get("y", 0) for p in points])
        return [
            float(num_points),
            float(np.mean(xs)), float(np.std(xs)),
            float(np.mean(ys)), float(np.std(ys)),
            float(np.sqrt(np.mean(xs)**2 + np.mean(ys)**2)),
        ]
    elif sensor_type == "imu":
        data = reading.data
        accel = data.get("accel", [0, 0, 0])
        gyro = data.get("gyro", [0, 0, 0])
        gx = float(gyro[0]) if len(gyro) > 0 else 0.0
        gy = float(gyro[1]) if len(gyro) > 1 else 0.0
        gz = float(gyro[2]) if len(gyro) > 2 else 0.0
        ax = float(accel[0]) if len(accel) > 0 else 0.0
        ay = float(accel[1]) if len(accel) > 1 else 0.0
        az = float(accel[2]) if len(accel) > 2 else 0.0
        # Apply gains & deadband (set via CLI args)
        ax *= _accel_gain
        ay *= _accel_gain
        az *= _accel_gain
        gx *= _gyro_gain
        gy *= _gyro_gain
        gz *= _gyro_gain
        if abs(gx) < _gyro_deadband:
            gx = 0.0
        if abs(gy) < _gyro_deadband:
            gy = 0.0
        if abs(gz) < _gyro_deadband:
            gz = 0.0
        return [ax, ay, az, gx, gy, gz]
    elif sensor_type == "uwb":
        data = reading.data
        ranges = data.get("ranges_cm", [])
        raw_ranges = data.get("raw_ranges", [])
        if not ranges:
            ranges = raw_ranges
        ranges_m = [r / 100.0 for r in ranges if isinstance(r, (int, float)) and r > 0]
        if not ranges_m:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        return [
            float(len(ranges_m)),
            float(np.mean(ranges_m)),
            float(np.std(ranges_m)) if len(ranges_m) > 1 else 0.0,
            float(np.min(ranges_m)),
            float(np.max(ranges_m)),
            float(np.median(ranges_m)),
        ]
    return []


def _compute_linear_accel(raw_accel: list[float], alpha: float = 0.80) -> list[float]:
    """Remove gravity from raw accelerometer using a single-pole high-pass filter.

    The low-pass filter tracks the gravity vector, which changes as the wrist
    rotates. Subtracting it from raw accel isolates true linear acceleration
    (actual movement), solving the false-positive problem where wrist rotation
    masquerades as linear motion through gravity projection.

    alpha: smoothing factor. 0.80 ≈ 125ms time constant at ~40 fps (cutoff ~1.3 Hz).
           Higher = suppresses slow rotation better but adapts slower (movement
                    score lingers longer after gesture stops).
           Lower = movement drops back to idle faster after gesture ends.
    """
    global _accel_lp
    raw = np.array(raw_accel, dtype=float)
    if _accel_lp is None:
        _accel_lp = raw.copy()
    else:
        # Low-pass: slowly track gravity
        _accel_lp = alpha * _accel_lp + (1 - alpha) * raw
    # Linear accel = raw - gravity estimate  (residual = true linear acceleration)
    linear = raw - _accel_lp
    return linear.tolist()


def _movement_score(readings: list, sensor_type: str) -> float:
    """Movement score based on linear acceleration (gravity removed via HPF).

    Uses a single-pole high-pass filter on the accelerometer to track and
    subtract the gravity vector. This ensures that wrist rotation (which changes
    how gravity projects onto sensor axes) produces minimal movement signal.
    Only true linear acceleration contributes to the score.

    Applies the global ``_accel_gain`` so that ``--accel-gain`` actually affects
    when the system enters/leaves idle — not just the model's feature vector.

    Rotational gestures (soli, making-fist-open, t-arm, bye-bye) are caught
    by _check_gyro_oscillation() which detects their distinctive gyro patterns
    and overrides idle even when linear acceleration is low.
    """
    if sensor_type == "imu":
        if len(readings) < 2:
            return 0.0
        # Apply gains so --accel-gain affects idle detection, not just model features
        linear_accels = []
        for r in readings:
            raw = r.data.get("accel", [0, 0, 0])
            # Apply accel gain BEFORE the HPF so it affects movement score
            raw_gained = [v * _accel_gain for v in raw]
            linear = _compute_linear_accel(raw_gained)
            linear_accels.append(linear)
        # Movement score from linear accel deltas (transients only)
        deltas = []
        for i in range(1, len(linear_accels)):
            d = sum(abs(linear_accels[i][j] - linear_accels[i-1][j]) for j in range(3))
            deltas.append(d)
        return float(np.mean(deltas)) if deltas else 0.0
    elif sensor_type == "mmwave":
        cents = []
        for r in readings:
            data = r.data
            pts = data.get("points", [])
            if pts:
                cents.append((np.mean([p.get("x",0) for p in pts]), np.mean([p.get("y",0) for p in pts])))
            else:
                cents.append((0.0, 0.0))
        dist = 0.0
        for i in range(1, len(cents)):
            dist += np.sqrt((cents[i][0]-cents[i-1][0])**2 + (cents[i][1]-cents[i-1][1])**2)
        return float(dist)
    return 0.0


def _check_gyro_oscillation(readings: list) -> bool:
    """Detect oscillatory gestures (soli finger rub) via gyro signature.

    The HPF-based movement score removes gravity projection, which suppresses
    false push/pull triggers from wrist rotation. But rotational gestures
    (soli, making-fist-open, t-arm, bye-bye) also show low linear accel.
    This function detects their distinctive gyro oscillation and overrides
    idle so the model can classify them.

    Applies the same gyro gain + deadband as the feature pipeline so that
    the override is consistent with what the model actually sees.

    Uses deliberately high thresholds so that only deliberate finger-rub
    oscillation triggers the override — tiny wrist jitter at rest stays idle.

    Returns True if at least 2 of 3 gyro channels show both:
      - Strong amplitude (RMS > 4.0 dps after deadband)
      - Fast oscillation (zero-crossing rate > 0.35)
    """
    if len(readings) < 5:
        return False
    gyros = []
    for r in readings:
        g = r.data.get("gyro", [0, 0, 0])
        gx = g[0] * _gyro_gain if abs(g[0] * _gyro_gain) >= _gyro_deadband else 0.0
        gy = g[1] * _gyro_gain if abs(g[1] * _gyro_gain) >= _gyro_deadband else 0.0
        gz = g[2] * _gyro_gain if abs(g[2] * _gyro_gain) >= _gyro_deadband else 0.0
        gyros.append([gx, gy, gz])
    gyro_data = np.array(gyros)
    oscillating = 0
    for c in range(3):
        col = gyro_data[:, c]
        rms = float(np.sqrt(np.mean(col ** 2)))
        if rms < 4.0:
            continue
        centered = col - np.mean(col)
        crossings = int(np.sum((centered[:-1] * centered[1:]) < 0))
        zcr = crossings / len(col) if len(col) > 0 else 0.0
        if zcr > 0.35:
            oscillating += 1
    return oscillating >= 2

def _gather_readings(window: list[dict], name: str) -> list:
    return [f[name] for f in window if name in f]


def _compute_features(readings: list, sensor_type: str) -> list[float]:
    sensor_feats = np.array([extract_features_from_reading(r, sensor_type) for r in readings])
    if len(sensor_feats) == 0:
        n = (60 if _use_extra_features else 32) + 8 * (len(readings) - 1) if sensor_type == "imu" else 13
        return [0.0] * n

    if sensor_type == "imu":
        accel = sensor_feats[:, 0:3]
        gyro = sensor_feats[:, 3:6]
        gyro_mag = np.sqrt(np.sum(gyro**2, axis=1, keepdims=True))
        accel_mag = np.sqrt(np.sum(accel**2, axis=1, keepdims=True))
        feats = np.concatenate([sensor_feats, gyro_mag, accel_mag], axis=1)

        out = list(np.mean(feats, axis=0))
        out.extend(np.std(feats, axis=0).tolist())

        # RMS — signal energy independent of direction
        out.extend(np.sqrt(np.mean(feats ** 2, axis=0)).tolist())

        # Zero-crossing rate — oscillation frequency (soli oscillates fast, t-arm doesn't)
        for c in range(8):
            col = feats[:, c]
            centered = col - np.mean(col)
            if len(centered) < 2:
                out.append(0.0)
            else:
                crossings = np.sum((centered[:-1] * centered[1:]) < 0)
                out.append(float(crossings) / len(col))

        out.extend((feats[1:, 0:3] - feats[:-1, 0:3]).flatten().tolist())
        out.extend((feats[1:, 3:6] - feats[:-1, 3:6]).flatten().tolist())
        out.extend((feats[1:, 6:8] - feats[:-1, 6:8]).flatten().tolist())

        # ── Spectral features (FFT-based) + Correlation features (v3+ models) ──
        # These are only computed when the loaded model was trained with them.
        if _use_extra_features:
            for c in range(6):
                col = feats[:, c] - np.mean(feats[:, c])  # center
                n_fft = len(col)
                if n_fft < 3:
                    out.extend([0.0, 0.0, 0.0, 0.0])
                    continue
                fft_vals = np.fft.fft(col)
                half = n_fft // 2
                fft_mag = np.abs(fft_vals[:half])
                freqs = np.fft.fftfreq(n_fft)[:half]
                fft_sum = np.sum(fft_mag) + 1e-10

                domfreq = float(freqs[np.argmax(fft_mag)]) if len(fft_mag) > 0 else 0.0
                energy = float(np.sum(fft_mag ** 2))
                power_norm = fft_mag / fft_sum
                spectral_entropy = float(-np.sum(power_norm * np.log2(power_norm + 1e-10)))
                centroid = float(np.sum(freqs * fft_mag) / fft_sum)

                out.extend([domfreq, energy, spectral_entropy, centroid])

            if len(feats) > 1:
                corr_ax_gx = float(np.corrcoef(feats[:, 0], feats[:, 3])[0, 1])
                corr_ay_gy = float(np.corrcoef(feats[:, 1], feats[:, 4])[0, 1])
                corr_az_gz = float(np.corrcoef(feats[:, 2], feats[:, 5])[0, 1])
                corr_amag_gmag = float(np.corrcoef(feats[:, 7], feats[:, 6])[0, 1])
            else:
                corr_ax_gx = corr_ay_gy = corr_az_gz = corr_amag_gmag = 0.0

            out.extend([corr_ax_gx, corr_ay_gy, corr_az_gz, corr_amag_gmag])

        return out
    else:
        out = list(np.mean(sensor_feats, axis=0))
        out.extend(np.std(sensor_feats, axis=0).tolist())
        cents = []
        for r in readings:
            data = r.data
            pts = data.get("points", [])
            if pts:
                cents.append((np.mean([p.get("x", 0) for p in pts]),
                              np.mean([p.get("y", 0) for p in pts])))
            else:
                cents.append((0.0, 0.0))
        dist = 0.0
        for i in range(1, len(cents)):
            dist += np.sqrt((cents[i][0] - cents[i-1][0])**2 + (cents[i][1] - cents[i-1][1])**2)
        out.append(float(dist))
        return out


def _predict(pipeline, gestures, features):
    X = np.array(features).reshape(1, -1)
    pred = pipeline.predict(X)[0]
    label = gestures[pred] if pred < len(gestures) else str(pred)
    conf = 0.0
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X)[0]
        conf = float(max(proba))
    return label, conf


def _compute_expected_feature_count(sensor_types: list[str], window: int) -> int:
    """Calculate how many features the pipeline will produce for given sensors & window."""
    total = 0
    for stype in sensor_types:
        if stype == "imu":
            # 8 means + 8 stds + 8 RMS + 8 ZCR + 8×(W-1) deltas
            # + 24 spectral + 4 correlation (only for v3+ models)
            base = 60 if _use_extra_features else 32
            total += base + 8 * (window - 1)
        elif stype == "mmwave":
            # 6 per-frame + 6 stds + 1 path_length
            total += 6 + 6 + 1
        elif stype == "uwb":
            # 6 per-frame means (no std/path)
            total += 6 + 6
    return total


def _check_feature_dims(n_computed: int, n_expected: int, sensor_names: list[str]) -> str | None:
    if n_computed == n_expected:
        return None
    # IMU: base + 8×(W-1) where base=32 (standard) or 60 (with spectral/corr)
    base = 60 if _use_extra_features else 32
    approx_w = (n_computed - base) // 8 + 1 if n_computed >= base else 0
    return (
        f"Feature mismatch: computed {n_computed} features, but model expects {n_expected}. "
        f"Check --window (~{approx_w} for IMU, or use different sensor)"
    )


def _check_sustained_oscillation(
    readings: list,
    min_crossings: int = 3,
    min_channels: int = 3,
) -> tuple[bool, float]:
    """Detect sustained multi-cycle oscillation in the raw IMU signal.

    Repetitive gestures (making-fist-open, clapping) produce a DISTINCTIVE
    pattern in the raw IMU: the signal oscillates back and forth over
    multiple cycles. Transient gestures (push, pull, left, right) produce
    a single spike that decays — no sustained oscillation.

    The check works across all 6 IMU channels (ax, ay, az, gx, gy, gz) and
    counts how many show **multiple zero crossings** at significant amplitude.
    This catches the physical reality: making a fist repeatedly means the
    hand opens and closes, which repeatedly reverses the acceleration and
    rotation direction.

    The ``min_crossings`` threshold controls per-channel sensitivity.
    3 crossings means the signal must reverse direction at least 3 times
    within the window — that's ~1.5+ full cycles, impossible for a single
    push/pull/left/right but routine for fist open/close. Lower to 2 for
    more sensitivity.

    ``min_channels`` controls how many of the 6 IMU channels need to
    oscillate. 3 = half the channels, catching the fact that fist
    open/close involves both linear acceleration and rotation.

    Args:
        readings: List of IMU sensor readings from the current window.
        min_crossings: Minimum zero crossings per channel to count as
                       oscillating (default: 3 ≈ 1.5+ cycles).
        min_channels: Minimum channels that must show oscillation
                      (default: 3 of 6).

    Returns:
        ``(is_oscillating, channel_fraction)``.
        ``is_oscillating`` is True when enough IMU channels show
        sustained oscillation.
        ``channel_fraction`` is the fraction (0-1) of channels that
        oscillated (useful for debugging or meters).
    """
    if len(readings) < 6:
        return False, 0.0

    n = len(readings)
    signal = np.zeros((n, 6))

    for i, r in enumerate(readings):
        accel = r.data.get("accel", [0, 0, 0])
        gyro = r.data.get("gyro", [0, 0, 0])
        signal[i, 0] = float(accel[0]) * _accel_gain if len(accel) > 0 else 0.0
        signal[i, 1] = float(accel[1]) * _accel_gain if len(accel) > 1 else 0.0
        signal[i, 2] = float(accel[2]) * _accel_gain if len(accel) > 2 else 0.0
        gx = float(gyro[0]) * _gyro_gain if len(gyro) > 0 else 0.0
        gy = float(gyro[1]) * _gyro_gain if len(gyro) > 1 else 0.0
        gz = float(gyro[2]) * _gyro_gain if len(gyro) > 2 else 0.0
        signal[i, 3] = gx if abs(gx) >= _gyro_deadband else 0.0
        signal[i, 4] = gy if abs(gy) >= _gyro_deadband else 0.0
        signal[i, 5] = gz if abs(gz) >= _gyro_deadband else 0.0

    oscillating = 0
    for c in range(6):
        col = signal[:, c]
        rms = float(np.sqrt(np.mean(col ** 2)))

        # Different thresholds for accel (m/s²) vs gyro (dps)
        min_rms = 0.3 if c < 3 else 0.5
        if rms < min_rms:
            continue

        # Count zero crossings — each crossing means a direction reversal
        centered = col - np.mean(col)
        crossings = int(np.sum((centered[:-1] * centered[1:]) < 0))

        # 3+ crossings = 1.5+ cycles = sustained oscillation
        # 0-1 crossings = transient signal (push, pull, etc.)
        if crossings >= min_crossings:
            oscillating += 1

    fraction = oscillating / 6.0
    return oscillating >= min_channels, fraction


def _fist_oscillation(readings: list) -> bool:
    """Open/close fist signature on a recorded segment.

    Making a fist moves the accelerometer SIDEWAYS (open/close → accel x/z
    zero crossings) while the wrist's vertical axis stays put (NO accel-y
    crossings), and the wrist rotation oscillates 2+ gyro channels above
    real amplitude. Vertical arm motions (t-arm, raise-arms) and wrist
    rolls (clockwise, anticlockwise) either cross accel-y or leave the
    accel still; soli/finger rub has no accel crossings at all; boxing is
    handled by the punch count; bye-bye/clapping are blocklisted. So this
    fires only for the repeated open/close pattern.
    """
    if len(readings) < 6:
        return False
    n = len(readings)
    sig = np.zeros((n, 6))
    for i, r in enumerate(readings):
        accel = r.data.get("accel", [0, 0, 0])
        gyro = r.data.get("gyro", [0, 0, 0])
        for j in range(3):
            sig[i, j] = float(accel[j]) * _accel_gain if len(accel) > j else 0.0
            sig[i, 3 + j] = float(gyro[j]) * _gyro_gain if len(gyro) > j else 0.0

    crossings = [0] * 6
    for ch in range(6):
        centered = sig[:, ch] - np.mean(sig[:, ch])
        deadband = 0.3 if ch < 3 else 3.0  # m/s² vs dps (breathing is too weak to repeat)
        signs = np.sign(centered) * (np.abs(centered) > deadband)
        prev = 0
        for v in signs:
            if v != 0:
                if prev != 0 and v != prev:
                    crossings[ch] += 1
                prev = v
    accel_xz = crossings[0] + crossings[2]  # sideways open/close motion
    accel_y = crossings[1]                  # vertical arm motion
    gyro_channels = sum(1 for ch in range(3, 6) if crossings[ch] >= 3)
    return accel_xz >= 1 and accel_y == 0 and gyro_channels >= 2


def _push_pull_escape(readings: list) -> bool:
    """Rescue the pull/push mislabel.

    The model often calls a repeated open/close fist 'pull' (or 'push') —
    both are single-impulse linear gestures. A real push/pull is ONE
    smooth impulse (at most a tiny settle bounce); a fist repeats, so 2+
    gyro channels with several reversals above 2 dps is a fist, not a pull.
    """
    if len(readings) < 6:
        return False
    n = len(readings)
    sig = np.zeros((n, 3))
    for i, r in enumerate(readings):
        gyro = r.data.get("gyro", [0, 0, 0])
        for j in range(3):
            sig[i, j] = float(gyro[j]) * _gyro_gain if len(gyro) > j else 0.0
    crossings = [0] * 3
    for ch in range(3):
        centered = sig[:, ch] - np.mean(sig[:, ch])
        signs = np.sign(centered) * (np.abs(centered) > 3.0)
        prev = 0
        for v in signs:
            if v != 0:
                if prev != 0 and v != prev:
                    crossings[ch] += 1
                prev = v
    return sum(1 for ch in range(3) if crossings[ch] >= 3) >= 2


def _count_fist_cycles(readings: list, deadband_dps: float = 2.5,
                       max_gap_frames: int = 6) -> int:
    """Maximum number of CONSECUTIVE open/close cycles in the segment.

    One open/close = two direction reversals above the deadband. 'Consecutive'
    means the reversals are tightly packed — no gap longer than
    ``max_gap_frames`` frames between them. A long slow drag (e.g. a
    multi-second slow push/pull) that crosses the deadband occasionally can
    accumulate crossings overall but never as a tight run, so it stays at 0-1
    consecutive cycles; only a real, repeated open/close produces 4+. Counts on
    the strongest gyro channel (the axis a fist rotates on). Recomputing from
    the frozen segment is deterministic (unlike the incremental counter, whose
    single-frame channel lock can miss) and drives the ``× N`` display.
    """
    if len(readings) < 2:
        return 0
    g = np.zeros((len(readings), 3))
    for i, r in enumerate(readings):
        gyro = r.data.get("gyro", [0, 0, 0])
        for j in range(3):
            g[i, j] = float(gyro[j]) * _gyro_gain if len(gyro) > j else 0.0
    rms = np.sqrt(np.mean(g ** 2, axis=0))
    ch = int(np.argmax(rms))
    # frame indices where the strongest channel flips direction across the deadband
    idx = []
    last_sign = 0
    for i, v in enumerate(g[:, ch]):
        sign = 1 if v > deadband_dps else (-1 if v < -deadband_dps else 0)
        if sign != 0:
            if last_sign != 0 and sign != last_sign:
                idx.append(i)
            last_sign = sign
    if len(idx) < 2:
        return 0
    # longest run of reversals with no gap longer than max_gap_frames
    best = 1
    run = 1
    for a, b in zip(idx, idx[1:]):
        if b - a <= max_gap_frames:
            run += 1
            if run > best:
                best = run
        else:
            run = 1
    return best // 2


def _repeat_label(gesture: str, count: int) -> str:
    """Upper-case display label with a repetition count, e.g.
    ``one-arm-boxing`` -> ``ONE ARM BOXING  × 5`` (plain when count == 0)."""
    name = gesture.upper().replace("-", " ")
    if count > 0:
        return f"{name}  × {count}"
    return name


class CycleCounter:
    """Counts repetitions of oscillatory gestures (open/close fist).

    One full open/close is two direction reversals, so it locks onto the
    strongest gyro channel when oscillation starts and counts every sign
    change (with a deadband so resting noise isn't counted). Runs while
    oscillation is sustained; ``reset()`` clears it for the next session.
    """

    def __init__(self):
        self.channel = -1  # locked gyro channel index (0..2), -1 = not locked
        self.last_sign = 0
        self.crossings = 0

    def start(self, window_readings: list) -> None:
        g = [[0.0, 0.0, 0.0] for _ in range(len(window_readings))]
        for i, r in enumerate(window_readings):
            gyro = r.data.get("gyro", [0, 0, 0]) if hasattr(r, "data") else [0, 0, 0]
            g[i][0] = float(gyro[0]) if len(gyro) > 0 else 0.0
            g[i][1] = float(gyro[1]) if len(gyro) > 1 else 0.0
            g[i][2] = float(gyro[2]) if len(gyro) > 2 else 0.0
        arr = np.asarray(g, dtype=float)
        rms = np.sqrt(np.mean(arr ** 2, axis=0))
        self.channel = int(np.argmax(rms)) if len(rms) == 3 else 0
        self.last_sign = 0
        self.crossings = 0

    def update(self, current_gyro: list | None) -> int:
        """Feed the current frame's gyro; returns open/close cycles so far."""
        if self.channel < 0 or not current_gyro or len(current_gyro) <= self.channel:
            return self.crossings // 2
        v = float(current_gyro[self.channel])
        sign = 1 if v > 5.0 else (-1 if v < -5.0 else 0)  # 5 dps deadband
        if sign != 0:
            if self.last_sign != 0 and sign != self.last_sign:
                self.crossings += 1
            self.last_sign = sign
        return self.crossings // 2

    def cycles(self) -> int:
        return self.crossings // 2

    def reset(self) -> None:
        self.channel = -1
        self.last_sign = 0
        self.crossings = 0


class ConfirmedGestureDetector:
    """Buffer-then-confirm detector: decide a gesture ONLY once the user stops.

    This exploits the trial structure: gestures are executed and then the user
    stops moving. Instead of classifying every frame continuously or gating on
    a movement "onset", it keeps a rolling buffer of the movement:

        1. BUFFER     — every moving frame is appended to the segment. No onset
                        streak is required and nothing is classified yet.
        2. CONFIRM    — the FIRST still frame after movement FREEZES the segment
                        forever. From then on a confirm clock counts EVERY frame
                        — still or wobble — with no pausing and no resetting for
                        wrist jitter. Once it fills (``confirm_seconds`` /
                        ``confirm_frames``), the frozen segment (the movement
                        that led up to the stop) is classified EXACTLY once.
        3. COOLDOWN   — a brief lockout so a single stop can never emit twice.

    Because the segment is frozen the instant stillness starts, post-gesture
    wrist wobble can neither change the prediction nor restart the wait — the
    decision is made once, from the data that led up to the stop. Extra gates
    (confidence, tail-window vote, min/max segment length) reject random motion.
    """

    def __init__(self, args, pipeline, gestures, sensor_types, expected_n_features,
                 gesture_conf_overrides: dict[str, float] | None = None,
                 gesture_movement_overrides: dict[str, float] | None = None):
        self.args = args
        self.pipeline = pipeline
        self.gestures = gestures
        self.sensor_types = sensor_types
        self.expected_n_features = expected_n_features
        self.gesture_conf_overrides = gesture_conf_overrides or {}
        self.gesture_movement_overrides = gesture_movement_overrides or {}

        self.window = args.window
        self.idle_threshold = args.idle_threshold
        self.confirm_frames = args.confirm_frames
        # consecutive moving frames required before recording starts — filters
        # one-frame noise blips so the detector ignores little movements
        self.onset_frames = int(getattr(args, "onset_frames", 0) or 0)
        self._onset_streak = 0
        # time-based stillness confirmation (fps-independent); 0 = fall back to frames
        self.confirm_seconds = float(getattr(args, "confirm_seconds", 0.0) or 0.0)
        # min segment: reject tiny noise blips, but don't demand a full window
        # (gestures at low fps can be shorter than the training window)
        self.min_segment_frames = args.min_segment_frames if args.min_segment_frames > 0 \
            else max(2, self.window // 4)
        self.max_segment_frames = args.max_segment_frames
        self.resume_frames = int(getattr(args, "resume_frames", 3) or 3)
        self.emit_conf = args.emit_conf
        self.vote_windows = args.vote_windows
        self.vote_majority = args.vote_majority
        self.cooldown_frames = args.cooldown_frames
        self.sensors = args.sensors

        self.frame_buffer = deque(maxlen=self.window)
        # rolling history of ALL frames (for building model windows when the
        # gesture segment is shorter than the training window)
        self.raw_tail = deque(maxlen=max(self.window * 4, self.window + 10))
        self.movement_history: deque[float] = deque(maxlen=3)
        self.segment: list[dict] = []
        self.state = "idle"
        self.moving_streak = 0
        self.still_streak = 0
        self.cooldown = 0
        self.emissions = 0
        self.onset_len = 0   # len(raw_tail) at segment start (for padding)
        # snapshot of the frames immediately BEFORE the gesture, taken at onset.
        # raw_tail is a shifting bounded deque, so re-slicing it by index during
        # the confirm window yields the wrong (post-gesture still) frames and
        # makes the frozen prediction flicker — snapshot instead.
        self.pre_onset_frames: list[dict] = []

        # ── per-segment punch + fist repetition counting ──────────────────
        # Punches are accel transients between consecutive frames; fist
        # open/close cycles come from the CycleCounter's gyro-channel lock.
        # Both are surfaced on the emission: boxing type is decided by the
        # punch count (1 punch → two-arm, 2+ → one-arm) and the count is
        # shown alongside the label (e.g. "ONE ARM BOXING  × 5").
        self.punch_total = 0
        self._punch_prev_accel: list[float] | None = None
        self._was_in_punch = False
        self._punch_cooldown = 0
        self.fist_counter = CycleCounter()
        self.last_punch_count = 0    # punch total of the most recent emission
        self.last_fist_cycles = 0    # open/close cycles of the most recent emission
        self.last_emitted_label: str | None = None  # raw model label (pre boxing-type override)
        self.last_override = "model"  # how the last emission got its label: "model" | "osc" | "escape"

        # making-fist-open signal override — ALWAYS on (replaces the model's
        # unreliable fist label). Hardened so it fires only on a real open/close
        # pattern (sideways accel, no accel-y, 2+ gyro channels above 4 dps).
        # --fist-counter only adds the × N repetition counter display.
        self.fist_counter_enabled = bool(getattr(args, "fist_counter", False))
        self.osc_gesture = getattr(args, "oscillation_gesture", "making-fist-open")
        self.osc_block = list(getattr(args, "oscillation_block", ["bye-bye", "clapping"]) or [])

        # ── live diagnostics (surfaced in the under-the-hood GUI panel) ──
        self.last_movement = 0.0      # movement score of the most recent frame
        self.last_moving = False      # whether the last frame counted as 'moving'
        self.last_reason: str | None = None   # why the last segment was rejected/aborted
        self.last_emit: tuple | None = None   # (label, conf, n_agree, delay_frames)
        self.segment_peak = 0.0       # peak movement during the current segment
        self.event_log: deque[str] = deque(maxlen=60)

        # time-based stillness accumulation (fps-independent confirm)
        self.still_time = 0.0
        self._dt = 0.0
        self._last_frame_time: float | None = None

    # ── internals ────────────────────────────────────────────────────
    def _reset(self) -> None:
        self.state = "idle"
        self.segment = []
        self.still_streak = 0
        self.moving_streak = 0
        self.segment_peak = 0.0
        self.still_time = 0.0
        self._onset_streak = 0

    def _log(self, msg: str) -> None:
        """Record an event (GUI log) and mirror it to the terminal under --debug."""
        self.event_log.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        if self.args.debug:
            print(msg)

    def _movement(self, window: list[dict]) -> float:
        """Movement score over the most recent frames (responsive, not window-wide)."""
        raw = 0.0
        for name in self.sensors:
            readings = _gather_readings(window, name)
            recent = readings[-3:] if len(readings) > 3 else readings
            raw += _movement_score(recent, self.sensor_types[name])
        self.movement_history.append(raw)
        return float(np.mean(self.movement_history))

    def _is_moving(self, window: list[dict]) -> bool:
        movement = self._movement(window)
        self.last_movement = float(movement)
        if movement >= self.idle_threshold:
            self.last_moving = True
            return True
        imu_readings = _gather_readings(window, "imu")
        osc = bool(imu_readings and len(imu_readings) >= 5
                   and _check_gyro_oscillation(imu_readings[-5:]))
        self.last_moving = osc
        return osc

    def _resolve_boxing(self, label: str) -> str:
        """Boxing type is decided by the punch count, not the model: a single
        punch is two-arm-boxing, two or more is one-arm-boxing."""
        if label in ("one-arm-boxing", "two-arm-boxing"):
            return "one-arm-boxing" if self.punch_total >= 2 else "two-arm-boxing"
        return label

    def _count_segment_frame(self, frame_data: dict) -> None:
        """Count punches (accel transients) and fist open/close cycles while
        the gesture segment is being recorded, so the emission can show a
        repetition count (boxing → punch total, making-fist-open → cycles)."""
        imu_frame = None
        for name, stype in self.sensor_types.items():
            if stype == "imu" and name in frame_data and frame_data[name]:
                imu_frame = frame_data[name]
                break
        if imu_frame is None or not imu_frame.data:
            return
        accel = imu_frame.data.get("accel")
        gyro = imu_frame.data.get("gyro")
        if accel is not None:
            accel = [float(a) for a in accel]
            if self._punch_prev_accel is not None:
                delta = sum(abs(accel[i] - self._punch_prev_accel[i]) for i in range(3))
                in_punch = delta > getattr(self.args, "punch_threshold", 1.0)
                if self._punch_cooldown > 0:
                    self._punch_cooldown -= 1
                elif in_punch and not self._was_in_punch:
                    self.punch_total += 1
                    self._punch_cooldown = getattr(self.args, "punch_cooldown", 10)
                self._was_in_punch = in_punch
            self._punch_prev_accel = accel[:]
        if gyro is not None:
            if self.fist_counter.channel < 0:
                self.fist_counter.start([imu_frame])
            self.fist_counter.update(list(float(g) for g in gyro))

    # ── main entry: feed one frame ──────────────────────────────────
    def feed(self, frame_data: dict):
        """Feed one frame. Returns None, or an emission tuple
        ``(label, conf, n_agree, delay_frames)`` when a gesture is confirmed."""
        now = time.time()
        self._dt = (now - self._last_frame_time) if self._last_frame_time is not None else 0.0
        self._last_frame_time = now
        self.frame_buffer.append(frame_data)
        self.raw_tail.append(frame_data)
        if len(self.frame_buffer) < 2:
            return None
        window = list(self.frame_buffer)
        moving = self._is_moving(window)

        # cooldown lockout — a single stop can never emit twice
        if self.cooldown > 0:
            self.cooldown -= 1
            if self.cooldown == 0:
                self._reset()
            return None

        # "real" movement = accelerometer transients only. Gyro oscillation
        # (soli/wobble) is enough to BUFFER frames, but never to resume or to
        # reset the confirm clock — a wrist wobble must not pollute the frozen
        # gesture nor restart the wait.
        accel_moving = self.last_movement >= self.idle_threshold

        if moving:
            # movement: buffer the moving frames. The gesture is frozen and
            # decided ONLY once the user stops (see the still branch below).
            if self.state == "idle":
                if self._onset_streak < self.onset_frames:
                    self._onset_streak += 1
                    return None
                self.state = "recording"
                self.segment = [frame_data]
                self.onset_len = len(self.raw_tail) - 1
                self.pre_onset_frames = list(self.raw_tail)[:-1]
                self.segment_peak = self.last_movement
                self.last_reason = None
                self.punch_total = 0
                self._punch_prev_accel = None
                self._was_in_punch = False
                self._punch_cooldown = 0
                self.fist_counter.reset()
                self._count_segment_frame(frame_data)
                return None
            if self.state == "recording":
                self.segment.append(frame_data)
                self._count_segment_frame(frame_data)
                self.segment_peak = max(self.segment_peak, self.last_movement)
                if len(self.segment) > self.max_segment_frames:
                    # too much continuous movement to be one gesture — abort
                    self.last_reason = f"aborted: segment too long ({len(self.segment)}f)"
                    self._log(f"[confirm] {self.last_reason}")
                    self._reset()
                return None
            # state == "confirming": fall through so the confirming branch can
            # run its resume logic on this moving frame
        else:
            if self.state == "recording":
                # freeze the gesture at its last moving frame
                self.state = "confirming"
                self.still_streak = 0
                self.still_time = 0.0
                self.moving_streak = 0
            self._onset_streak = 0

        if self.state == "confirming":
            if accel_moving:
                # genuine gesture continuation — only a SUSTAINED run of real
                # movement resumes buffering (and restarts the wait).
                self.moving_streak += 1
                if self.moving_streak >= self.resume_frames:
                    self.state = "recording"
                    self.segment.append(frame_data)
                    self._count_segment_frame(frame_data)
                    self.segment_peak = max(self.segment_peak, self.last_movement)
                    self.still_streak = 0
                    self.still_time = 0.0
                    self.moving_streak = 0
                    return None
            else:
                self.moving_streak = 0
            # the confirm clock counts every frame — still or gyro-wobble — and
            # never pauses for wrist jitter. Genuine accel movement pauses it
            # (that's the gesture still going, not a stop), and only a SUSTAINED
            # run of it resumes the segment (above).
            if not accel_moving:
                self.still_streak += 1
                self.still_time += self._dt
            if (self.confirm_seconds > 0 and self.still_time >= self.confirm_seconds) or \
               (self.confirm_seconds <= 0 and self.still_streak >= self.confirm_frames):
                return self._emit()
        return None

    # ── classify the recorded segment exactly once ──────────────────
    def _emit(self):
        seg = self.segment
        n = len(seg)
        usable_end = n  # segment holds only MOVING frames; the still tail is no longer included

        if usable_end < self.min_segment_frames:
            self.last_reason = f"rejected: segment {usable_end}f < min {self.min_segment_frames}f"
            self._log(f"[confirm] {self.last_reason}")
            self._reset()
            return None

        votes: list[tuple[str, float]] = []
        # Build windows of exactly ``self.window`` frames ending at the last
        # MOVING frame (usable_end). Segments shorter than the window are padded
        # on the left with the frames that immediately preceded onset (idle),
        # which mirrors how the rolling classifier saw the gesture live.
        for k in range(self.vote_windows):
            end = usable_end - k
            if end <= 0:
                break
            L = end
            pre = self.window - L
            if pre > 0:
                pre_frames = self.pre_onset_frames[-pre:] if self.pre_onset_frames else []
            else:
                pre_frames = []
            win = pre_frames + seg[:end]
            if len(win) < self.window:
                break  # not enough history — treat as no window
            win = win[-self.window:]
            features = []
            for name in self.sensors:
                readings = _gather_readings(win, name)
                features.extend(_compute_features(readings, self.sensor_types[name]))
            try:
                label, conf = _predict(self.pipeline, self.gestures, features)
            except Exception as e:
                if self.args.debug:
                    print(f"  ⚠ prediction error: {e}")
                continue
            votes.append((label, conf))

        if not votes:
            self.last_reason = "rejected: no valid window"
            self._log(f"[confirm] {self.last_reason}")
            self._reset()
            return None

        tally: dict[str, list[float]] = {}
        for label, conf in votes:
            tally.setdefault(label, []).append(conf)
        winner = max(tally, key=lambda lb: (len(tally[lb]), max(tally[lb])))
        n_agree = len(tally[winner])
        majority = self.vote_majority if self.vote_majority > 0 else len(votes)

        if n_agree < majority:
            detail = ", ".join(f"{lb}:{len(c)}" for lb, c in tally.items())
            self.last_reason = f"rejected: votes split ({detail})"
            self._log(f"[confirm] {self.last_reason}")
            self._reset()
            return None

        conf = max(tally[winner])
        min_conf = self.gesture_conf_overrides.get(winner, self.emit_conf)
        if conf < min_conf:
            self.last_reason = f"rejected: {winner} conf={conf:.3f} < {min_conf:.3f}"
            self._log(f"[confirm] {self.last_reason}")
            self._reset()
            return None

        self.emissions += 1
        self.state = "cooldown"
        self.cooldown = self.cooldown_frames
        delay_frames = self.still_streak
        self.last_punch_count = self.punch_total
        self.last_emitted_label = winner
        winner = self._resolve_boxing(winner)
        # ── making-fist-open signal override ─────────────────────────
        # The model often mislabels open/close fist (trained on sparse data),
        # predicting palm-up-down/etc. instead. A repeated open/close is the
        # ONE gesture that moves the accelerometer sideways (accel x/z
        # crossings, no accel-y) AND oscillates 2+ gyro channels above a real
        # amplitude. Breathing/resting noise is too weak to cross the gyro
        # deadband repeatedly, so it no longer fires on every breath. The × N
        # repetition counter is only shown with --fist-counter.
        readings = _gather_readings(seg, "imu")
        self.last_fist_cycles = _count_fist_cycles(readings) if readings \
            else self.fist_counter.cycles()
        self.last_override = "model"
        if winner != self.osc_gesture \
                and winner not in ("one-arm-boxing", "two-arm-boxing") \
                and winner not in self.osc_block:
            if _fist_oscillation(readings) and self.last_fist_cycles >= 4:
                winner = self.osc_gesture
                self.last_override = "osc"
            # The model mislabels a repeated open/close fist as pull/push
            # (both single-impulse linear gestures). A real push/pull is one
            # impulse (even with a hard-stop bounce); a fist repeats — so ≥4
            # CONSECUTIVE cycles + gyro oscillation on 2+ channels is a fist.
            elif winner in ("push", "pull") \
                    and self.last_fist_cycles >= 4 \
                    and _push_pull_escape(readings):
                winner = self.osc_gesture
                self.last_override = "escape"
        self.segment = []
        self.segment_peak = 0.0
        self.still_time = 0.0
        self.still_streak = 0
        self.moving_streak = 0
        self.last_reason = None
        self.last_emit = (winner, conf, n_agree, delay_frames)
        self._log(f"> EMIT {winner}  conf={conf:.3f}  votes={n_agree}/{len(votes)}  +{delay_frames}f")
        return winner, conf, n_agree, delay_frames

    def predict_live(self) -> tuple[str, float] | None:
        """What the model would predict on the CURRENT rolling window.

        This is the raw per-frame prediction the old rolling demo showed
        continuously. In confirmed mode it's advisory only (the state machine
        gates actual emissions), but seeing it live tells you whether the
        model is even registering your gesture before the segment completes.
        Returns None until the rolling buffer is full.
        """
        if len(self.frame_buffer) < self.window:
            return None
        win = list(self.frame_buffer)[-self.window:]
        features = []
        for name in self.sensors:
            readings = _gather_readings(win, name)
            features.extend(_compute_features(readings, self.sensor_types[name]))
        try:
            return _predict(self.pipeline, self.gestures, features)
        except Exception:
            return None

    def predict_frozen(self) -> tuple[str, float] | None:
        """Prediction on the FROZEN segment tail — the answer ``_emit`` will use.

        During the still/confirm window the rolling buffer fills with stillness,
        so ``predict_live`` starts flickering through boxing/raise-arms/palm
        garbage. This predicts on the frozen gesture instead, so the display
        stays stable and matches the eventual emission.
        """
        seg = self.segment
        n = len(seg)
        if n == 0:
            return None
        end = n
        L = end
        pre = self.window - L
        if pre > 0:
            pre_frames = self.pre_onset_frames[-pre:] if self.pre_onset_frames else []
        else:
            pre_frames = []
        win = pre_frames + seg[:end]
        if len(win) < self.window:
            return None
        win = win[-self.window:]
        features = []
        for name in self.sensors:
            readings = _gather_readings(win, name)
            features.extend(_compute_features(readings, self.sensor_types[name]))
        try:
            label, conf = _predict(self.pipeline, self.gestures, features)
            return self._resolve_boxing(label), conf
        except Exception:
            return None


def run_confirmed(args, pipeline, expected_n_features, gestures, reader_map, sensor_types,
                  gesture_conf_overrides: dict[str, float] | None = None,
                  gesture_movement_overrides: dict[str, float] | None = None):
    """Terminal loop for confirmed-gesture mode."""
    global _accel_lp
    _accel_lp = None
    detector = ConfirmedGestureDetector(args, pipeline, gestures, sensor_types,
                                        expected_n_features, gesture_conf_overrides,
                                        gesture_movement_overrides)
    print("\n=== Confirmed-Gesture Mode ===")
    print("Perform a gesture, stop moving, and the result appears")
    if detector.confirm_seconds > 0:
        print(f"~{detector.confirm_seconds:.1f} s of stillness after you stop.\n")
    else:
        print(f"~{detector.confirm_frames} frames after you stop (the 1-2 s confirmation window).\n")
    if (_accel_gain, _gyro_gain, _gyro_deadband) != (1.0, 1.0, 0.0):
        print("  💡 Tip: the model was trained on raw accel/gyro. For maximum confidence")
        print("     use --accel-gain 1.0 --gyro-gain 1.0 --gyro-deadband 0.0")
        print()
    print("Waiting for gesture...")
    try:
        while True:
            frame_data = {name: reader.read() for name, reader in reader_map.items()}
            emitted = detector.feed(frame_data)
            if emitted is not None:
                label, conf, n_agree, delay = emitted
                if label == getattr(detector.args, "oscillation_gesture", "making-fist-open") \
                        and detector.fist_counter_enabled:
                    label = _repeat_label(label, detector.last_fist_cycles)
                via = f"  [via {detector.last_override}]" \
                    if label.startswith("MAKING FIST OPEN") \
                    else f"  [model={detector.last_emitted_label}]"
                print(f"> {label}  (conf={conf:.2f}, votes={n_agree}/{detector.vote_windows}, "
                      f"+{delay}f after gesture){via}")
            time.sleep(0.02)
    except KeyboardInterrupt:
        print(f"\n\nDemo stopped. {detector.emissions} gesture(s) emitted.")
    finally:
        for reader in reader_map.values():
            reader.stop()


def run_confirmed_gui(args, pipeline, expected_n_features, gestures, reader_map, sensor_types,
                      gesture_conf_overrides: dict[str, float] | None = None,
                      gesture_movement_overrides: dict[str, float] | None = None):
    """Under-the-hood GUI for confirmed-gesture mode.

    Big label on top (what was emitted), plus a live diagnostics panel that
    shows everything the detector is doing in real time:

      • movement score vs the idle threshold (the onset trigger)
      • the model's raw per-frame prediction (what it sees RIGHT NOW)
      • state-machine internals: onset streak, segment length, peak movement,
        still/confirm streak, cooldown countdown
      • every transition, rejection and emission in a scrolling event log

    If a gesture isn't being picked up you can SEE why: movement stuck above
    the threshold (never confirms), prediction below the per-gesture conf gate
    (rejected), or the model seeing something else entirely.
    """
    try:
        import tkinter as tk
    except ImportError:
        print("--gui requires tkinter (install python-tk)")
        return

    global _accel_lp
    _accel_lp = None
    detector = ConfirmedGestureDetector(args, pipeline, gestures, sensor_types,
                                        expected_n_features, gesture_conf_overrides,
                                        gesture_movement_overrides)

    PALETTE = {
        "bg": "#0d0d10",
        "panel": "#141419",
        "line": "#2a2a32",
        "fg": "#d4d4dc",
        "dim": "#6a6a72",
        "muted": "#4a4a52",
        "accent": "#4a7cbf",
        "ok": "#5a9a6a",
        "warn": "#b89a3a",
        "log": "#9a9aa2",
    }
    FONTS = {
        "big": ("Helvetica", 80, "bold"),
        "title": ("Helvetica", 17, "bold"),
        "mono": ("Courier", 17),
        "mono_small": ("Courier", 15),
        "mono_tiny": ("Courier", 14),
    }

    root = tk.Tk()
    root.title("Gesture Recognition")
    root.geometry("760x800")
    root.configure(bg=PALETTE["bg"])
    root.minsize(560, 680)

    tk.Label(root, text="CONFIRMED-GESTURE MODE", font=FONTS["title"],
             fg=PALETTE["accent"], bg=PALETTE["bg"]).pack(pady=(12, 0))
    tk.Label(root, text="do the gesture → stop & hold still ~1s → result appears",
             font=FONTS["mono_tiny"], fg=PALETTE["muted"], bg=PALETTE["bg"]).pack(pady=(2, 0))
    status = tk.Label(root, text="○ waiting for a gesture…", font=("Courier", 20),
                      fg=PALETTE["dim"], bg=PALETTE["bg"])
    status.pack(pady=(2, 0))
    gesture_label = tk.Label(root, text="—", font=FONTS["big"],
                             fg="#ffffff", bg=PALETTE["bg"])
    gesture_label.pack(expand=True, fill="both", pady=(2, 0))

    # ── under-the-hood panel ────────────────────────────────────────
    panel = tk.Frame(root, bg=PALETTE["panel"], highlightthickness=1,
                     highlightbackground=PALETTE["line"])
    panel.pack(fill="x", padx=14, pady=(0, 4))

    # movement meter (the onset trigger)
    mv_row = tk.Frame(panel, bg=PALETTE["panel"])
    mv_row.pack(fill="x", padx=10, pady=(8, 2))
    tk.Label(mv_row, text="movement  (onset trigger)", font=FONTS["mono_small"],
             fg=PALETTE["muted"], bg=PALETTE["panel"], anchor="w").pack(side="left")
    move_val = tk.Label(mv_row, text="0.0000", font=FONTS["mono"],
                        fg=PALETTE["fg"], bg=PALETTE["panel"], anchor="e")
    move_val.pack(side="right")
    move_bar = tk.Canvas(panel, height=8, bg=PALETTE["bg"], highlightthickness=1,
                         highlightbackground=PALETTE["line"])
    move_bar.pack(fill="x", padx=10, pady=(0, 2))
    moving_lbl = tk.Label(panel, text="○ still", font=FONTS["mono_tiny"],
                          fg=PALETTE["dim"], bg=PALETTE["panel"], anchor="w")
    moving_lbl.pack(fill="x", padx=10, pady=(0, 2))

    # raw per-frame model prediction
    model_label = tk.Label(panel, text="model sees: — (warming up 0/20)",
                           font=FONTS["mono_small"], fg=PALETTE["dim"],
                           bg=PALETTE["panel"], anchor="w")
    model_label.pack(fill="x", padx=10, pady=(1, 2))

    # state-machine internals
    state_line = tk.Label(panel, text="IDLE · onset 0/3 · still 0/40 · seg 0f · peak 0.000",
                          font=FONTS["mono_small"], fg=PALETTE["dim"],
                          bg=PALETTE["panel"], anchor="w")
    state_line.pack(fill="x", padx=10, pady=(1, 2))

    # last rejection / abort reason
    reason = tk.Label(panel, text="—", font=FONTS["mono_small"], fg=PALETTE["muted"],
                      bg=PALETTE["panel"], anchor="w", wraplength=700, justify="left")
    reason.pack(fill="x", padx=10, pady=(1, 2))

    # event log
    tk.Label(panel, text="EVENT LOG — every transition, rejection & emission",
             font=FONTS["mono_tiny"], fg=PALETTE["muted"], bg=PALETTE["panel"],
             anchor="w").pack(fill="x", padx=10, pady=(6, 0))
    log_label = tk.Label(panel, text="", font=FONTS["mono_tiny"], fg=PALETTE["log"],
                         bg=PALETTE["bg"], anchor="w", justify="left", wraplength=700)
    log_label.pack(fill="x", padx=10, pady=(2, 8))

    info = tk.Label(root, text="emissions: 0", font=FONTS["mono_small"],
                    fg=PALETTE["dim"], bg=PALETTE["bg"])
    info.pack(pady=(0, 10))

    print("Under-the-hood GUI:\n"
          "  movement meter  → the onset trigger (idle threshold)\n"
          "  model sees:     → raw per-frame prediction (advisory only)\n"
          "  state line      → state machine internals (onset/still streaks)\n"
          "  event log       → why segments were rejected/emitted")

    running = [True]

    def close():
        running[0] = False
        root.destroy()

    def _draw_bar(canvas: tk.Canvas, fraction: float, color: str) -> None:
        w = canvas.winfo_width()
        if w < 2:
            return
        canvas.delete("all")
        bar_w = max(1, int(w * fraction))
        canvas.create_rectangle(0, 0, bar_w, 10, fill=color, outline="")
        canvas.create_rectangle(bar_w, 0, w, 10, fill=PALETTE["bg"], outline="")

    def poll():
        if not running[0]:
            return
        try:
            frame_data = {name: reader.read() for name, reader in reader_map.items()}
            emitted = detector.feed(frame_data)

            # movement meter
            mv = detector.last_movement
            move_val.config(
                text=f"{mv:.4f}  (idle τ={detector.idle_threshold:.2f})")
            _draw_bar(move_bar, min(mv / max(detector.idle_threshold * 3, 0.01), 1.0),
                      PALETTE["ok"] if detector.last_moving else PALETTE["muted"])
            if detector.last_moving:
                moving_lbl.config(text="● moving", fg=PALETTE["ok"])
            else:
                moving_lbl.config(text="○ still (confirming)", fg=PALETTE["dim"])

            # model prediction — live window while moving, frozen gesture tail
            # while confirming (so it doesn't flicker through garbage on the
            # still frames); hidden otherwise
            if detector.state == "recording":
                live = detector.predict_live()
            elif detector.state == "confirming":
                live = detector.predict_frozen()
            else:
                live = None
            if live is None:
                model_label.config(text="model sees: —", fg=PALETTE["dim"])
            else:
                lbl, conf = live
                gate = detector.gesture_conf_overrides.get(lbl, detector.emit_conf)
                model_label.config(
                    text=f"model sees: {lbl}  (conf={conf:.2f}, gate={gate:.2f})",
                    fg=PALETTE["ok"] if conf >= gate else PALETTE["warn"])

            # state-machine internals
            st = detector.state
            if st == "recording":
                state_line.config(
                    text=f"BUFFERING · seg {len(detector.segment)}f · peak {detector.segment_peak:.3f}",
                    fg=PALETTE["warn"])
            elif st == "confirming":
                if detector.confirm_seconds > 0:
                    still_txt = f"still {detector.still_time:.1f}s/{detector.confirm_seconds:.1f}s"
                else:
                    still_txt = f"still {detector.still_streak}/{detector.confirm_frames}"
                state_line.config(
                    text=f"CONFIRMING · seg frozen {len(detector.segment)}f · {still_txt}",
                    fg=PALETTE["warn"])
            elif st == "cooldown":
                state_line.config(text=f"COOLDOWN · {detector.cooldown}f lockout", fg=PALETTE["ok"])
            else:
                state_line.config(
                    text=f"IDLE · still {detector.still_streak} · cooldown {detector.cooldown}",
                    fg=PALETTE["dim"])

            # rejection / abort reason
            if detector.last_reason:
                reason.config(text=f"⚠ {detector.last_reason}", fg=PALETTE["warn"])
            else:
                reason.config(text="—", fg=PALETTE["muted"])

            # event log (tail)
            log_label.config(text="\n".join(list(detector.event_log)[-12:]) or "—")

            # status line
            if st == "recording":
                status.config(text=f"● moving ({len(detector.segment)}f) — stop when done",
                              fg=PALETTE["warn"])
            elif st == "confirming":
                if detector.confirm_seconds > 0:
                    hold = f"{detector.still_time:.1f}s/{detector.confirm_seconds:.1f}s"
                else:
                    hold = f"{detector.still_streak}/{detector.confirm_frames}"
                status.config(text=f"○ stop & hold… {hold}", fg=PALETTE["warn"])
            elif st == "cooldown":
                status.config(text="✓ gesture locked", fg=PALETTE["ok"])
            else:
                status.config(text="○ waiting for a gesture…", fg=PALETTE["dim"])

            if emitted is not None:
                label, conf, n_agree, delay = emitted
                if label == getattr(detector.args, "oscillation_gesture", "making-fist-open") \
                        and detector.fist_counter_enabled:
                    display = _repeat_label(label, detector.last_fist_cycles)
                else:
                    display = label.upper().replace("-", "\n")
                gesture_label.config(text=display)
                info.config(text=f"emissions: {detector.emissions}   last: {display}  conf={conf:.2f}  "
                                 f"votes={n_agree}/{detector.vote_windows}  +{delay}f")
                via = f"  [via {detector.last_override}]" \
                    if display.startswith("MAKING FIST OPEN") \
                    else f"  [model={detector.last_emitted_label}]"
                print(f"> {display}  (conf={conf:.2f}, votes={n_agree}/{detector.vote_windows}, +{delay}f){via}")
                root.after(1500, lambda: gesture_label.config(text="—"))
        except Exception as e:
            status.config(text=f"⚠ {e}", fg=PALETTE["warn"])
        root.after(25, poll)

    root.protocol("WM_DELETE_WINDOW", close)
    root.after(100, poll)
    try:
        root.mainloop()
    finally:
        for reader in reader_map.values():
            reader.stop()


def run_terminal(args, pipeline, expected_n_features, gestures, reader_map, sensor_types,
                 gesture_conf_overrides: dict[str, float] | None = None,
                 gesture_movement_overrides: dict[str, float] | None = None):
    if gesture_conf_overrides is None:
        gesture_conf_overrides = {}
    if gesture_movement_overrides is None:
        gesture_movement_overrides = {}
    frame_buffer = deque(maxlen=args.window)
    smooth_buffer: deque[str] = deque(maxlen=args.smooth)
    frame_count = 0
    displayed = None
    challenge_label: str | None = None
    challenge_count = 0
    hold_counter = 0
    max_hold_frames = 3
    movement_history: deque[float] = deque(maxlen=3)
    min_display_frames = 5
    display_age = 0
    # Observation mode state (boxing type discrimination)
    observe_until: int | None = None
    punch_count = 0
    prev_accel: list[float] | None = None
    was_in_punch = False
    punch_cooldown = 0
    boxing_type_locked: str | None = None  # once set, other boxing type can't replace
    # Cooldown after idle clear — prevents flash of stale predictions
    idle_cooldown = 0
    # Oscillation persistence counter — requires ~2 seconds of sustained
    # oscillation before overriding the prediction. Resets if oscillation stops.
    osc_persistence = 0
    punch_total = 0  # punches counted in the current boxing session (kept while boxing stays up)
    fist_counter = CycleCounter()  # open/close fist repetition counter

    # Reset gravity-tracking HPF for a fresh session
    global _accel_lp
    _accel_lp = None

    print("\n=== Real-time Demo Started ===")
    print("Waiting for gestures...\n")

    try:
        while True:
            frame_data = {name: reader.read() for name, reader in reader_map.items()}
            frame_count += 1
            frame_buffer.append(frame_data)
            window = list(frame_buffer)

            if len(window) < args.window:
                continue

            current_accel: list[float] | None = None
            for name, stype in sensor_types.items():
                if stype == "imu" and name in frame_data:
                    reading = frame_data[name]
                    if reading and reading.data:
                        accel = reading.data.get("accel", None)
                        if accel is not None:
                            current_accel = list(map(float, accel))
                            break

            raw_movement = 0.0
            features = []
            for name in args.sensors:
                readings = _gather_readings(window, name)
                raw_movement += _movement_score(readings, sensor_types[name])
                features.extend(_compute_features(readings, sensor_types[name]))

            if frame_count == args.window:
                err = _check_feature_dims(len(features), expected_n_features, args.sensors)
                if err:
                    print(f"  ⚠ {err}")

            movement_history.append(raw_movement)
            movement = np.mean(movement_history)

            is_idle = movement < args.idle_threshold

            # ── gyro oscillation override ──────────────────────────────
            # Soli (finger rub) produces minimal wrist accel but distinctive
            # oscillatory gyro.  Override idle so the model gets to classify it.
            if is_idle:
                imu_readings = _gather_readings(window, "imu")
                if imu_readings and _check_gyro_oscillation(imu_readings):
                    is_idle = False

            # ── boxing: count punches ──────────────────────────────
            # Counts while the type is being determined (observation) AND
            # while boxing stays on screen, so the running punch count is
            # always visible.
            boxing_active = observe_until is not None or displayed in ("one-arm-boxing", "two-arm-boxing")
            if boxing_active and current_accel is not None and prev_accel is not None:
                delta = sum(abs(current_accel[i] - prev_accel[i]) for i in range(3))
                in_punch = delta > args.punch_threshold
                if args.debug:
                    print(f"  [obs] punch={punch_total}, delta={delta:.3f} (th={args.punch_threshold}){' ⚡' if in_punch else ''}")
                if punch_cooldown > 0:
                    punch_cooldown -= 1
                elif in_punch and not was_in_punch:
                    punch_total += 1
                    punch_cooldown = args.punch_cooldown
                    if displayed in ("one-arm-boxing", "two-arm-boxing"):
                        print(f"  [obs] → {displayed}")
                    was_in_punch = in_punch

            if current_accel is not None:
                prev_accel = current_accel[:]

            if observe_until is not None:
                if frame_count >= observe_until:
                    if punch_total >= 2:
                        displayed = "one-arm-boxing"
                    else:
                        displayed = "two-arm-boxing"
                    boxing_type_locked = displayed  # lock this boxing type
                    print(f"> {displayed}")
                    display_age = 0
                    challenge_count = 0
                    challenge_label = None
                    hold_counter = max_hold_frames
                    # stop observing but KEEP counting punches while boxing stays up
                    observe_until = None
                    was_in_punch = False
                    punch_cooldown = 0
                    time.sleep(0.02)
                    continue
                else:
                    if args.debug:
                        print(f"  [obs] awaiting — punch={punch_total}")
                    time.sleep(0.02)
                    continue

            # ── idle ──
            if is_idle:
                if displayed is not None:
                    if hold_counter > 0:
                        hold_counter -= 1
                        display_age += 1
                    else:
                        print("  -> idle")
                        displayed = None
                        smooth_buffer.clear()
                        movement_history.clear()
                        challenge_count = 0
                        challenge_label = None
                        observe_until = None
                        punch_count = 0
                        prev_accel = None
                        was_in_punch = False
                        punch_cooldown = 0
                        boxing_type_locked = None
                        osc_persistence = 0
                        punch_total = 0
                        fist_counter.reset()
                        idle_cooldown = 15  # suppress predictions briefly to prevent flash
                else:
                    smooth_buffer.clear()
                    challenge_count = 0
                    challenge_label = None
                    observe_until = None
                    punch_count = 0
                    prev_accel = None
                    was_in_punch = False
                    punch_cooldown = 0
                    boxing_type_locked = None
                    osc_persistence = 0
                    punch_total = 0
                    fist_counter.reset()
                continue

            # hold_counter is set to max_hold_frames when a new gesture is displayed
            # (in the challenge acceptance section below). Do NOT reset it here — that
            # would make displayed gestures stick forever while movement stays active.

            # ── idle cooldown ─────────────────────────────────────────
            # After clearing a gesture, briefly suppress predictions to prevent
            # a flash from transient movement noise (the buffer was just cleared,
            # so the model sees an empty window and may produce a spurious prediction).
            if idle_cooldown > 0:
                idle_cooldown -= 1
                # Cancel cooldown early if significant new movement starts
                if movement > args.idle_threshold * 2.0:
                    idle_cooldown = 0
                else:
                    time.sleep(0.02)
                    continue

            # ── sustained oscillation detection (signal-based) ──
            # Check the raw IMU signal for multi-cycle oscillation.
            # Requires ~2 seconds of sustained oscillation before overriding
            # the model — prevents false positives from brief wrist jitter.
            imu_readings = _gather_readings(window, "imu")
            sustained_osc, osc_frac = False, 0.0
            if imu_readings and len(imu_readings) >= 6:
                sustained_osc, osc_frac = _check_sustained_oscillation(
                    imu_readings,
                    args.oscillation_min_crossings,
                    args.oscillation_min_channels,
                )

            # Track persistence: oscillation must hold for ~2 seconds
            if sustained_osc:
                osc_persistence += 1
            else:
                osc_persistence = 0

            # ── open/close fist cycle counter ─────────────────────
            # While oscillation is sustained, feed the current frame's gyro
            # into the counter (locks onto the strongest channel at start).
            if sustained_osc:
                if fist_counter.channel < 0:
                    fist_counter.start(imu_readings)
                cur_gyro = None
                for name, stype in sensor_types.items():
                    if stype == "imu" and name in frame_data and frame_data[name]:
                        cur_gyro = frame_data[name].data.get("gyro")
                        break
                fist_counter.update(cur_gyro)
            else:
                fist_counter.reset()

            # ── oscillation sustained — check displayed gesture first ──
            # By the time oscillation has persisted for ~1s, the model may no
            # longer confidently predict the original gesture (the window now
            # contains the oscillating signal). Check what's ALREADY displayed
            # — if it's a blocked gesture, don't override it.
            if osc_persistence >= args.oscillation_min_frames:
                if displayed in args.oscillation_block:
                    # Already showing a blocked gesture — don't override
                    try:
                        label, conf = _predict(pipeline, gestures, features)
                    except Exception as e:
                        print(f"  ⚠ Prediction error: {e}")
                        time.sleep(0.02)
                        continue
                    if args.debug:
                        print(f"  [osc] protected — {displayed} is in blocklist")
                else:
                    try:
                        model_label, model_conf = _predict(pipeline, gestures, features)
                    except Exception as e:
                        model_label, model_conf = None, 0.0
                    if model_label in args.oscillation_block and model_conf >= args.min_conf:
                        label = model_label
                        conf = model_conf
                        if args.debug:
                            print(f"  [osc] blocked — {label} ({conf:.2f}) accepted")
                    else:
                        label = args.oscillation_gesture
                        conf = 0.95
                        if args.debug:
                            print(f"  [osc] {osc_persistence}f sustained → {label}")
            else:
                if sustained_osc and args.debug:
                    print(f"  [osc] accumulating ({osc_persistence}/{args.oscillation_min_frames})...")
                try:
                    label, conf = _predict(pipeline, gestures, features)
                except Exception as e:
                    print(f"  ⚠ Prediction error: {e}")
                    time.sleep(0.02)
                    continue

            # ── per-gesture confidence threshold ──────────────────
            min_conf = gesture_conf_overrides.get(label, args.min_conf)
            if conf < min_conf:
                if args.debug:
                    print(f"  [filter] {label} conf={conf:.3f} < {min_conf:.3f}")
                continue

            # ── per-gesture minimum movement ──────────────────────
            min_movement = gesture_movement_overrides.get(label, 0.0)
            if movement < min_movement:
                if args.debug:
                    print(f"  [filter] {label} movement={movement:.3f} < {min_movement:.3f}")
                continue

            smooth_buffer.append(label)

            if len(smooth_buffer) >= args.min_vote:
                counts = {}
                for lbl in smooth_buffer:
                    counts[lbl] = counts.get(lbl, 0) + 1
                smoothed = max(counts, key=counts.get)

                if smoothed == displayed:
                    challenge_count = 0
                    challenge_label = None
                    display_age += 1
                elif display_age < min_display_frames:
                    display_age += 1
                elif smoothed == challenge_label:
                    challenge_count += 1
                else:
                    challenge_label = smoothed
                    challenge_count = 1

                if challenge_count >= args.change_frames:
                    # ── boxing type lock ─────────────────────────────
                    # Once the observation determines one-arm or two-arm,
                    # the other boxing type can't replace it until idle.
                    if smoothed in ("one-arm-boxing", "two-arm-boxing") and smoothed != boxing_type_locked and boxing_type_locked is not None:
                        if args.debug:
                            print(f"  [lock] {smoothed} ignored — {boxing_type_locked} is locked")
                        challenge_count = 0
                        challenge_label = None
                        continue
                    if smoothed in ("one-arm-boxing", "two-arm-boxing"):
                        if movement < args.min_boxing_movement:
                            if args.debug:
                                print(f"  ⚠ ignoring boxing — low movement ({movement:.3f} < {args.min_boxing_movement})")
                            # Fall through: show the model's prediction anyway
                            punch_total = 0
                            print(f"> {smoothed}  (conf={conf:.2f}, low mvmt)")
                            displayed = smoothed
                            display_age = 0
                            challenge_count = 0
                            challenge_label = None
                            hold_counter = max_hold_frames
                        else:
                            observe_until = frame_count + args.boxing_delay_frames
                            punch_total = 0
                            was_in_punch = False
                            punch_cooldown = 0
                            print(f"  observing for punches ({args.boxing_delay_frames} frames)...")
                    else:
                        print(f"> {_repeat_label(smoothed, fist_counter.cycles() if smoothed == args.oscillation_gesture else 0)}  (conf={conf:.2f})")
                        displayed = smoothed
                        display_age = 0
                        challenge_count = 0
                        challenge_label = None
                        hold_counter = max_hold_frames

            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\n\nDemo stopped.")
    finally:
        for reader in reader_map.values():
            reader.stop()
    print(f"Frames: {frame_count}")


def run_gui(args, pipeline, expected_n_features, gestures, reader_map, sensor_types,
             gesture_conf_overrides: dict[str, float] | None = None,
             gesture_movement_overrides: dict[str, float] | None = None):
    if gesture_conf_overrides is None:
        gesture_conf_overrides = {}
    if gesture_movement_overrides is None:
        gesture_movement_overrides = {}
    """Clean realtime GUI — resizable, fullscreen, no-nonsense."""
    try:
        import tkinter as tk
    except ImportError:
        print("--gui requires tkinter (install python-tk)")
        return

    # ── Reset gravity-tracking HPF for a fresh session ─────
    global _accel_lp
    _accel_lp = None

    # ── state variables ──────────────────────────────────────────────
    frame_buffer = deque(maxlen=args.window)
    smooth_buffer: deque[str] = deque(maxlen=args.smooth)
    frame_count = 0
    displayed = None
    challenge_label: str | None = None
    challenge_count = 0
    running = True
    movement_history: deque[float] = deque(maxlen=5)
    hold_counter = 0
    max_hold_frames = 8
    min_display_frames = 5
    display_age = 0
    dims_warned = False
    # Observation mode state (boxing type discrimination)
    observe_until: int | None = None
    punch_count = 0
    prev_accel: list[float] | None = None
    was_in_punch = False
    punch_cooldown = 0
    boxing_type_locked: str | None = None  # once set, other boxing type can't replace
    idle_cooldown = 0  # suppresses predictions briefly after idle clears
    osc_persistence = 0  # frames of sustained oscillation (requires ~2s to override)
    punch_total = 0  # punches counted in the current boxing session (kept while boxing stays up)
    fist_counter = CycleCounter()  # open/close fist repetition counter
    active_start: float | None = None
    total_active_time = 0.0
    fps_samples: deque[float] = deque(maxlen=60)
    last_fps_time: float | None = None
    idle_start: float | None = None

    # ── colour palette (plain data-science) ─────────────────────────
    PALETTE = {
        "bg": "#0d0d10",
        "fg": "#d4d4dc",
        "fg_dim": "#6a6a72",
        "fg_muted": "#4a4a52",
        "accent": "#4a7cbf",
        "success": "#5a9a6a",
        "warn": "#b89a3a",
    }

    # ── fonts (plain) ───────────────────────────────────────────────
    FONTS = {
        "mono": ("Courier", 17),
        "mono_small": ("Courier", 15),
        "mono_tiny": ("Courier", 14),
    }

    # ── root window ─────────────────────────────────────────────────
    root = tk.Tk()
    root.title("REALTIME GESTURE CLASSIFIER")
    root.geometry("800x620")
    root.configure(bg=PALETTE["bg"])
    root.resizable(True, True)
    root.minsize(600, 450)

    # fullscreen state
    _fullscreen = [False]

    def _toggle_fullscreen(event=None):
        _fullscreen[0] = not _fullscreen[0]
        root.attributes("-fullscreen", _fullscreen[0])
        if not _fullscreen[0]:
            root.geometry("800x620")

    root.bind("<F11>", _toggle_fullscreen)

    # ── top bar ──────────────────────────────────────────────────────
    top = tk.Frame(root, bg=PALETTE["bg"])
    top.pack(fill="x", padx=16, pady=(10, 2))

    tk.Label(top, text="● REAL-TIME GESTURE CLASSIFIER", font=("Helvetica", 14, "bold"),
             fg=PALETTE["accent"], bg=PALETTE["bg"]).pack(side="left")
    tk.Label(top, text="v1",
             font=("Helvetica", 12), fg=PALETTE["fg_dim"], bg=PALETTE["bg"]).pack(side="right")

    # thin rule
    rule = tk.Frame(root, height=1, bg=PALETTE["fg_muted"])
    rule.pack(fill="x", padx=16, pady=(4, 8))

    # ── main content area (single centered column) ──────────────────
    main = tk.Frame(root, bg=PALETTE["bg"])
    main.pack(fill="both", expand=True)

    # weight so the gesture label can expand into available space
    main.grid_rowconfigure(0, weight=1)
    main.grid_rowconfigure(1, weight=0)
    main.grid_rowconfigure(2, weight=0)
    main.grid_columnconfigure(0, weight=1)

    # ═════════════════════════════════════════════════════════════════
    # GESTURE PREDICTION (centred, expands)
    # ═════════════════════════════════════════════════════════════════

    gesture_label = tk.Label(main, text="—", font=("Helvetica", 120, "bold"),
                             fg="#ffffff", bg=PALETTE["bg"])
    gesture_label.grid(row=0, column=0, sticky="nsew")

    # ── movement meter ───────────────────────────────────────────────
    move_frame = tk.Frame(main, bg=PALETTE["bg"])
    move_frame.grid(row=1, column=0, pady=(0, 8), padx=60, sticky="ew")
    move_frame.grid_columnconfigure(0, weight=1)

    move_row = tk.Frame(move_frame, bg=PALETTE["bg"])
    move_row.grid(row=0, column=0, sticky="ew", pady=(0, 2))
    move_row.grid_columnconfigure(0, weight=1)
    move_row.grid_columnconfigure(1, weight=0)

    tk.Label(move_row, text="movement", font=FONTS["mono_small"],
             fg=PALETTE["fg_muted"], bg=PALETTE["bg"], anchor="w").grid(row=0, column=0, sticky="w")
    move_val_label = tk.Label(move_row, text="0.0000", font=FONTS["mono"],
                              fg=PALETTE["fg"], bg=PALETTE["bg"], anchor="e")
    move_val_label.grid(row=0, column=1, sticky="e")

    move_bar = tk.Canvas(move_frame, height=6, bg=PALETTE["bg"],
                         highlightthickness=1, highlightbackground=PALETTE["fg_muted"])
    move_bar.grid(row=1, column=0, sticky="ew")

    # ── status line ──────────────────────────────────────────────────
    led_frame = tk.Frame(main, bg=PALETTE["bg"])
    led_frame.grid(row=2, column=0, pady=(4, 6))
    status_led = tk.Label(led_frame, text="○  idle  ", font=FONTS["mono_small"],
                          fg=PALETTE["fg_muted"], bg=PALETTE["bg"])
    status_led.pack(side="left")
    frame_label = tk.Label(led_frame, text="frame  0", font=FONTS["mono_small"],
                           fg=PALETTE["fg_muted"], bg=PALETTE["bg"])
    frame_label.pack(side="left")

    # ═════════════════════════════════════════════════════════════════
    # BOTTOM BAR — Session Statistics
    # ═════════════════════════════════════════════════════════════════

    rule2 = tk.Frame(root, height=1, bg=PALETTE["fg_muted"])
    rule2.pack(fill="x", padx=16, pady=(6, 4))

    stats_bar = tk.Frame(root, bg=PALETTE["bg"])
    stats_bar.pack(fill="x", padx=16, pady=(0, 8))

    stats_labels = {}
    for i, (key, text) in enumerate([
        ("frames", "frames"),
        ("active", "active"),
        ("fps", "fps"),
        ("idle", "idle"),
        ("sensor", "sensor"),
        ("mode", "mode"),
    ]):
        lbl = tk.Label(stats_bar, text=f"{text}  —", font=FONTS["mono_tiny"],
                       fg=PALETTE["fg_dim"], bg=PALETTE["bg"])
        lbl.pack(side="left", padx=(0, 12))
        stats_labels[key] = lbl

    error_label = tk.Label(root, text="", font=FONTS["mono_tiny"],
                           fg=PALETTE["warn"], bg=PALETTE["bg"], wraplength=680)
    error_label.pack(side="bottom", padx=16, pady=(0, 4), anchor="w")

    # ═════════════════════════════════════════════════════════════════
    # POLLING LOOP
    # ═════════════════════════════════════════════════════════════════

    last_movement = 0.0

    def _draw_bar(canvas: tk.Canvas, fraction: float, color: str) -> None:
        w = canvas.winfo_width()
        if w < 2:
            return
        canvas.delete("all")
        bar_w = max(1, int(w * fraction))
        canvas.create_rectangle(0, 0, bar_w, 8, fill=color, outline="")
        canvas.create_rectangle(bar_w, 0, w, 8, fill=PALETTE["bg"], outline="")



    def poll():
        nonlocal running, frame_count, displayed, challenge_label, challenge_count
        nonlocal hold_counter, movement_history, display_age, dims_warned, last_movement

        nonlocal idle_cooldown, active_start, total_active_time, osc_persistence
        nonlocal fps_samples, last_fps_time, idle_start
        nonlocal observe_until, punch_count, prev_accel, was_in_punch, punch_cooldown, boxing_type_locked
        nonlocal punch_total, fist_counter

        if not running:
            return

        now = time.time()

        try:
            frame_data = {name: reader.read() for name, reader in reader_map.items()}
            frame_count += 1
            frame_buffer.append(frame_data)
            window = list(frame_buffer)

            # ── FPS tracking ────────────────────────────────────────
            if last_fps_time is not None:
                dt = now - last_fps_time
                if dt > 0:
                    fps_samples.append(1.0 / dt)
            last_fps_time = now

            fps = np.mean(fps_samples) if fps_samples else 0.0

            # ── update frame counter ────────────────────────────────
            frame_label.config(text=f"frame  {frame_count}")
            stats_labels["frames"].config(text=f"frames  {frame_count}")
            stats_labels["fps"].config(text=f"fps  {fps:.1f}")

            if len(window) >= args.window:
                # ── gather IMU data for boxing detection ─────────
                current_accel: list[float] | None = None
                for name, stype in sensor_types.items():
                    if stype == "imu" and name in frame_data:
                        reading = frame_data[name]
                        if reading and reading.data:
                            accel = reading.data.get("accel", None)
                            if accel is not None:
                                current_accel = list(map(float, accel))
                                break

                # ── compute features ────────────────────────────────
                raw_movement = 0.0
                features = []
                for name in args.sensors:
                    readings = _gather_readings(window, name)
                    raw_movement += _movement_score(readings, sensor_types[name])
                    features.extend(_compute_features(readings, sensor_types[name]))

                if not dims_warned:
                    err = _check_feature_dims(len(features), expected_n_features, args.sensors)
                    if err:
                        error_label.config(text=f"⚠ {err}", fg=PALETTE["warn"])
                    dims_warned = True

                movement_history.append(raw_movement)
                movement = np.mean(movement_history)
                last_movement = movement

                is_idle = movement < args.idle_threshold

                # ── gyro oscillation override ────────────────────────
                # Soli (finger rub) produces minimal wrist accel but
                # distinctive oscillatory gyro. Override idle so the model
                # gets to classify it.
                if is_idle:
                    imu_readings = _gather_readings(window, "imu")
                    if imu_readings and _check_gyro_oscillation(imu_readings):
                        is_idle = False

                # ── boxing: count punches ────────────────────────────────
                # Counts while the type is being determined (observation) AND
                # while boxing stays on screen, so the running punch count is
                # always visible. A "punch" = a per-frame accel transient above
                # punch_threshold, debounced by punch_cooldown.
                boxing_active = observe_until is not None or displayed in ("one-arm-boxing", "two-arm-boxing")
                if boxing_active and current_accel is not None and prev_accel is not None:
                    delta = sum(abs(current_accel[i] - prev_accel[i]) for i in range(3))
                    in_punch = delta > args.punch_threshold
                    if punch_cooldown > 0:
                        punch_cooldown -= 1
                    elif in_punch and not was_in_punch:
                        punch_total += 1
                        punch_cooldown = args.punch_cooldown
                        if displayed in ("one-arm-boxing", "two-arm-boxing"):
                            gesture_label.config(text=displayed.upper().replace("-", "\n"),
                                                 fg=PALETTE["success"])
                    was_in_punch = in_punch

                if current_accel is not None:
                    prev_accel = current_accel[:]

                if observe_until is not None:
                    if frame_count >= observe_until:
                        if punch_total >= 2:
                            displayed = "one-arm-boxing"
                        else:
                            displayed = "two-arm-boxing"
                        print(f"[{time.strftime('%H:%M:%S')}] {displayed}")
                        boxing_type_locked = displayed  # lock this boxing type
                        gesture_label.config(text=displayed.upper().replace("-", "\n"), fg=PALETTE["success"])
                        display_age = 0
                        challenge_count = 0
                        challenge_label = None
                        hold_counter = max_hold_frames
                        error_label.config(text="")
                        # stop observing but KEEP counting punches while boxing stays up
                        observe_until = None
                        was_in_punch = False
                        punch_cooldown = 0
                        root.after(25, poll)
                        return
                    else:
                        gesture_label.config(fg=PALETTE["warn"])
                        error_label.config(text=f"observing... {punch_total} punches", fg=PALETTE["fg_dim"])
                        root.after(25, poll)
                        return

                # ── idle / active tracking ───────────────────────────
                if is_idle:
                    if active_start is not None:
                        total_active_time += now - active_start
                        active_start = None
                    if idle_start is None:
                        idle_start = now
                    idle_duration = now - idle_start
                    status_led.config(text="○  idle", fg=PALETTE["fg_muted"])
                    stats_labels["active"].config(
                        text=f"active  {total_active_time:.1f}s")
                    stats_labels["idle"].config(
                        text=f"idle  {idle_duration:.1f}s")
                else:
                    if active_start is None:
                        active_start = now
                    active_duration = total_active_time + (now - active_start)
                    idle_start = None
                    status_led.config(text="●  active", fg=PALETTE["success"])
                    stats_labels["active"].config(
                        text=f"active  {active_duration:.1f}s")
                    stats_labels["idle"].config(text="idle  0.0s")

                # ── update movement / conf bars ──────────────────────
                move_val_label.config(text=f"{movement:.4f}  (τ={args.idle_threshold:.2f})")
                _draw_bar(move_bar, min(movement / max(args.idle_threshold * 3, 0.01), 1.0),
                               PALETTE["fg_dim"])

                # ── idle ──
                if is_idle:
                    if displayed is not None:
                        if hold_counter > 0:
                            hold_counter -= 1
                            display_age += 1
                        else:
                            gesture_label.config(text="—", fg="#ffffff")
                            displayed = None
                            smooth_buffer.clear()
                            movement_history.clear()
                            challenge_count = 0
                            challenge_label = None
                            observe_until = None
                            punch_count = 0
                            prev_accel = None
                            was_in_punch = False
                            punch_cooldown = 0
                            boxing_type_locked = None
                            osc_persistence = 0
                            punch_total = 0
                            fist_counter.reset()
                            idle_cooldown = 15  # suppress predictions briefly to prevent flash
                            error_label.config(text="")
                    else:
                        smooth_buffer.clear()
                        challenge_count = 0
                        challenge_label = None
                        observe_until = None
                        punch_count = 0
                        prev_accel = None
                        was_in_punch = False
                        punch_cooldown = 0
                        boxing_type_locked = None
                        osc_persistence = 0
                        punch_total = 0
                        fist_counter.reset()
                    root.after(25, poll)
                    return

                # hold_counter is set to max_hold_frames when a new gesture is
                # displayed (in the GUI update section below). Do NOT reset it
                # here — that would make displayed gestures stick forever.

                # ── idle cooldown ─────────────────────────────────────
                # After clearing a gesture, briefly suppress predictions so
                # transient movement noise doesn't flash a stale prediction.
                if idle_cooldown > 0:
                    idle_cooldown -= 1
                    # Cancel cooldown early if genuinely new movement starts
                    if movement > args.idle_threshold * 2.0:
                        idle_cooldown = 0
                    else:
                        root.after(25, poll)
                        return

                # ── sustained oscillation detection (signal-based) ──
                # Check the raw IMU signal for multi-cycle oscillation.
                imu_readings = _gather_readings(window, "imu")
                sustained_osc, osc_frac = False, 0.0
                if imu_readings and len(imu_readings) >= 6:
                    sustained_osc, osc_frac = _check_sustained_oscillation(
                        imu_readings,
                        args.oscillation_min_crossings,
                        args.oscillation_min_channels,
                    )

                # Track persistence: oscillation must hold for ~2 seconds
                if sustained_osc:
                    osc_persistence += 1
                else:
                    osc_persistence = 0

                # ── open/close fist cycle counter ─────────────────────
                # While oscillation is sustained, feed the current frame's
                # gyro into the counter (locks onto the strongest channel).
                if sustained_osc:
                    if fist_counter.channel < 0:
                        fist_counter.start(imu_readings)
                    cur_gyro = None
                    for name, stype in sensor_types.items():
                        if stype == "imu" and name in frame_data and frame_data[name]:
                            cur_gyro = frame_data[name].data.get("gyro")
                            break
                    fist_counter.update(cur_gyro)
                    if displayed == args.oscillation_gesture:
                        gesture_label.config(
                            text=_repeat_label(displayed, fist_counter.cycles()), fg="#ffffff")
                else:
                    fist_counter.reset()

                # ── oscillation sustained — check displayed gesture first ──
                if osc_persistence >= args.oscillation_min_frames:
                    if displayed in args.oscillation_block:
                        # Already showing a blocked gesture — don't override
                        try:
                            label, conf = _predict(pipeline, gestures, features)
                        except Exception as e:
                            error_label.config(text=f"⚠ prediction error: {e}", fg=PALETTE["warn"])
                            root.after(30, poll)
                            return
                        if args.debug:
                            print(f"  [osc] protected — {displayed} is in blocklist")
                    else:
                        try:
                            model_label, model_conf = _predict(pipeline, gestures, features)
                        except Exception as e:
                            model_label, model_conf = None, 0.0
                        if model_label in args.oscillation_block and model_conf >= args.min_conf:
                            label = model_label
                            conf = model_conf
                            if args.debug:
                                print(f"  [osc] blocked — {label} ({conf:.2f}) accepted")
                        else:
                            label = args.oscillation_gesture
                            conf = 0.95
                            if args.debug:
                                print(f"  [osc] {osc_persistence}f sustained → {label}")
                else:
                    if sustained_osc and args.debug:
                        print(f"  [osc] accumulating ({osc_persistence}/{args.oscillation_min_frames})...")
                    try:
                        label, conf = _predict(pipeline, gestures, features)
                    except Exception as e:
                        error_label.config(text=f"⚠ prediction error: {e}", fg=PALETTE["warn"])
                        root.after(30, poll)
                        return

                # ── per-gesture confidence threshold ─────────────────
                min_conf = gesture_conf_overrides.get(label, args.min_conf)
                # ── per-gesture minimum movement ───────────────────
                min_movement = gesture_movement_overrides.get(label, 0.0)

                if conf >= min_conf and movement >= min_movement:
                    smooth_buffer.append(label)

                if len(smooth_buffer) >= args.min_vote:
                    counts = {}
                    for lbl in smooth_buffer:
                        counts[lbl] = counts.get(lbl, 0) + 1
                    smoothed = max(counts, key=counts.get)

                    if smoothed == displayed:
                        challenge_count = 0
                        challenge_label = None
                        display_age += 1
                        gesture_label.config(fg="#ffffff")
                    elif display_age < min_display_frames:
                        display_age += 1
                    elif smoothed == challenge_label:
                        challenge_count += 1
                    else:
                        challenge_label = smoothed
                        challenge_count = 1

                    if challenge_count >= args.change_frames:
                        # ── boxing type lock ─────────────────────────
                        # Once the observation determines one-arm or two-arm,
                        # the other boxing type can't replace it until idle.
                        if smoothed in ("one-arm-boxing", "two-arm-boxing") and smoothed != boxing_type_locked and boxing_type_locked is not None:
                            challenge_count = 0
                            challenge_label = None
                            root.after(25, poll)
                            return
                        if smoothed in ("one-arm-boxing", "two-arm-boxing"):
                            if movement < args.min_boxing_movement:
                                error_label.config(text=f"{smoothed} (low movement: {movement:.2f})", fg=PALETTE["warn"])
                                punch_total = 0
                                gesture_label.config(text=smoothed.upper().replace("-", "\n"), fg=PALETTE["warn"])
                                displayed = smoothed
                                print(f"[{time.strftime('%H:%M:%S')}] {smoothed}")
                                display_age = 0
                                challenge_count = 0
                                challenge_label = None
                                hold_counter = max_hold_frames
                            else:
                                observe_until = frame_count + args.boxing_delay_frames
                                punch_total = 0
                                was_in_punch = False
                                punch_cooldown = 0
                                gesture_label.config(fg=PALETTE["warn"])
                                error_label.config(text=f"observing for punches ({args.boxing_delay_frames} frames)...", fg=PALETTE["fg_dim"])
                        else:
                            display_upper = _repeat_label(smoothed, fist_counter.cycles() if smoothed == args.oscillation_gesture else 0)
                            gesture_label.config(text=display_upper, fg="#ffffff")
                            displayed = smoothed
                            print(f"[{time.strftime('%H:%M:%S')}] {display_upper}")
                            display_age = 0
                            challenge_count = 0
                            challenge_label = None
                            hold_counter = max_hold_frames
                            error_label.config(text="")

            # ── update stats bar sensor info ─────────────────────────
            stats_labels["sensor"].config(text=f"sensor  {', '.join(args.sensors)}")
            stats_labels["mode"].config(text=f"mode  {args.mode}")

        except Exception as e:
            error_label.config(text=f"⚠ {e}", fg=PALETTE["warn"])

        root.after(25, poll)

    # ── Signal handlers (Ctrl+C / Ctrl+Z) ────────────────────────────
    # tkinter's C-level mainloop blocks KeyboardInterrupt, so the window
    # stays stuck on Ctrl+C.  We register a handler that schedules
    # root.quit() + root.destroy() on the event loop, causing mainloop()
    # to exit cleanly so readers get stopped in the finally block.
    #
    # Ctrl+Z (SIGTSTP) suspends the whole process — the window freezes
    # and becomes "not responding".  We catch it, clean up readers first,
    # then re-raise with the default handler so the shell can suspend.

    def _on_sigint(signum, frame):
        nonlocal running
        running = False
        print("\n  Shutting down (Ctrl+C)...")
        # Schedule both on the event loop so they run on the main thread
        root.after_idle(root.quit)
        root.after_idle(root.destroy)

    def _on_sigtstp(signum, frame):
        nonlocal running
        running = False
        print("\n  Suspending (Ctrl+Z) — cleaning up.")
        for reader in reader_map.values():
            reader.stop()
        # Schedule window close for when the process resumes after 'fg'
        root.after_idle(root.quit)
        root.after_idle(root.destroy)
        # Restore default handler and re-raise so the shell can suspend
        if hasattr(signal, "SIGTSTP"):
            signal.signal(signal.SIGTSTP, signal.SIG_DFL)
            os.kill(os.getpid(), signal.SIGTSTP)

    signal.signal(signal.SIGINT, _on_sigint)
    if hasattr(signal, "SIGTSTP"):
        signal.signal(signal.SIGTSTP, _on_sigtstp)

    # ── start ───────────────────────────────────────────────────────
    def on_close():
        nonlocal running
        running = False
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.after(100, poll)
    try:
        root.mainloop()
    finally:
        print("Stopping readers...")
        for reader in reader_map.values():
            reader.stop()
        print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time gesture classification demo")
    parser.add_argument("--model", default="models/imu_v3_best_model.pkl",
                        help="Path to trained model pickle")
    parser.add_argument("--features", default=None,
                        help="Path to features NPZ (for metadata)")
    parser.add_argument("--sensors", nargs="+", default=["imu"],
                        choices=list(SENSOR_REGISTRY.keys()),
                        help="Sensors to use")
    parser.add_argument("--mode", default="mock",
                        choices=["mock", "serial"],
                        help="Sensor mode")
    parser.add_argument("--uwb-ports", nargs="+", default=["/dev/ttyACM0"],
                        help="Serial ports for UWB devices")
    parser.add_argument("--window", type=int, default=20,
                        help="Window size (matches training; v3 model = 20)")
    parser.add_argument("--idle-threshold", type=float, default=0.10,
                        help="Movement score (linear accel from HPF) below this = idle (default: 0.10). "
                             "Higher values (e.g. 0.45) fragment a gesture into tiny sub-segments "
                             "between the soli's intermittent movement dips, and those get rejected "
                             "as too short.")
    parser.add_argument("--min-conf", type=float, default=0.68,
                        help="Minimum prediction confidence to accept (default: 0.68)")
    parser.add_argument("--smooth", type=int, default=10,
                        help="Smoothing buffer size (majority vote over last N frames)")
    parser.add_argument("--min-vote", type=int, default=3,
                        help="Minimum frames in buffer before showing prediction")
    parser.add_argument("--change-frames", type=int, default=6,
                        help="Require new label to dominate this many consecutive frames before switching (default: 6)")
    parser.add_argument("--gui", action="store_true",
                        help="Show prediction GUI window")
    parser.add_argument("--punch-threshold", type=float, default=1.0,
                        help="Per-frame accel delta threshold for detecting a punch (default: 1.0)")
    parser.add_argument("--punch-cooldown", type=int, default=10,
                        help="Frames to wait between consecutive punch detections (default: 10)")
    parser.add_argument("--boxing-delay-frames", type=int, default=60,
                        help="Frames to observe after boxing is detected before classifying one-arm vs two-arm (default: 60)")
    parser.add_argument("--min-boxing-movement", type=float, default=0.8,
                        help="Minimum movement score to trigger boxing observation (default: 0.8)")
    parser.add_argument("--accel-gain", type=float, default=1.0,
                        help="Scale factor for accelerometer values (default: 1.0 — model trained on raw accel)")
    parser.add_argument("--gyro-gain", type=float, default=1.0,
                        help="Scale factor for gyro values before model (default: 1.0 — model trained on raw gyro)")
    parser.add_argument("--gyro-deadband", type=float, default=0.0,
                        help="Gyro deadband in dps — values below this are zeroed out (default: 0.0)")
    parser.add_argument("--gesture-conf", nargs="+", default=[],
                        help="Per-gesture confidence thresholds: gesture=threshold (e.g. push=0.85 soli=0.9). "
                             "Overrides --min-conf for specific gestures.")
    parser.add_argument("--gesture-min-movement", nargs="+", default=[],
                        help="Per-gesture minimum movement: gesture=threshold (e.g. push=0.15 soli=0.2). "
                             "Require more movement for specific gestures to be accepted.")
    parser.add_argument("--oscillation-gesture", default="making-fist-open",
                        help="Gesture to display when the raw IMU signal shows sustained "
                             "multi-cycle oscillation (default: making-fist-open). "
                             "Repetitive gestures produce this signature; transient ones don't.")
    parser.add_argument("--oscillation-min-crossings", type=int, default=3,
                        help="Minimum zero-crossings per IMU channel to count as oscillating "
                             "(default: 3 = ~1.5+ cycles). Lower to 2 for more sensitivity.")
    parser.add_argument("--oscillation-min-channels", type=int, default=3,
                        help="Minimum IMU channels (of 6: ax,ay,az,gx,gy,gz) that must show "
                             "sustained oscillation (default: 3 = half the channels)")
    parser.add_argument("--oscillation-min-frames", type=int, default=40,
                        help="Minimum consecutive frames of oscillation before overriding the "
                             "prediction ~1 second at 40fps (default: 40)")
    parser.add_argument("--oscillation-block", nargs="+", default=["bye-bye", "clapping"],
                        help="Gesture labels that should NOT be overridden by oscillation "
                             "detection. These gestures (e.g., bye-bye, clapping, wave) also "
                             "produce oscillation but should keep their model-predicted label "
                             "(default: bye-bye clapping)")
    parser.add_argument("--debug", action="store_true",
                        help="Print per-frame predictions")
    parser.add_argument("--imu-port", default=None,
                        help="IMU serial port")
    parser.add_argument("--imu-baud", type=int, default=115200,
                        help="IMU serial baud rate")
    # ── Confirmed-gesture mode (buffer-then-confirm) ─────────────────
    parser.add_argument("--confirmed", action="store_true",
                        help="Confirmed-gesture mode: buffer movement continuously, and the moment "
                             "you stop moving (after confirm-seconds of stillness) emit exactly one "
                             "prediction from the frozen last-second of data. Zero false positives.")
    parser.add_argument("--confirm-frames", type=int, default=40,
                        help="Still frames after the gesture that confirm completion, used only when "
                             "--confirm-seconds is 0 (~1s at 40fps, default: 40)")
    parser.add_argument("--confirm-seconds", type=float, default=1.5,
                        help="Seconds of stillness required after the gesture before emitting — "
                             "time-based and fps-independent (default: 1.5). Set 0 to fall back to "
                             "--confirm-frames (frame-based).")
    parser.add_argument("--min-segment-frames", type=int, default=0,
                        help="Min gesture length in frames; 0 = auto (max(2, window//4)) "
                             "(default: 0)")
    parser.add_argument("--resume-frames", type=int, default=3,
                        help="Consecutive moving frames during the confirm window that count as a "
                             "genuine gesture continuation and resume buffering (wobble shorter than "
                             "this never pauses or restarts the confirm clock) (default: 3)")
    parser.add_argument("--onset-frames", type=int, default=2,
                        help="Consecutive moving frames required before recording a gesture starts. "
                             "Raises this (e.g. 4-6) to ignore small movements/wrist jitter; the "
                             "detector won't start until movement is sustained (default: 2)")
    parser.add_argument("--fist-counter", action="store_true",
                        help="Show the open/close-fist repetition counter ('MAKING FIST OPEN × N'). "
                             "The making-fist-open detection itself is ALWAYS on (signal override, "
                             "hardened against breathing); this flag only adds the × N counter "
                             "display. Default: off, plain 'MAKING FIST OPEN'.")
    parser.add_argument("--max-segment-frames", type=int, default=240,
                        help="Max gesture length in frames; longer segments are rejected (default: 240 = 6s at 40fps)")
    parser.add_argument("--emit-conf", type=float, default=0.30,
                        help="Minimum confidence to emit a confirmed prediction. The zero-false- "
                             "positive guarantee comes from the state machine (stillness "
                             "confirmation + one emission per segment + cooldown); this gate is a "
                             "secondary filter. Short gestures padded with idle frames can score as "
                             "low as 0.3, so the default is permissive; raise it (e.g. 0.6+) to "
                             "show only high-confidence results at the cost of more misses "
                             "(default: 0.30)")
    parser.add_argument("--vote-windows", type=int, default=3,
                        help="Number of overlapping tail windows voted on (default: 3)")
    parser.add_argument("--vote-majority", type=int, default=2,
                        help="Votes required to emit; 0 = all windows must agree (default: 2 of --vote-windows)")
    parser.add_argument("--cooldown-frames", type=int, default=30,
                        help="Frames to lock out after emitting so one trial emits once (default: 30)")
    args = parser.parse_args()

    # ── Parse per-gesture overrides ─────────────────────────────────
    gesture_conf_overrides: dict[str, float] = {}
    for item in args.gesture_conf:
        if "=" in item:
            gesture, threshold = item.split("=", 1)
            gesture = gesture.strip().lower()
            threshold = threshold.strip()
            try:
                gesture_conf_overrides[gesture] = float(threshold)
            except ValueError:
                print(f"  ⚠ Warning: ignoring malformed --gesture-conf entry '{item}' — "
                      f"'{threshold}' is not a valid number")
        else:
            print(f"  ⚠ Warning: ignoring malformed --gesture-conf entry '{item}' (expected gesture=threshold)")

    gesture_movement_overrides: dict[str, float] = {}
    for item in args.gesture_min_movement:
        if "=" in item:
            gesture, threshold = item.split("=", 1)
            gesture = gesture.strip().lower()
            threshold = threshold.strip()
            try:
                gesture_movement_overrides[gesture] = float(threshold)
            except ValueError:
                print(f"  ⚠ Warning: ignoring malformed --gesture-min-movement entry '{item}' — "
                      f"'{threshold}' is not a valid number")
        else:
            print(f"  ⚠ Warning: ignoring malformed --gesture-min-movement entry '{item}' (expected gesture=threshold)")

    if gesture_conf_overrides or gesture_movement_overrides:
        print("Per-gesture overrides:")
        if gesture_conf_overrides:
            items = ", ".join(f"{k}={v}" for k, v in sorted(gesture_conf_overrides.items()))
            print(f"  Confidence: {items}")
        if gesture_movement_overrides:
            items = ", ".join(f"{k}={v}" for k, v in sorted(gesture_movement_overrides.items()))
            print(f"  Min movement: {items}")

    # Show oscillation detection config
    print(f"Oscillation override: {args.oscillation_gesture} "
          f"(min {args.oscillation_min_crossings} crossings, "
          f"min {args.oscillation_min_channels}/6 channels, "
          f"require {args.oscillation_min_frames} consecutive frames)")

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: model not found: {model_path}. Run train.py first.")
        return

    with open(model_path, "rb") as f:
        raw = pickle.load(f)
    if isinstance(raw, dict):
        pipeline = raw["pipeline"]
        gestures_list = raw.get("gestures", [])
    else:
        pipeline = raw
        gestures_list = []

    if not gestures_list and args.features:
        features_path = Path(args.features)
        if features_path.exists():
            data = np.load(features_path, allow_pickle=True)
            gestures_list = data["gestures"].tolist() if "gestures" in data else []

    expected_n_features = pipeline.n_features_in_

    # ── Detect whether model has extra spectral/correlation features ──
    global _use_extra_features
    model_features = raw.get("feature_names", []) if isinstance(raw, dict) else []
    n_extra = sum(1 for fn in model_features if "spec_" in fn or "corr_" in fn)
    _use_extra_features = n_extra > 0

    # ── pre-flight feature compatibility check ──────────────────────
    computed_n = _compute_expected_feature_count(args.sensors, args.window)
    if computed_n != expected_n_features:
        if args.sensors == ["imu"]:
            # Determine which formula to use for the suggestion
            base = 60 if _use_extra_features else 32
            suggested_w = (expected_n_features - base) // 8 + 1
            if suggested_w >= 2:
                print(
                    f"Error: sensor/window mismatch.\n"
                    f"  Model expects {expected_n_features} features, but --sensors {args.sensors}"
                    f" with --window {args.window} produces {computed_n} features."
                    f"\n  → Try: --window {suggested_w}"
                )
            else:
                print(
                    f"Error: sensor/window mismatch.\n"
                    f"  Model expects {expected_n_features} features, but --sensors {args.sensors}"
                    f" with --window {args.window} produces {computed_n} features."
                )
        else:
            print(
                f"Error: sensor/window mismatch.\n"
                f"  Model expects {expected_n_features} features, but --sensors {args.sensors}"
                f" with --window {args.window} produces {computed_n} features."
            )
        return

    print(f"Loaded: {model_path}")
    if gestures_list:
        print(f"Gestures: {', '.join(gestures_list)}")
    print(f"Features expected by model: {expected_n_features} (computed: {computed_n}) ✓")
    print(f"Window: {args.window}, threshold: {args.idle_threshold}")
    print()

    reader_map: dict[str, any] = {}
    sensor_types: dict[str, str] = {}
    for name in args.sensors:
        cls = SENSOR_REGISTRY[name]
        if name == "uwb":
            for i, port in enumerate(args.uwb_ports):
                key = f"uwb_{i}"
                reader = cls(mode=args.mode, serial_port=port, sensor_id=f"uwb-{i}")
                reader.start()
                reader_map[key] = reader
                sensor_types[key] = "uwb"
                print(f"  Started {key} reader ({args.mode} mode, port={port})")
        elif name == "imu":
            if args.mode == "serial" and args.imu_port is None:
                print("  ⚠ Warning: --mode serial requires --imu-port. Falling back to mock data.")
                imu_mode = "mock"
            else:
                imu_mode = args.mode
            reader = cls(mode=imu_mode, serial_port=args.imu_port, baudrate=args.imu_baud)
            reader.start()
            reader_map[name] = reader
            sensor_types[name] = reader.sensor_type
            port_info = f" port={args.imu_port}" if args.imu_port else ""
            print(f"  Started {name} reader ({imu_mode} mode{port_info})")
        else:
            reader = cls(mode=args.mode)
            reader.start()
            reader_map[name] = reader
            sensor_types[name] = reader.sensor_type
            print(f"  Started {name} reader ({args.mode} mode)")

    # Apply IMU processing params
    global _accel_gain, _gyro_gain, _gyro_deadband
    _accel_gain = args.accel_gain
    _gyro_gain = args.gyro_gain
    _gyro_deadband = args.gyro_deadband

    if args.confirmed:
        if args.gui:
            run_confirmed_gui(args, pipeline, expected_n_features, gestures_list, reader_map,
                              sensor_types, gesture_conf_overrides, gesture_movement_overrides)
        else:
            run_confirmed(args, pipeline, expected_n_features, gestures_list, reader_map,
                          sensor_types, gesture_conf_overrides, gesture_movement_overrides)
    elif args.gui:
        run_gui(args, pipeline, expected_n_features, gestures_list, reader_map,
                sensor_types, gesture_conf_overrides, gesture_movement_overrides)
    else:
        run_terminal(args, pipeline, expected_n_features, gestures_list, reader_map,
                     sensor_types, gesture_conf_overrides, gesture_movement_overrides)


if __name__ == "__main__":
    main()