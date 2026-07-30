from __future__ import annotations

import argparse
import pickle
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


def extract_features_from_reading(reading: any, sensor_type: str) -> list[float]:
    if sensor_type == "mmwave":
        data = reading.data
        points = data.get("points", [])
        num_points = data.get("num_points", len(points))
        range_profile = data.get("range_profile", [])
        if not points:
            return [float(num_points), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        xs = np.array([p.get("x", 0) for p in points])
        ys = np.array([p.get("y", 0) for p in points])
        return [
            float(num_points),
            float(np.mean(xs)), float(np.std(xs)),
            float(np.min(xs)),  # closest point
            float(np.mean(ys)), float(np.std(ys)),
            float(range_profile[0]) if range_profile else 0.0,
            float(np.sqrt(np.mean(xs)**2 + np.mean(ys)**2)),  # distance from origin
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


def _movement_score(readings: list, sensor_type: str) -> float:
    """Movement score based on accel frame-to-frame deltas (accel-only).

    Gyro is intentionally excluded — even holding still, the wrist IMU
    picks up tiny rotational jitter (tremor, sensor noise) that would make
    the system never reach idle.  Subtle rotational gestures like soli are
    instead caught by _check_gyro_oscillation() which checks for the
    specific oscillatory pattern of finger rub.
    """
    if sensor_type == "imu":
        if len(readings) < 2:
            return 0.0
        accel_deltas = []
        for i in range(1, len(readings)):
            a0 = readings[i-1].data.get("accel", [0,0,0])
            a1 = readings[i].data.get("accel", [0,0,0])
            d = abs(a1[0]-a0[0]) + abs(a1[1]-a0[1]) + abs(a1[2]-a0[2])
            accel_deltas.append(d)
        return float(np.mean(accel_deltas)) if accel_deltas else 0.0
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
    """Detect subtle oscillatory gestures (soli finger rub) that accel-only
    idle detection would miss.

    Applies the same gyro gain + deadband as the feature pipeline so that
    the override is consistent with what the model actually sees.

    Uses deliberately high thresholds so that only deliberate finger-rub
    oscillation triggers the override — tiny wrist jitter at rest stays idle.

    Returns True if at least 2 of 3 gyro channels show both:
      - Strong amplitude (RMS > 1.5 dps after deadband)
      - Fast oscillation (zero-crossing rate > 0.30)
    """
    if len(readings) < 4:
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
        if rms < 1.5:
            continue
        centered = col - np.mean(col)
        crossings = int(np.sum((centered[:-1] * centered[1:]) < 0))
        zcr = crossings / len(col) if len(col) > 0 else 0.0
        if zcr > 0.30:
            oscillating += 1
    return oscillating >= 2

def _gather_readings(window: list[dict], name: str) -> list:
    return [f[name] for f in window if name in f]


def _compute_features(readings: list, sensor_type: str) -> list[float]:
    sensor_feats = np.array([extract_features_from_reading(r, sensor_type) for r in readings])
    if len(sensor_feats) == 0:
        n = (6 + 2) + (6 + 2) + ((3 + 3 + 2) * (len(readings) - 1)) if sensor_type == "imu" else 17
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


def _check_feature_dims(n_computed: int, n_expected: int, sensor_names: list[str]) -> str | None:
    if n_computed == n_expected:
        return None
    # IMU: 8 means + 8 stds + 8 RMS + 8 ZCR + 8×(W-1) deltas = 32 + 8×(W-1)
    approx_w = (n_computed - 32) // 8 + 1 if n_computed >= 32 else 0
    return (
        f"Feature mismatch: computed {n_computed} features, but model expects {n_expected}. "
        f"Check --window (~{approx_w} for IMU, or use different sensor)"
    )


def _infer_window_size(
    n_features_expected: int,
    sensors: list[str],
    feature_names: list[str] | None = None,
) -> int | None:
    """Infer the required window size from the model's feature count.

    The feature computation in ``_compute_features`` produces a deterministic
    number of features given the window size.  For IMU + mmwave:

        mmwave: 8 mean + 8 std + 1 path   = 17
        IMU:    8 mean + 8 std + 8 RMS + 8 ZCR
                + 8*(W-1) deltas           = 32 + 8*(W-1)
        total (base):  49 + 8*(W-1)

    Some models (e.g. imu_v3) include extra features (spectral, correlation).
    We detect those via ``feature_names`` and subtract them before computing
    the base IMU feature count.

    Returns
    -------
    int | None
        Inferred window size, or None if it cannot be determined.
    """
    has_mm = "mmwave" in sensors
    has_imu = "imu" in sensors

    # Subtract extra (non-base) features that some model versions add.
    extra = 0
    if feature_names:
        extra = sum(1 for fn in feature_names
                    if 'spec_' in fn or 'corr_' in fn)

    if has_mm and has_imu:
        # 17 mmwave + 32 IMU base + 8*(W-1) IMU deltas
        remaining = n_features_expected - extra - 49
        if remaining >= 0 and remaining % 8 == 0:
            return remaining // 8 + 1
    elif has_imu and not has_mm:
        # 32 IMU base + 8*(W-1) IMU deltas
        remaining = n_features_expected - extra - 32
        if remaining >= 0 and remaining % 8 == 0:
            return remaining // 8 + 1
    elif has_mm and not has_imu:
        if n_features_expected == 17 + extra:
            return 1

    # Last resort: brute-force search plausible window sizes
    plausible = [2, 3, 5, 10, 15, 20, 25, 30]
    for w in plausible:
        base = (17 if has_mm else 0) + (32 + 8 * (w - 1) if has_imu else 0)
        if base + extra == n_features_expected:
            return w

    return None


def run_terminal(args, pipeline, expected_n_features, gestures, reader_map, sensor_types):
    frame_buffer = deque(maxlen=args.window)
    smooth_buffer: deque[str] = deque(maxlen=args.smooth)
    frame_count = 0
    displayed = None
    challenge_label: str | None = None
    challenge_count = 0
    hold_counter = 0
    max_hold_frames = 8
    movement_history: deque[float] = deque(maxlen=5)
    min_display_frames = 5
    display_age = 0
    # ── gesture timeout / cooldown (prevents infinite sticking) ─
    GESTURE_MAX_AGE = 100
    GESTURE_COOLDOWN = 25
    gesture_cooldown = 0
    # ── sensor staleness tracking ─────────────────────────────────
    _sensor_signature: dict[str, int] = {}    # sensor name -> id(reading)
    _stale_frame_count: dict[str, int] = {}   # consecutive frames with same sig
    _SENSOR_STALE_LIMIT = 30                  # frames of identical object before forcing idle
    # Observation mode state (boxing type discrimination)
    observe_until: int | None = None
    punch_count = 0
    prev_accel: list[float] | None = None
    was_in_punch = False
    punch_cooldown = 0

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

            # ── sensor staleness check ──────────────────────────────
            # Uses reading *object identity* (id()) so that freshly-constructed
            # Reading objects — even with identical data content — are NOT
            # flagged as stale.  This matters for sensors like mmWave which
            # create a new Reading for every read() call, even when no
            # targets are detected.  Staleness should only catch readers
            # that return the *exact same object* (e.g. IMU returning
            # _last_reading when the device disconnects).
            force_idle = False
            if frame_count > args.window * 2:
                for sname in sensor_types:
                    reading = frame_data.get(sname)
                    if reading is None:
                        continue
                    sig = id(reading)
                    prev = _sensor_signature.get(sname)
                    if prev is not None and sig == prev:
                        _stale_frame_count[sname] = _stale_frame_count.get(sname, 0) + 1
                    else:
                        _stale_frame_count[sname] = 0
                    _sensor_signature[sname] = sig
                # Only force idle when EVERY sensor has been returning
                # the exact same Reading object for many frames.
                # Increased limit from 15 → 30 to avoid false positives
                # during brief sensor dropouts.
                stale_sensors = sum(
                    1 for s in sensor_types
                    if _stale_frame_count.get(s, 0) >= _SENSOR_STALE_LIMIT
                )
                if stale_sensors == len(sensor_types) and stale_sensors > 0:
                    if args.debug:
                        print(f"  [stale] ALL sensors stale for >= {_SENSOR_STALE_LIMIT} frames — forcing idle")
                        for s in sensor_types:
                            print(f"          {s}: stale_count={_stale_frame_count.get(s,0)}")
                    force_idle = True

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

            # ── sensor staleness override ────────────────────────────
            if force_idle:
                is_idle = True

            # ── --no-idle override (debugging) ────────────────────────
            if args.no_idle:
                is_idle = False

            # ── gyro oscillation override ──────────────────────────────
            # Soli (finger rub) produces minimal wrist accel but distinctive
            # oscillatory gyro.  Override idle so the model gets to classify it.
            if is_idle:
                imu_readings = _gather_readings(window, "imu")
                if imu_readings and _check_gyro_oscillation(imu_readings):
                    is_idle = False

            # ── debug readings ────────────────────────────────────────
            if args.debug_readings and frame_count % args.print_readings_every == 0:
                for name in args.sensors:
                    readings = _gather_readings(window, name)
                    if readings:
                        last = readings[-1]
                        if sensor_types[name] == "imu":
                            a = last.data.get("accel", [0,0,0])
                            g = last.data.get("gyro", [0,0,0])
                            print(f"  [reading] {name}: accel=({a[0]:.3f},{a[1]:.3f},{a[2]:.3f}) "
                                  f"gyro=({g[0]:.3f},{g[1]:.3f},{g[2]:.3f})")
                        elif sensor_types[name] == "mmwave":
                            pts = last.data.get("points", [])
                            print(f"  [reading] {name}: num_points={len(pts)}")
                print(f"  [reading] movement={movement:.4f} (τ={args.idle_threshold:.2f}) is_idle={is_idle}")

            # ── boxing observation: count punches, determine type ──
            if observe_until is not None and current_accel is not None and prev_accel is not None:
                delta = sum(abs(current_accel[i] - prev_accel[i]) for i in range(3))
                in_punch = delta > args.punch_threshold
                if args.debug:
                    print(f"  [obs] punch={punch_count}, delta={delta:.3f} (th={args.punch_threshold}){' ⚡' if in_punch else ''}")
                if punch_cooldown > 0:
                    punch_cooldown -= 1
                elif in_punch and not was_in_punch:
                    punch_count += 1
                    punch_cooldown = args.punch_cooldown
                    was_in_punch = in_punch

            if current_accel is not None:
                prev_accel = current_accel[:]

            if observe_until is not None:
                if frame_count >= observe_until:
                    if punch_count >= 2:
                        displayed = "one-arm-boxing"
                        print("> one-arm-boxing")
                    else:
                        displayed = "two-arm-boxing"
                        print("> two-arm-boxing")
                    display_age = 0
                    challenge_count = 0
                    challenge_label = None
                    hold_counter = max_hold_frames
                    # reset observation state
                    observe_until = None
                    punch_count = 0
                    was_in_punch = False
                    punch_cooldown = 0
                    gesture_cooldown = 0
                    time.sleep(0.02)
                    continue
                else:
                    if args.debug:
                        print(f"  [obs] awaiting — punch={punch_count}")
                    time.sleep(0.02)
                    continue

            # ── predict (BEFORE idle check so raw predictions are visible) ──
            try:
                label, conf = _predict(pipeline, gestures, features)
            except Exception as e:
                print(f"  ⚠ Prediction error: {e}")
                time.sleep(0.02)
                continue

            # ── raw predictions (always print, even when idle) ───────
            if args.raw_predictions:
                print(f"  [raw] {label}  (conf={conf:.3f})  movement={movement:.4f}  is_idle={is_idle}")

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
                        challenge_count = 0
                        challenge_label = None
                        gesture_cooldown = GESTURE_COOLDOWN
                else:
                    smooth_buffer.clear()
                    challenge_count = 0
                    challenge_label = None
                continue

            hold_counter = max_hold_frames

            # ── gesture max-age timeout (prevents infinite sticking) ──
            if displayed is not None and display_age >= GESTURE_MAX_AGE:
                print(f"  -> gesture timeout ({display_age} frames)")
                displayed = None
                smooth_buffer.clear()
                challenge_count = 0
                challenge_label = None
                gesture_cooldown = GESTURE_COOLDOWN
                continue

            # ── gesture cooldown (wait before re-displaying) ─────────
            if gesture_cooldown > 0:
                gesture_cooldown -= 1
                time.sleep(0.02)
                continue

            if conf < args.min_conf:
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
                    if smoothed in ("one-arm-boxing", "two-arm-boxing"):
                        if movement < args.min_boxing_movement:
                            if args.debug:
                                print(f"  ⚠ ignoring boxing — low movement ({movement:.3f} < {args.min_boxing_movement})")
                            # Fall through: show the model's prediction anyway
                            print(f"> {smoothed}  (conf={conf:.2f}, low mvmt)")
                            displayed = smoothed
                            display_age = 0
                            challenge_count = 0
                            challenge_label = None
                            hold_counter = max_hold_frames
                        else:
                            observe_until = frame_count + args.boxing_delay_frames
                            punch_count = 0
                            was_in_punch = False
                            punch_cooldown = 0
                            print(f"  observing for punches ({args.boxing_delay_frames} frames)...")
                    else:
                        print(f"> {smoothed}  (conf={conf:.2f})")
                        displayed = smoothed
                        display_age = 0
                        challenge_count = 0
                        challenge_label = None
                        hold_counter = max_hold_frames
                        gesture_cooldown = 0  # fresh gesture, reset cooldown

            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\n\nDemo stopped.")
    finally:
        for reader in reader_map.values():
            reader.stop()
    print(f"Frames: {frame_count}")


def run_gui(args, pipeline, expected_n_features, gestures, reader_map, sensor_types):
    """Clean realtime GUI — resizable, fullscreen, no-nonsense."""
    try:
        import tkinter as tk
    except ImportError:
        print("--gui requires tkinter (install python-tk)")
        return

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
    # GESTURE_TIMEOUT — prevents infinite gesture sticking
    GESTURE_MAX_AGE = 100          # auto-dismiss after ~2.5s (at 25ms/frame)
    GESTURE_COOLDOWN = 25          # wait ~0.6s before re-displaying a gesture
    gesture_cooldown = 0
    # Observation mode state (boxing type discrimination)
    observe_until: int | None = None
    punch_count = 0
    prev_accel: list[float] | None = None
    was_in_punch = False
    punch_cooldown = 0
    active_start: float | None = None
    total_active_time = 0.0
    fps_samples: deque[float] = deque(maxlen=60)
    last_fps_time: float | None = None
    idle_start: float | None = None
    # ── gesture log (timestamp, gesture, confidence) ───────────────
    gesture_log: deque[dict] = deque(maxlen=200)
    last_conf: float = 0.0
    _session_start_time = time.time()
    # ── sensor staleness tracking ─────────────────────────────────
    _sensor_signature: dict[str, int] = {}    # sensor name -> id(reading)
    _stale_frame_count: dict[str, int] = {}   # consecutive frames with same sig
    _SENSOR_STALE_LIMIT = 30                  # frames of identical object before forcing idle

    # ── colour palette (dark) ───────────────────────────────────────
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
        "mono": ("Courier", 11),
        "mono_small": ("Courier", 10),
        "mono_tiny": ("Courier", 9),
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

    tk.Label(top, text="● REAL-TIME GESTURE CLASSIFIER", font=("Helvetica", 9, "bold"),
             fg=PALETTE["accent"], bg=PALETTE["bg"]).pack(side="left")
    tk.Label(top, text="v1",
             font=("Helvetica", 8), fg=PALETTE["fg_dim"], bg=PALETTE["bg"]).pack(side="right")

    # thin rule
    rule = tk.Frame(root, height=1, bg=PALETTE["fg_muted"])
    rule.pack(fill="x", padx=16, pady=(4, 8))

    # ── content area: main + sidebar ──────────────────────────────
    content = tk.Frame(root, bg=PALETTE["bg"])
    content.pack(fill="both", expand=True)
    content.grid_columnconfigure(0, weight=1)
    content.grid_columnconfigure(1, weight=0)
    content.grid_rowconfigure(0, weight=1)

    # ── main content area (left, centered column) ──────────────────
    main = tk.Frame(content, bg=PALETTE["bg"])
    main.grid(row=0, column=0, sticky="nsew")

    # weight so the gesture label can expand into available space
    main.grid_rowconfigure(0, weight=1)
    main.grid_rowconfigure(1, weight=0)
    main.grid_rowconfigure(2, weight=0)
    main.grid_columnconfigure(0, weight=1)

    # ═════════════════════════════════════════════════════════════════
    # SIDEBAR — Gesture History Log
    # ═════════════════════════════════════════════════════════════════
    sidebar = tk.Frame(content, bg=PALETTE["bg"], width=280)
    sidebar.grid(row=0, column=1, sticky="ns", padx=(0, 0))
    sidebar.grid_propagate(False)  # enforce width

    # left border for sidebar
    tk.Frame(sidebar, width=1, bg=PALETTE["fg_muted"]).pack(fill="y", side="left")

    # sidebar header
    sidebar_header = tk.Frame(sidebar, bg=PALETTE["bg"])
    sidebar_header.pack(fill="x", padx=8, pady=(8, 2))
    tk.Label(sidebar_header, text="GESTURE LOG", font=("Helvetica", 8, "bold"),
             fg=PALETTE["accent"], bg=PALETTE["bg"]).pack(side="left")
    tk.Label(sidebar_header, text=f"max {gesture_log.maxlen}", font=("Helvetica", 7),
             fg=PALETTE["fg_muted"], bg=PALETTE["bg"]).pack(side="right")

    # thin rule under header
    tk.Frame(sidebar, height=1, bg=PALETTE["fg_muted"]).pack(fill="x", padx=8, pady=(2, 4))

    # scrollable log area
    log_canvas = tk.Canvas(sidebar, bg=PALETTE["bg"], highlightthickness=0)
    log_scroll = tk.Scrollbar(sidebar, orient="vertical", command=log_canvas.yview)
    log_canvas.configure(yscrollcommand=log_scroll.set)

    log_canvas.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=(0, 8))
    log_scroll.pack(side="right", fill="y", pady=(0, 8))

    log_frame = tk.Frame(log_canvas, bg=PALETTE["bg"])
    log_canvas.create_window((0, 0), window=log_frame, anchor="nw", tags="inner")

    def _configure_log_inner(event=None):
        log_canvas.itemconfigure("inner", width=log_canvas.winfo_width())
    log_canvas.bind("<Configure>", _configure_log_inner)

    # Pre-populate with a placeholder until first gesture arrives
    placeholder = tk.Label(log_frame, text="[no gestures yet]", font=FONTS["mono_tiny"],
                           fg=PALETTE["fg_muted"], bg=PALETTE["bg"])
    placeholder.pack(padx=4, pady=2)

    def _add_log_entry(gesture_name: str, confidence: float):
        """Append a gesture event to the sidebar log and auto-scroll to show latest."""
        now = time.time()
        entry = {"time": now, "gesture": gesture_name, "confidence": confidence}
        gesture_log.append(entry)

        # Remove placeholder on first entry
        if placeholder.winfo_exists():
            placeholder.destroy()

        # Build row
        row = tk.Frame(log_frame, bg=PALETTE["bg"])
        row.pack(fill="x", padx=4, pady=(1, 1))

        # Timestamp (seconds elapsed since session start)
        elapsed = now - _session_start_time
        ts_lbl = tk.Label(row, text=f"+{elapsed:5.1f}s", font=FONTS["mono_tiny"],
                          fg=PALETTE["fg_muted"], bg=PALETTE["bg"], width=7, anchor="e")
        ts_lbl.pack(side="left", padx=(0, 4))

        # Gesture name
        g_lbl = tk.Label(row, text=gesture_name.upper(), font=("Courier", 8, "bold"),
                         fg=PALETTE["fg"], bg=PALETTE["bg"], anchor="w")
        g_lbl.pack(side="left", fill="x", expand=True)

        # Confidence bar (max 50px wide, colored by threshold)
        bar_frame = tk.Frame(row, bg=PALETTE["bg"], width=50, height=10)
        bar_frame.pack(side="right", padx=(4, 0))
        bar_frame.pack_propagate(False)
        conf_pct = confidence * 100
        bar_w = max(1, int(50 * min(confidence, 1.0)))
        bar_color = PALETTE["success"] if confidence >= 0.8 else (PALETTE["warn"] if confidence >= 0.5 else PALETTE["fg_dim"])
        inner = tk.Frame(bar_frame, bg=bar_color, width=bar_w, height=10)
        inner.pack(side="left")
        tk.Frame(bar_frame, bg=PALETTE["bg"], width=50 - bar_w, height=10).pack(side="left")

        # Confidence percentage label
        tk.Label(row, text=f"{conf_pct:.0f}%", font=FONTS["mono_tiny"],
                 fg=bar_color, bg=PALETTE["bg"], width=3, anchor="e").pack(side="right")

        # Auto-scroll to bottom so the most recent entry is always visible
        log_canvas.yview_moveto(1.0)



    # ═════════════════════════════════════════════════════════════════
    # GESTURE PREDICTION (centred, expands)
    # ═════════════════════════════════════════════════════════════════

    gesture_label = tk.Label(main, text="—", font=("Helvetica", 80, "bold"),
                             fg="#ffffff", bg=PALETTE["bg"])
    gesture_label.grid(row=0, column=0, sticky="nsew")

    # ── movement meter ───────────────────────────────────────────────
    move_frame = tk.Frame(main, bg=PALETTE["bg"])
    move_frame.grid(row=1, column=0, pady=(0, 8), padx=30, sticky="ew")
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

    stats_labels: dict[str, tk.Label] = {}
    common_stats = [
        ("frames", "frames"),
        ("active", "active"),
        ("fps", "fps"),
        ("idle", "idle"),
        ("conf", "conf"),
        ("sensor", "sensor"),
        ("mode", "mode"),
    ]
    for key, text in common_stats:
        lbl = tk.Label(stats_bar, text=f"{text}  —", font=FONTS["mono_tiny"],
                       fg=PALETTE["fg_dim"], bg=PALETTE["bg"])
        lbl.pack(side="left", padx=(0, 12))
        stats_labels[key] = lbl

    # ── sensor-specific status labels (one per active sensor) ────────
    # Created here but updated each frame; empty if no sensors active.
    _sensor_status_labels: dict[str, tk.Label] = {}
    for name, stype in sensor_types.items():
        prefix = stype[:3]
        lbl = tk.Label(stats_bar, text=f"{prefix}  ○ —", font=FONTS["mono_tiny"],
                       fg=PALETTE["fg_dim"], bg=PALETTE["bg"])
        lbl.pack(side="left", padx=(0, 12))
        _sensor_status_labels[name] = lbl

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
        nonlocal observe_until, punch_count, prev_accel, was_in_punch, punch_cooldown
        nonlocal active_start, total_active_time
        nonlocal fps_samples, last_fps_time, idle_start, last_conf
        nonlocal gesture_cooldown, _sensor_signature, _stale_frame_count

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
                    # ── sensor staleness check ──────────────────────────
                # Uses reading *object identity* (id()) so that freshly-
                # constructed Reading objects are NOT flagged as stale
                # even when data content is identical.
                force_idle = False
                if frame_count > args.window * 2:
                    for sname in sensor_types:
                        reading = frame_data.get(sname)
                        if reading is None:
                            continue
                        sig = id(reading)
                        prev = _sensor_signature.get(sname)
                        if prev is not None and sig == prev:
                            _stale_frame_count[sname] = _stale_frame_count.get(sname, 0) + 1
                        else:
                            _stale_frame_count[sname] = 0
                        _sensor_signature[sname] = sig
                    # Only force idle when ALL sensors return the exact
                    # same Reading object for many frames.
                    stale_sensors = sum(
                        1 for s in sensor_types
                        if _stale_frame_count.get(s, 0) >= _SENSOR_STALE_LIMIT
                    )
                    if stale_sensors == len(sensor_types) and stale_sensors > 0:
                        force_idle = True

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

                # ── sensor staleness override ────────────────────────
                # When all sensors return identical data for many frames
                # (disconnected / stale), force idle to prevent lock-on.
                if force_idle:
                    is_idle = True

                # ── --no-idle override (debugging) ────────────────────
                if args.no_idle:
                    is_idle = False

                # ── gyro oscillation override ────────────────────────
                # Soli (finger rub) produces minimal wrist accel but
                # distinctive oscillatory gyro. Override idle so the model
                # gets to classify it.
                if is_idle:
                    imu_readings = _gather_readings(window, "imu")
                    if imu_readings and _check_gyro_oscillation(imu_readings):
                        is_idle = False

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

                # ── boxing observation: count punches, determine type ──
                if observe_until is not None and current_accel is not None and prev_accel is not None:
                    delta = sum(abs(current_accel[i] - prev_accel[i]) for i in range(3))
                    in_punch = delta > args.punch_threshold
                    if punch_cooldown > 0:
                        punch_cooldown -= 1
                    elif in_punch and not was_in_punch:
                        punch_count += 1
                        punch_cooldown = args.punch_cooldown
                    was_in_punch = in_punch

                if current_accel is not None:
                    prev_accel = current_accel[:]

                if observe_until is not None:
                    if frame_count >= observe_until:
                        if punch_count >= 2:
                            display_text = "ONE-ARM-BOXING"
                            displayed = "one-arm-boxing"
                        else:
                            display_text = "TWO-ARM-BOXING"
                            displayed = "two-arm-boxing"
                        gesture_label.config(text=display_text, fg=PALETTE["success"])
                        _add_log_entry(displayed, last_conf)
                        display_age = 0
                        challenge_count = 0
                        challenge_label = None
                        hold_counter = max_hold_frames
                        error_label.config(text="")
                        # reset observation state
                        observe_until = None
                        punch_count = 0
                        was_in_punch = False
                        punch_cooldown = 0
                        root.after(25, poll)
                        return
                    else:
                        gesture_label.config(fg=PALETTE["warn"])
                        error_label.config(text=f"observing... {punch_count} punches", fg=PALETTE["fg_dim"])
                        root.after(25, poll)
                        return

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
                            challenge_count = 0
                            challenge_label = None
                            error_label.config(text="")
                            gesture_cooldown = GESTURE_COOLDOWN
                    else:
                        smooth_buffer.clear()
                        challenge_count = 0
                        challenge_label = None
                    root.after(25, poll)
                    return

                hold_counter = max_hold_frames

                # ── gesture max-age timeout (prevents infinite sticking) ──
                if displayed is not None and display_age >= GESTURE_MAX_AGE:
                    gesture_label.config(text="—", fg="#ffffff")
                    displayed = None
                    smooth_buffer.clear()
                    challenge_count = 0
                    challenge_label = None
                    error_label.config(text="")
                    gesture_cooldown = GESTURE_COOLDOWN
                    root.after(25, poll)
                    return

                # ── gesture cooldown (wait before re-displaying) ─────────
                if gesture_cooldown > 0:
                    gesture_cooldown -= 1
                    root.after(25, poll)
                    return

                # ── predict ──────────────────────────────────────────
                try:
                    label, conf = _predict(pipeline, gestures, features)
                except Exception as e:
                    error_label.config(text=f"⚠ prediction error: {e}", fg=PALETTE["warn"])
                    root.after(30, poll)
                    return
                last_conf = conf

                if conf >= args.min_conf:
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
                        if smoothed in ("one-arm-boxing", "two-arm-boxing"):
                            if movement < args.min_boxing_movement:
                                error_label.config(text=f"{smoothed} (low movement: {movement:.2f})", fg=PALETTE["warn"])
                                display_upper = smoothed.upper()
                                gesture_label.config(text=display_upper, fg=PALETTE["warn"])
                                _add_log_entry(smoothed, conf)
                                displayed = smoothed
                                display_age = 0
                                challenge_count = 0
                                challenge_label = None
                                hold_counter = max_hold_frames
                            else:
                                observe_until = frame_count + args.boxing_delay_frames
                                punch_count = 0
                                was_in_punch = False
                                punch_cooldown = 0
                                gesture_label.config(fg=PALETTE["warn"])
                                error_label.config(text=f"observing for punches ({args.boxing_delay_frames} frames)...", fg=PALETTE["fg_dim"])
                        else:
                            display_upper = smoothed.upper()
                            gesture_label.config(text=display_upper, fg="#ffffff")
                            _add_log_entry(smoothed, conf)
                            displayed = smoothed
                            display_age = 0
                            challenge_count = 0
                            challenge_label = None
                            hold_counter = max_hold_frames
                            gesture_cooldown = 0  # fresh gesture, reset cooldown
                            error_label.config(text="")

            # ── update stats bar sensor info ─────────────────────────
            stats_labels["sensor"].config(text=f"sensor  {', '.join(args.sensors)}")
            stats_labels["mode"].config(text=f"mode  {args.mode}")

            # ── live sensor status readout ───────────────────────────
            for name, stype in sensor_types.items():
                lbl = _sensor_status_labels.get(name)
                if lbl is None:
                    continue
                reading = frame_data.get(name)
                if reading is None or not reading.data:
                    lbl.config(text=f"{stype[:3]}  ○ connecting…", fg=PALETTE["fg_dim"])
                    continue

                if stype == "imu":
                    accel = reading.data.get("accel", None)
                    gyro = reading.data.get("gyro", None)
                    if accel is not None:
                        accel_str = f"a({accel[0]:+.2f},{accel[1]:+.2f},{accel[2]:+.2f})"
                        gyro_str = f"g({gyro[0]:+.1f},{gyro[1]:+.1f},{gyro[2]:+.1f})" if gyro else ""
                        lbl.config(
                            text=f"imu  ● {accel_str} {gyro_str}",
                            fg=PALETTE["success"],
                        )
                    else:
                        lbl.config(text="imu  ○ no data", fg=PALETTE["fg_muted"])

                elif stype == "mmwave":
                    npts = reading.data.get("num_points")
                    if npts is not None:
                        lbl.config(
                            text=f"mmw  ● {npts}pts",
                            fg=PALETTE["success"],
                        )
                    else:
                        lbl.config(text="mmw  ○ no detect", fg=PALETTE["warn"])

                elif stype == "uwb":
                    ranges = reading.data.get("ranges_cm", reading.data.get("raw_ranges", []))
                    valid = [r for r in ranges if isinstance(r, (int, float)) and r > 0]
                    if valid:
                        lbl.config(
                            text=f"{name}  ● {len(valid)}tg {min(valid):.0f}cm",
                            fg=PALETTE["success"],
                        )
                    else:
                        lbl.config(text=f"{name}  ○ no tags", fg=PALETTE["fg_muted"])

            # ── per-frame confidence ─────────────────────────────────
            # Show the latest raw prediction confidence (before smoothing)
            if last_conf > 0:
                conf_color = PALETTE["success"] if last_conf >= 0.8 else (PALETTE["warn"] if last_conf >= 0.5 else PALETTE["fg_dim"])
                stats_labels["conf"].config(text=f"conf  {last_conf*100:.0f}%", fg=conf_color)

        except Exception as e:
            error_label.config(text=f"⚠ {e}", fg=PALETTE["warn"])

        root.after(25, poll)

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
        for reader in reader_map.values():
            reader.stop()


def _find_latest_model(models_dir: str = "models", pattern: str = "best_model.pkl") -> str:
    candidates = sorted(Path(models_dir).glob(f"train_*/{pattern}"))
    return str(candidates[-1]) if candidates else f"models/{pattern}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time gesture classification demo")
    parser.add_argument("--model", default=_find_latest_model(),
                        help="Path to trained model pickle (default: latest train_*/best_model.pkl)")
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
    parser.add_argument("--window", type=int, default=5,
                        help="Window size (matches training)")
    parser.add_argument("--idle-threshold", type=float, default=0.12,
                        help="Movement score below this = idle (default: 0.12)")
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
    parser.add_argument("--accel-gain", type=float, default=1.05,
                        help="Scale factor for accelerometer values (default: 1.05 — tiny emphasis on linear movement)")
    parser.add_argument("--gyro-gain", type=float, default=1.0,
                        help="Scale factor for gyro values before model (default: 1.0)")
    parser.add_argument("--gyro-deadband", type=float, default=0.0,
                        help="Gyro deadband in dps — values below this are zeroed out (default: 0.0 — no deadband)")
    parser.add_argument("--debug", action="store_true",
                        help="Print per-frame predictions")
    parser.add_argument("--mmwave-port", default="/dev/cu.usbserial-BH00LUQT",
                        help="mmWave serial port (default: /dev/cu.usbserial-BH00LUQT)")
    parser.add_argument("--mmwave-cfg", default="config/point_cloud.cfg",
                        help="mmWave config file path (default: config/point_cloud.cfg)")
    parser.add_argument("--imu-port", default=None,
                        help="IMU serial port")
    parser.add_argument("--imu-baud", type=int, default=115200,
                        help="IMU serial baud rate")
    parser.add_argument("--no-idle", action="store_true",
                        help="Disable idle detection — always run prediction")
    parser.add_argument("--raw-predictions", action="store_true",
                        help="Print ALL raw predictions to terminal, regardless of confidence")
    parser.add_argument("--debug-readings", action="store_true",
                        help="Print raw sensor readings and movement score every N frames")
    parser.add_argument("--print-readings-every", type=int, default=30,
                        help="How often to print raw readings with --debug-readings (default: 30 frames)")
    args = parser.parse_args()

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

    # ── auto-detect sensor order from model's feature_names ───────────
    model_feature_names = raw.get("feature_names", [])
    if model_feature_names and args.sensors:
        # Detect which sensors were used during training from feature names
        trained_sensors = []
        imu_count = sum(1 for fn in model_feature_names if fn.startswith("imu_"))
        mm_count = sum(1 for fn in model_feature_names if fn.startswith("mm_"))
        uwb_count = sum(1 for fn in model_feature_names if fn.startswith("uwb") or "uwb_" in fn)
        if mm_count > 0:
            trained_sensors.append("mmwave")
        if imu_count > 0:
            trained_sensors.append("imu")
        if uwb_count > 0:
            trained_sensors.append("uwb")
        if trained_sensors and trained_sensors != args.sensors:
            print(f"  ⚠ Auto-overriding --sensors from {args.sensors} to {trained_sensors}")
            print(f"     (model was trained with {trained_sensors}; "
                  f"add --sensors to override)")
            args.sensors = trained_sensors

    # ── auto-detect window size from model metadata or feature count ──
    train_params = raw.get("train_params", {})
    model_window = train_params.get("window_size", None)
    if model_window is not None and model_window != args.window:
        original_w = args.window
        args.window = model_window
        print(f"  ⚠ Window size mismatch: model was trained with window={model_window}, "
              f"but --window={original_w} was specified.")
        print(f"     Auto-overriding to --window {model_window} to match the model.")
    else:
        inferred_w = _infer_window_size(expected_n_features, args.sensors, model_feature_names)
        if inferred_w is not None and inferred_w != args.window:
            original_w = args.window
            args.window = inferred_w
            print(f"  ⚠ Window size mismatch: model was trained with window={inferred_w}, "
                  f"but --window={original_w} was specified.")
            print(f"     Auto-overriding to --window {inferred_w} to match the model.")
        elif inferred_w is None:
            print(f"  ⚠ Could not auto-detect window size from model features.")
            print(f"     Make sure --window matches the value used during training.")
            print(f"     Model expects {expected_n_features} features, but the current setup "
                  f"(sensors={args.sensors}) would produce different dimensions.")
            print(f"     Consider using the latest model at models/train_all_data/best_model.pkl")

    print(f"Loaded: {model_path}")
    if gestures_list:
        print(f"Gestures: {', '.join(gestures_list)}")
    print(f"Features expected by model: {expected_n_features}")
    if model_feature_names:
        n_mm = sum(1 for fn in model_feature_names if fn.startswith('mm_'))
        n_imu = sum(1 for fn in model_feature_names if fn.startswith('imu_'))
        print(f"  Sensor breakdown: mmwave={n_mm} feats, imu={n_imu} feats")
        extra = [fn for fn in model_feature_names if 'spec_' in fn or 'corr_' in fn]
        if extra:
            print(f"  ⚠ Model has {len(extra)} extra features (spectral/correlation) that")
            print(f"     the runtime may not produce. Consider retraining with the current feature set.")
    print(f"Window: {args.window}, threshold: {args.idle_threshold}")
    if len(args.sensors) > 1:
        print(f"  Sensor order: {', '.join(args.sensors)} — features are concatenated in this order.")
        print(f"     Must match the order used during training (typically mmwave first, then imu).")
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
        elif name == "mmwave":
            reader = cls(mode=args.mode, serial_port=args.mmwave_port, cfg_path=args.mmwave_cfg)
            reader.start()
            reader_map[name] = reader
            sensor_types[name] = reader.sensor_type
            print(f"  Started {name} reader ({args.mode} mode, port={args.mmwave_port})")
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

    if args.gui:
        run_gui(args, pipeline, expected_n_features, gestures_list, reader_map, sensor_types)
    else:
        run_terminal(args, pipeline, expected_n_features, gestures_list, reader_map, sensor_types)


if __name__ == "__main__":
    main()
