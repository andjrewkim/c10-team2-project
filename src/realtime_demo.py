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
        n = (6 + 2) + (6 + 2) + ((3 + 3 + 2) * (len(readings) - 1)) if sensor_type == "imu" else 13
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
                    time.sleep(0.02)
                    continue
                else:
                    if args.debug:
                        print(f"  [obs] awaiting — punch={punch_count}")
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
                        challenge_count = 0
                        challenge_label = None
                else:
                    smooth_buffer.clear()
                    challenge_count = 0
                    challenge_label = None
                continue

            hold_counter = max_hold_frames

            try:
                label, conf = _predict(pipeline, gestures, features)
            except Exception as e:
                print(f"  ⚠ Prediction error: {e}")
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

    gesture_label = tk.Label(main, text="—", font=("Helvetica", 80, "bold"),
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
        nonlocal observe_until, punch_count, prev_accel, was_in_punch, punch_cooldown
        nonlocal active_start, total_active_time
        nonlocal fps_samples, last_fps_time, idle_start

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
                    else:
                        smooth_buffer.clear()
                        challenge_count = 0
                        challenge_label = None
                    root.after(25, poll)
                    return

                hold_counter = max_hold_frames

                # ── predict ──────────────────────────────────────────
                try:
                    label, conf = _predict(pipeline, gestures, features)
                except Exception as e:
                    error_label.config(text=f"⚠ prediction error: {e}", fg=PALETTE["warn"])
                    root.after(30, poll)
                    return

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
                            displayed = smoothed
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time gesture classification demo")
    parser.add_argument("--model", default="models/v2/imu_v2_best_model.pkl",
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
    parser.add_argument("--window", type=int, default=5,
                        help="Window size (matches training)")
    parser.add_argument("--idle-threshold", type=float, default=0.45,
                        help="Movement score below this = idle (default: 0.45)")
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
    parser.add_argument("--gyro-gain", type=float, default=0.6,
                        help="Scale factor for gyro values before model (default: 0.6)")
    parser.add_argument("--gyro-deadband", type=float, default=2.0,
                        help="Gyro deadband in dps — values below this are zeroed out (default: 2.0)")
    parser.add_argument("--debug", action="store_true",
                        help="Print per-frame predictions")
    parser.add_argument("--imu-port", default=None,
                        help="IMU serial port")
    parser.add_argument("--imu-baud", type=int, default=115200,
                        help="IMU serial baud rate")
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
    print(f"Loaded: {model_path}")
    if gestures_list:
        print(f"Gestures: {', '.join(gestures_list)}")
    print(f"Features expected by model: {expected_n_features}")
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

    if args.gui:
        run_gui(args, pipeline, expected_n_features, gestures_list, reader_map, sensor_types)
    else:
        run_terminal(args, pipeline, expected_n_features, gestures_list, reader_map, sensor_types)


if __name__ == "__main__":
    main()
