from __future__ import annotations

import argparse
import pickle
import time
from collections import deque
from pathlib import Path

import numpy as np

from src.sensors.imu_reader import ImuReader
from src.sensors.mmWave.mmwave_reader import MmWaveReader

SENSOR_REGISTRY = {
    "mmwave": MmWaveReader,
    "imu": ImuReader,
}


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
        return [
            float(accel[0]) if len(accel) > 0 else 0.0,
            float(accel[1]) if len(accel) > 1 else 0.0,
            float(accel[2]) if len(accel) > 2 else 0.0,
            float(gyro[0]) if len(gyro) > 0 else 0.0,
            float(gyro[1]) if len(gyro) > 1 else 0.0,
            float(gyro[2]) if len(gyro) > 2 else 0.0,
        ]
    return []


def _movement_score(readings: list, sensor_type: str) -> float:
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


def _gather_readings(window: list[dict], name: str) -> list:
    return [f[name] for f in window if name in f]


def _compute_features(readings: list, sensor_type: str) -> list[float]:
    sensor_feats = np.array([extract_features_from_reading(r, sensor_type) for r in readings])
    if len(sensor_feats) == 0:
        n = (6 + 2) + (6 + 2) + ((3 + 3 + 2) * (len(readings) - 1)) if sensor_type == "imu" else 13
        return [0.0] * n

    if sensor_type == "imu":
        # Augment base 6-channel vector with gyro_mag and accel_mag
        accel = sensor_feats[:, 0:3]
        gyro = sensor_feats[:, 3:6]
        gyro_mag = np.sqrt(np.sum(gyro**2, axis=1, keepdims=True))
        accel_mag = np.sqrt(np.sum(accel**2, axis=1, keepdims=True))
        feats = np.concatenate([sensor_feats, gyro_mag, accel_mag], axis=1)  # (N, 8)

        out = list(np.mean(feats, axis=0))
        out.extend(np.std(feats, axis=0).tolist())
        # Accel deltas (3 channels, window-1 time steps)
        out.extend((feats[1:, 0:3] - feats[:-1, 0:3]).flatten().tolist())
        # Gyro deltas (3 channels, window-1 time steps)
        out.extend((feats[1:, 3:6] - feats[:-1, 3:6]).flatten().tolist())
        # Magnitude deltas (2 channels, window-1 time steps)
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


def run_terminal(args, pipeline, gestures, reader_map, sensor_types):
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

            raw_movement = 0.0
            features = []
            for name in args.sensors:
                readings = _gather_readings(window, name)
                raw_movement += _movement_score(readings, sensor_types[name])
                features.extend(_compute_features(readings, sensor_types[name]))

            movement_history.append(raw_movement)
            movement = np.mean(movement_history)

            is_idle = movement < args.idle_threshold

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
            else:
                hold_counter = max_hold_frames

                label, conf = _predict(pipeline, gestures, features)
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


def run_gui(args, pipeline, gestures, reader_map, sensor_types):
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError:
        print("--gui requires tkinter (install python-tk)")
        return

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

    root = tk.Tk()
    root.title("Gesture Demo")
    root.geometry("520x420")
    root.configure(bg="#1e1e2e")

    colors = {
        "bg": "#1e1e2e", "fg": "#cdd6f4", "accent": "#89b4fa",
        "success": "#a6e3a1", "warn": "#f9e2af", "surface": "#313244",
    }

    gesture_font = ("Helvetica", 64, "bold")
    conf_font = ("Helvetica", 18)
    status_font = ("Helvetica", 14)
    small_font = ("Helvetica", 11)

    gesture_label = tk.Label(root, text="—", font=gesture_font,
                             fg=colors["accent"], bg=colors["bg"])
    gesture_label.pack(pady=(40, 0))

    conf_label = tk.Label(root, text="", font=conf_font,
                          fg=colors["fg"], bg=colors["bg"])
    conf_label.pack(pady=(5, 0))

    movement_frame = tk.Frame(root, bg=colors["bg"])
    movement_frame.pack(pady=(15, 0), padx=60, fill="x")
    tk.Label(movement_frame, text="Movement:", font=small_font,
             fg=colors["fg"], bg=colors["bg"]).pack(anchor="w")
    movement_bar = ttk.Progressbar(movement_frame, length=400, mode="determinate", maximum=50)
    movement_bar.pack(fill="x", pady=(2, 0))
    movement_text = tk.Label(movement_frame, text="", font=small_font,
                             fg=colors["fg"], bg=colors["bg"], anchor="e")
    movement_text.pack(anchor="e")

    last_frame = tk.Label(root, text="", font=small_font,
                          fg=colors["fg"], bg=colors["bg"])
    last_frame.pack(pady=(15, 0))

    info = tk.Label(root, text="Make a gesture", font=("Helvetica", 10),
                    fg="#6c7086", bg=colors["bg"])
    info.pack(side="bottom", pady=15)

    def poll():
        nonlocal running, frame_count, displayed, challenge_label, challenge_count
        nonlocal hold_counter, movement_history, display_age
        if not running:
            return

        frame_data = {name: reader.read() for name, reader in reader_map.items()}
        frame_count += 1
        frame_buffer.append(frame_data)
        window = list(frame_buffer)

        if len(window) >= args.window:
            raw_movement = 0.0
            features = []
            for name in args.sensors:
                readings = _gather_readings(window, name)
                raw_movement += _movement_score(readings, sensor_types[name])
                features.extend(_compute_features(readings, sensor_types[name]))

            movement_history.append(raw_movement)
            movement = np.mean(movement_history)

            movement_bar["value"] = min(movement, 300)
            movement_text.config(text=f"{movement:.2f}  (th={args.idle_threshold})")

            is_idle = movement < args.idle_threshold

            if is_idle:
                if displayed is not None:
                    if hold_counter > 0:
                        hold_counter -= 1
                        display_age += 1
                    else:
                        gesture_label.config(text="—", fg=colors["accent"])
                        conf_label.config(text="")
                        displayed = None
                        smooth_buffer.clear()
                        challenge_count = 0
                        challenge_label = None
                        info.config(text="Make a gesture")
                else:
                    smooth_buffer.clear()
                    challenge_count = 0
                    challenge_label = None
            else:
                hold_counter = max_hold_frames

                label, conf = _predict(pipeline, gestures, features)
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
                        gesture_label.config(fg=colors["success"])
                        conf_label.config(fg=colors["fg"])
                    elif display_age < min_display_frames:
                        display_age += 1
                    elif smoothed == challenge_label:
                        challenge_count += 1
                    else:
                        challenge_label = smoothed
                        challenge_count = 1

                    if challenge_count >= args.change_frames:
                        gesture_label.config(text=smoothed.upper(), fg=colors["success"])
                        conf_label.config(text=f"conf={conf:.2f}", fg=colors["fg"])
                        displayed = smoothed
                        display_age = 0
                        challenge_count = 0
                        challenge_label = None
                        hold_counter = max_hold_frames
                        info.config(text="")

        last_frame.config(text=f"Frame: {frame_count}")
        root.after(30, poll)

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
    parser.add_argument("--model", default="models/imu_window5_best_model.pkl",
                        help="Path to trained model pickle")
    parser.add_argument("--features", default="data/processed/imu_features_window5.npz",
                        help="Path to features NPZ (for metadata)")
    parser.add_argument("--sensors", nargs="+", default=["mmwave"],
                        choices=list(SENSOR_REGISTRY.keys()),
                        help="Sensors to use")
    parser.add_argument("--mode", default="mock",
                        choices=["mock", "serial"],
                        help="Sensor mode")
    parser.add_argument("--window", type=int, default=5,
                        help="Window size (matches training)")
    parser.add_argument("--idle-threshold", type=float, default=0.2,
                        help="Movement score below this = idle")
    parser.add_argument("--min-conf", type=float, default=0.65,
                        help="Minimum prediction confidence to accept")
    parser.add_argument("--smooth", type=int, default=10,
                        help="Smoothing buffer size (majority vote over last N frames)")
    parser.add_argument("--min-vote", type=int, default=3,
                        help="Minimum frames in buffer before showing prediction")
    parser.add_argument("--change-frames", type=int, default=5,
                        help="Require new label to dominate this many consecutive frames before switching")
    parser.add_argument("--gui", action="store_true",
                        help="Show prediction GUI window")
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

    if not gestures_list:
        features_path = Path(args.features)
        if features_path.exists():
            data = np.load(features_path, allow_pickle=True)
            gestures_list = data["gestures"].tolist() if "gestures" in data else []

    print(f"Loaded: {model_path}")
    if gestures_list:
        print(f"Gestures: {', '.join(gestures_list)}")
    print(f"Window: {args.window}, threshold: {args.idle_threshold}")
    print()

    reader_map = {}
    sensor_types = {}
    for name in args.sensors:
        cls = SENSOR_REGISTRY[name]
        kwargs = {"mode": args.mode}
        if name == "imu" and args.imu_port:
            kwargs["serial_port"] = args.imu_port
        reader = cls(**kwargs)
        reader.start()
        reader_map[name] = reader
        sensor_types[name] = reader.sensor_type
        print(f"Started {name} ({args.mode} mode)")

    if args.gui:
        run_gui(args, pipeline, gestures_list, reader_map, sensor_types)
    else:
        run_terminal(args, pipeline, gestures_list, reader_map, sensor_types)


if __name__ == "__main__":
    main()
