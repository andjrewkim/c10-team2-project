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


def extract_mmwave_features(reading):
    data = reading.data
    pts = data.get("points", [])
    n = data.get("num_points", len(pts))
    if not pts:
        return [float(n), 0.0, 0.0, 0.0, 0.0, 0.0]
    xs = np.array([p.get("x", 0) for p in pts])
    ys = np.array([p.get("y", 0) for p in pts])
    return [
        float(n),
        float(np.mean(xs)), float(np.std(xs)),
        float(np.mean(ys)), float(np.std(ys)),
        float(np.sqrt(np.mean(xs)**2 + np.mean(ys)**2)),
    ]


def extract_imu_features(reading):
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


def mmwave_window_features(window_readings):
    feats = np.array([extract_mmwave_features(r) for r in window_readings])
    result = list(np.mean(feats, axis=0))
    result.extend(list(np.std(feats, axis=0)))
    cents = []
    for r in window_readings:
        data = r.data
        pts = data.get("points", [])
        if pts:
            xs = [p.get("x", 0) for p in pts]
            ys = [p.get("y", 0) for p in pts]
            cents.append((np.mean(xs), np.mean(ys)))
        else:
            cents.append((0.0, 0.0))
    dist = 0.0
    for i in range(1, len(cents)):
        dist += np.sqrt((cents[i][0] - cents[i-1][0])**2 + (cents[i][1] - cents[i-1][1])**2)
    result.append(float(dist))
    return result


def imu_window_features(window_readings):
    feats = np.array([extract_imu_features(r) for r in window_readings])
    result = list(np.mean(feats, axis=0))
    result.extend(list(np.std(feats, axis=0)))
    return result


def _find_latest_model(models_dir: str = "models", pattern: str = "best_model.pkl") -> str:
    candidates = sorted(Path(models_dir).glob(f"train_*/{pattern}"))
    return str(candidates[-1]) if candidates else f"models/{pattern}"


def main() -> None:
    default_mm = _find_latest_model(pattern="mmwave_best_model.pkl")
    default_imu = _find_latest_model(pattern="imu_best_model.pkl")
    parser = argparse.ArgumentParser(description="Fused mmWave + IMU real-time demo")
    parser.add_argument("--mmwave-model", default=default_mm,
                        help="mmWave trained model")
    parser.add_argument("--imu-model", default=default_imu,
                        help="IMU trained model")
    parser.add_argument("--mode", default="mock", choices=["mock", "serial"])
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--idle-threshold", type=float, default=0.5,
                        help="Movement score below this = idle (no prediction)")
    args = parser.parse_args()

    mm_path = Path(args.mmwave_model)
    imu_path = Path(args.imu_model)
    if not mm_path.exists():
        print(f"Error: {mm_path} not found")
        return
    if not imu_path.exists():
        print(f"Error: {imu_path} not found")
        return

    with open(mm_path, "rb") as f:
        mm_data = pickle.load(f)
    mm_pipeline = mm_data["pipeline"] if isinstance(mm_data, dict) else mm_data
    mm_gestures = mm_data.get("gestures", []) if isinstance(mm_data, dict) else []

    with open(imu_path, "rb") as f:
        imu_data = pickle.load(f)
    imu_pipeline = imu_data["pipeline"] if isinstance(imu_data, dict) else imu_data
    imu_gestures = imu_data.get("gestures", []) if isinstance(imu_data, dict) else []

    gestures = mm_gestures if mm_gestures else imu_gestures
    print(f"Loaded models: mmWave ({len(mm_gestures)} classes) + IMU ({len(imu_gestures)} classes)")
    print(f"Gestures: {', '.join(gestures) if gestures else 'unknown'}")
    print(f"Window: {args.window}, Stride: {args.stride}")
    print()

    mm_reader = SENSOR_REGISTRY["mmwave"](mode=args.mode)
    imu_reader = SENSOR_REGISTRY["imu"](mode=args.mode)
    mm_reader.start()
    imu_reader.start()
    print(f"Started mmWave reader ({args.mode} mode)")
    print(f"Started IMU reader ({args.mode} mode)")

    mm_buffer: deque = deque(maxlen=args.window)
    imu_buffer: deque = deque(maxlen=args.window)
    frame_count = 0
    last_label = None

    print(f"Idle threshold: {args.idle_threshold}")
    print("\n=== Fused Real-time Demo Started ===")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            mm_reading = mm_reader.read()
            imu_reading = imu_reader.read()
            frame_count += 1
            mm_buffer.append(mm_reading)
            imu_buffer.append(imu_reading)

            if len(mm_buffer) >= args.window and len(imu_buffer) >= args.window and (
                frame_count % args.stride == 0
            ):
                mm_feats = mmwave_window_features(list(mm_buffer))
                imu_feats = imu_window_features(list(imu_buffer))

                # movement score: mean of std features + path_length contribution
                mm_movement = sum(abs(v) for v in mm_feats[6:12]) / 6 + mm_feats[12] * 0.5
                imu_movement = sum(abs(v) for v in imu_feats[6:12]) / 6
                movement = (mm_movement + imu_movement) / 2

                if movement < args.idle_threshold:
                    if last_label is not None:
                        print(f"  -> idle")
                        last_label = None
                    continue

                mm_proba = mm_pipeline.predict_proba(np.array([mm_feats]))[0]
                imu_proba = imu_pipeline.predict_proba(np.array([imu_feats]))[0]

                fused = (mm_proba + imu_proba) / 2.0
                pred = int(np.argmax(fused))
                conf = float(fused[pred])
                label = gestures[pred] if pred < len(gestures) else str(pred)

                if label == last_label:
                    continue

                mm_pred = int(np.argmax(mm_proba))
                mm_conf = float(mm_proba[mm_pred])
                mm_label = gestures[mm_pred] if mm_pred < len(gestures) else str(mm_pred)

                imu_pred = int(np.argmax(imu_proba))
                imu_conf = float(imu_proba[imu_pred])
                imu_label = gestures[imu_pred] if imu_pred < len(gestures) else str(imu_pred)

                last_label = label
                print(
                    f"  Gesture: {label:>15s} ({conf:.2f})  "
                    f"[mmWave: {mm_label:>15s} ({mm_conf:.2f})  "
                    f"IMU: {imu_label:>15s} ({imu_conf:.2f})]"
                )

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\nDemo stopped.")
    finally:
        mm_reader.stop()
        imu_reader.stop()

    print(f"\nTotal frames: {frame_count}")


if __name__ == "__main__":
    main()
