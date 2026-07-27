from __future__ import annotations

import argparse
import pickle
import time
from collections import deque
from pathlib import Path

import numpy as np

from src.sensors.imu_reader import ImuReader
from src.sensors.mmwave_reader import MmWaveReader
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


def extract_features_from_reading(reading: any, sensor_type: str) -> list[float]:
    if sensor_type == "mmwave":
        data = reading.data
        points = data.get("points", [])
        num_points = data.get("num_points", len(points))
        if not points:
            return [num_points, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        xs = np.array([p.get("x", 0) for p in points])
        ys = np.array([p.get("y", 0) for p in points])
        zs = np.array([p.get("z", 0) for p in points])
        vs = np.array([p.get("velocity", 0) for p in points])
        return [
            float(num_points),
            float(np.mean(xs)), float(np.std(xs)),
            float(np.mean(ys)), float(np.std(ys)),
            float(np.mean(zs)), float(np.std(zs)),
            float(np.mean(np.abs(vs))), float(np.std(vs)),
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time gesture classification demo")
    parser.add_argument("--model", default="models/best_model.pkl",
                        help="Path to trained model pickle")
    parser.add_argument("--features", default="data/processed/features.npz",
                        help="Path to features NPZ (for metadata)")
    parser.add_argument("--sensors", nargs="+", default=["mmwave"],
                        choices=list(SENSOR_REGISTRY.keys()),
                        help="Sensors to use")
    parser.add_argument("--mode", default="mock",
                        choices=["mock", "serial"],
                        help="Sensor mode")
    parser.add_argument("--uwb-ports", nargs="+", default=["/dev/ttyACM0"],
                        help="Serial ports for UWB devices")
    parser.add_argument("--window", type=int, default=10,
                        help="Window size (matches training)")
    parser.add_argument("--stride", type=int, default=5,
                        help="Frame stride between predictions")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: model not found: {model_path}. Run train.py first.")
        return

    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)

    features_path = Path(args.features)
    gestures = []
    if features_path.exists():
        data = np.load(features_path, allow_pickle=True)
        gestures = data["gestures"].tolist() if "gestures" in data else []

    print(f"Loaded model: {model_path}")
    if gestures:
        print(f"Known gestures: {', '.join(gestures)}")
    print(f"Window: {args.window}, Stride: {args.stride}")
    print()

    reader_map: dict[str, any] = {}
    sensor_types: dict[str, str] = {}
    reader_keys: list[str] = []
    for name in args.sensors:
        cls = SENSOR_REGISTRY[name]
        if name == "uwb":
            for i, port in enumerate(args.uwb_ports):
                key = f"uwb_{i}"
                reader = cls(mode=args.mode, serial_port=port, sensor_id=f"uwb-{i}")
                reader.start()
                reader_map[key] = reader
                sensor_types[key] = "uwb"
                reader_keys.append(key)
                print(f"  Started {key} reader ({args.mode} mode, port={port})")
        else:
            reader = cls(mode=args.mode)
            reader.start()
            reader_map[name] = reader
            sensor_types[name] = reader.sensor_type
            reader_keys.append(name)
            print(f"  Started {name} reader ({args.mode} mode)")

    frame_buffer: deque[dict] = deque(maxlen=args.window)
    frame_count = 0
    predict_count = 0

    print("\n=== Real-time Demo Started ===")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            frame_data: dict[str, any] = {}
            for name, reader in reader_map.items():
                reading = reader.read()
                frame_data[name] = reading
            frame_count += 1

            frame_buffer.append(frame_data)

            if len(frame_buffer) >= args.window and (
                frame_count % args.stride == 0 or len(frame_buffer) == args.window
            ):
                window = list(frame_buffer)

                features: list[float] = []
                for name in reader_keys:
                    readings = [f[name] for f in window if name in f]
                    sensor_feats = []
                    for r in readings:
                        feats = extract_features_from_reading(r, sensor_types[name])
                        sensor_feats.append(feats)
                    if sensor_feats:
                        sensor_feats = np.array(sensor_feats)
                        features.extend(np.mean(sensor_feats, axis=0).tolist())
                    else:
                        n_feats = SENSOR_FEATURE_COUNTS.get(sensor_types[name], 6)
                        features.extend([0.0] * n_feats)

                if len(features) > 0:
                    X = np.array(features).reshape(1, -1)
                    pred = pipeline.predict(X)[0]
                    if hasattr(pipeline, "predict_proba"):
                        proba = pipeline.predict_proba(X)[0]
                        conf = float(max(proba))
                    else:
                        conf = 0.0

                    predict_count += 1
                    label = gestures[pred] if pred < len(gestures) else str(pred)
                    print(f"  [{predict_count}] Gesture: {label:>15s}  (confidence: {conf:.3f})")

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\nDemo stopped.")
    finally:
        for name, reader in reader_map.items():
            reader.stop()

    print(f"\nTotal frames: {frame_count}, Predictions: {predict_count}")


if __name__ == "__main__":
    main()
