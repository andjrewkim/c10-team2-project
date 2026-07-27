from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


MM_FEATURE_NAMES = [
    "num_points", "mean_x", "std_x", "mean_y", "std_y", "mean_range",
]


def extract_mmwave_features(frame: dict) -> list[float]:
    mm = frame.get("mmwave", {})
    data = mm.get("data", {})
    points = data.get("points", [])
    num_points = data.get("num_points", len(points))

    if not points:
        return [float(num_points), 0.0, 0.0, 0.0, 0.0, 0.0]

    xs = np.array([p.get("x", 0) for p in points])
    ys = np.array([p.get("y", 0) for p in points])

    return [
        float(num_points),
        float(np.mean(xs)),
        float(np.std(xs)),
        float(np.mean(ys)),
        float(np.std(ys)),
        float(np.sqrt(np.mean(xs)**2 + np.mean(ys)**2)),
    ]


def extract_imu_features(frame: dict) -> list[float]:
    im = frame.get("imu", {})
    data = im.get("data", {})
    accel = data.get("accel", [0, 0, 0])
    gyro = data.get("gyro", [0, 0, 0])

    features = [
        float(accel[0]) if len(accel) > 0 else 0.0,
        float(accel[1]) if len(accel) > 1 else 0.0,
        float(accel[2]) if len(accel) > 2 else 0.0,
        float(gyro[0]) if len(gyro) > 0 else 0.0,
        float(gyro[1]) if len(gyro) > 1 else 0.0,
        float(gyro[2]) if len(gyro) > 2 else 0.0,
    ]
    return features


def _total_path_length(frames: list[dict]) -> float:
    cents = []
    for f in frames:
        mm = f.get("mmwave", {}).get("data", {})
        pts = mm.get("points", [])
        if pts:
            xs = [p.get("x", 0) for p in pts]
            ys = [p.get("y", 0) for p in pts]
            cents.append((np.mean(xs), np.mean(ys)))
        else:
            cents.append((0.0, 0.0))
    dist = 0.0
    for i in range(1, len(cents)):
        dist += np.sqrt((cents[i][0] - cents[i-1][0])**2 + (cents[i][1] - cents[i-1][1])**2)
    return float(dist)


def extract_window_features(
    frames: list[dict],
    window_size: int = 10,
    stride: int = 5,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X_list = []
    y_list = []
    feature_names: list[str] = []

    mmwave_sensors = any("mmwave" in f for f in frames)
    imu_sensors = any("imu" in f for f in frames)

    if mmwave_sensors:
        feature_names.extend([f"mm_mean_{n}" for n in MM_FEATURE_NAMES])
        feature_names.extend([f"mm_std_{n}" for n in MM_FEATURE_NAMES])
        feature_names.append("mm_path_length")
    if imu_sensors:
        imu_base = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
        feature_names.extend([f"imu_mean_{b}" for b in imu_base])
        feature_names.extend([f"imu_std_{b}" for b in imu_base])

    for i in range(0, len(frames) - window_size + 1, stride):
        window = frames[i:i + window_size]
        gesture = window[0].get("gesture", "unknown")
        features: list[float] = []

        if mmwave_sensors:
            mm_features = np.array([extract_mmwave_features(f) for f in window])
            for col in range(mm_features.shape[1]):
                features.append(float(np.mean(mm_features[:, col])))
            for col in range(mm_features.shape[1]):
                features.append(float(np.std(mm_features[:, col])))
            features.append(_total_path_length(window))

        if imu_sensors:
            imu_features = np.array([extract_imu_features(f) for f in window])
            for col in range(imu_features.shape[1]):
                features.append(float(np.mean(imu_features[:, col])))
            for col in range(imu_features.shape[1]):
                features.append(float(np.std(imu_features[:, col])))

        X_list.append(features)
        y_list.append(gesture)

    return np.array(X_list), np.array(y_list), feature_names


def load_frames_from_csv(path: Path) -> list[dict]:
    frames = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                points = json.loads(row.get("points", "[]"))
            except json.JSONDecodeError:
                points = []
            frame = {
                "timestamp": row.get("timestamp", ""),
                "gesture": row.get("gesture", "unknown"),
                "trial": int(row.get("trial", 0)),
                "elapsed": float(row.get("elapsed", 0.0)),
                "mmwave": {
                    "data": {
                        "points": [
                            {
                                "x": p.get("x", 0),
                                "y": p.get("y", 0),
                                "z": p.get("z", 0),
                                "velocity": p.get("velocity", 0),
                                "snr": p.get("snr", 0),
                            }
                            for p in points
                        ],
                        "num_points": len(points),
                    },
                    "confidence": float(row.get("confidence", 0.8)),
                    "sensor_type": "mmwave",
                },
            }
            frames.append(frame)
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract features from raw gesture data")
    parser.add_argument("--input", default="data/processed",
                        help="Input directory containing combined_*.csv")
    parser.add_argument("--output", default="data/processed",
                        help="Output directory for feature matrices")
    parser.add_argument("--window", type=int, default=10,
                        help="Sliding window size in frames")
    parser.add_argument("--stride", type=int, default=5,
                        help="Window stride in frames")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Test set proportion")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    input_dir = Path(args.input)
    csv_files = sorted(input_dir.glob("combined_*.csv"))
    if not csv_files:
        print(f"Error: no combined_*.csv files found in {input_dir}")
        return

    csv_path = csv_files[-1]
    frames = load_frames_from_csv(csv_path)
    print(f"Loaded {len(frames)} frames from {csv_path}")

    X, y, feature_names = extract_window_features(
        frames, window_size=args.window, stride=args.stride,
    )
    print(f"\nExtracted {len(X)} windows with {len(feature_names)} features each")
    print(f"Feature names: {feature_names}")

    gestures = sorted(set(y))
    label_map = {g: i for i, g in enumerate(gestures)}
    y_int = np.array([label_map[g] for g in y])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_int, test_size=args.test_size,
        random_state=args.random_state, stratify=y_int,
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = out_dir / "features.npz"
    np.savez_compressed(
        npz_path,
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        feature_names=feature_names,
        gestures=gestures,
        label_map=label_map,
        window_size=args.window,
        stride=args.stride,
    )
    print(f"\nSaved: {npz_path}")
    print(f"  Train: {len(X_train)} windows")
    print(f"  Test:  {len(X_test)} windows")
    print(f"  Classes: {len(gestures)} ({', '.join(gestures)})")


if __name__ == "__main__":
    main()
