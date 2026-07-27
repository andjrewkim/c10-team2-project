from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


def extract_mmwave_features(frame: dict) -> list[float]:
    mm = frame.get("mmwave", {})
    data = mm.get("data", {})
    points = data.get("points", [])
    num_points = data.get("num_points", len(points))

    if not points:
        return [num_points, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    xs = np.array([p.get("x", 0) for p in points])
    ys = np.array([p.get("y", 0) for p in points])
    zs = np.array([p.get("z", 0) for p in points])
    vs = np.array([p.get("velocity", 0) for p in points])

    features = [
        float(num_points),
        float(np.mean(xs)),
        float(np.std(xs)),
        float(np.mean(ys)),
        float(np.std(ys)),
        float(np.mean(zs)),
        float(np.std(zs)),
        float(np.mean(np.abs(vs))),
        float(np.std(vs)),
        float(np.sqrt(np.mean(xs)**2 + np.mean(ys)**2)),
    ]
    return features


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


def extract_uwb_features_from_data(data: dict) -> list[float]:
    ranges = data.get("ranges_cm", [])
    if not ranges:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

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


def _sensor_keys_of_type(frame: dict, sensor_type: str) -> list[str]:
    skip_keys = {"timestamp", "gesture", "trial", "elapsed"}
    return [
        k for k, v in frame.items()
        if k not in skip_keys and isinstance(v, dict) and v.get("sensor_type") == sensor_type
    ]


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
    uwb_keys = _sensor_keys_of_type(frames[0], "uwb") if frames else []

    if mmwave_sensors:
        feature_names.extend([
            "mm_num_points", "mm_mean_x", "mm_std_x", "mm_mean_y", "mm_std_y",
            "mm_mean_z", "mm_std_z", "mm_mean_vel", "mm_std_vel", "mm_mean_range",
        ])
    if imu_sensors:
        feature_names.extend([
            "imu_accel_x", "imu_accel_y", "imu_accel_z",
            "imu_gyro_x", "imu_gyro_y", "imu_gyro_z",
        ])
    for uk in uwb_keys:
        feature_names.extend([
            f"{uk}_num_ranges", f"{uk}_mean_range_m", f"{uk}_std_range_m",
            f"{uk}_min_range_m", f"{uk}_max_range_m", f"{uk}_median_range_m",
        ])

    for i in range(0, len(frames) - window_size + 1, stride):
        window = frames[i:i + window_size]
        gesture = window[0].get("gesture", "unknown")

        features: list[float] = []

        if mmwave_sensors:
            mm_features = np.array([extract_mmwave_features(f) for f in window])
            features.extend([
                float(np.mean(mm_features[:, 0])),
                float(np.mean(mm_features[:, 1])),
                float(np.mean(mm_features[:, 2])),
                float(np.mean(mm_features[:, 3])),
                float(np.mean(mm_features[:, 4])),
                float(np.mean(mm_features[:, 5])),
                float(np.mean(mm_features[:, 6])),
                float(np.mean(mm_features[:, 7])),
                float(np.mean(mm_features[:, 8])),
                float(np.mean(mm_features[:, 9])),
            ])

        if imu_sensors:
            imu_features = np.array([extract_imu_features(f) for f in window])
            for col in range(imu_features.shape[1]):
                features.append(float(np.mean(imu_features[:, col])))

        for uk in uwb_keys:
            uwb_data = [f.get(uk, {}).get("data", {}) for f in window]
            uwb_feats = np.array([extract_uwb_features_from_data(d) for d in uwb_data])
            for col in range(uwb_feats.shape[1]):
                features.append(float(np.mean(uwb_feats[:, col])))

        X_list.append(features)
        y_list.append(gesture)

    return np.array(X_list), np.array(y_list), feature_names


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract features from raw gesture data")
    parser.add_argument("--input", default="data/processed/combined_dataset.json",
                        help="Input combined dataset JSON")
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

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found. Run combine_datasets.py first.")
        return

    with open(input_path) as f:
        dataset = json.load(f)

    frames = dataset["frames"]
    print(f"Loaded {len(frames)} frames from {input_path}")

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
