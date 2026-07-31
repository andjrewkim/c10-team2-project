from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


MM_FEATURE_NAMES = [
    "num_points", "mean_x", "std_x", "min_x", "mean_y", "std_y", "range_profile",
    "distance_from_origin",
]


def extract_mmwave_features(frame: dict) -> list[float]:
    mm = frame.get("mmwave", {})
    data = mm.get("data", {})
    points = data.get("points") or []
    num_points_raw = data.get("num_points")
    num_points = float(num_points_raw) if num_points_raw is not None else float(len(points))
    range_profile = data.get("range_profile") or []

    if not points:
        return [num_points, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    xs = np.array([p.get("x", 0) for p in points])
    ys = np.array([p.get("y", 0) for p in points])

    return [
        num_points,
        float(np.mean(xs)),
        float(np.std(xs)),
        float(np.min(xs)), # closest distance
        float(np.mean(ys)),
        float(np.std(ys)),
        range_profile[0] if range_profile else 0.0, # range profile
        float(np.sqrt(np.mean(xs)**2 + np.mean(ys)**2)), # distance from origin
    ]


def _get_imu_channel(data: dict, name: str) -> float:
    val = data.get(name, None)
    if val is not None:
        return float(val)
    if name in ("accel_x", "accel_y", "accel_z"):
        lst = data.get("accel", [])
        idx = {"accel_x": 0, "accel_y": 1, "accel_z": 2}[name]
        return float(lst[idx]) if isinstance(lst, list) and len(lst) > idx else 0.0
    if name in ("gyro_x", "gyro_y", "gyro_z"):
        lst = data.get("gyro", [])
        idx = {"gyro_x": 0, "gyro_y": 1, "gyro_z": 2}[name]
        return float(lst[idx]) if isinstance(lst, list) and len(lst) > idx else 0.0
    return 0.0


def extract_imu_features(frame: dict) -> list[float]:
    im = frame.get("imu", {})
    data = im.get("data", {})
    ax = _get_imu_channel(data, "accel_x")
    ay = _get_imu_channel(data, "accel_y")
    az = _get_imu_channel(data, "accel_z")
    gx = _get_imu_channel(data, "gyro_x")
    gy = _get_imu_channel(data, "gyro_y")
    gz = _get_imu_channel(data, "gyro_z")

    return [
        ax, ay, az, gx, gy, gz,
        np.sqrt(gx*gx + gy*gy + gz*gz),   # gyro magnitude
        np.sqrt(ax*ax + ay*ay + az*az),   # accel magnitude
    ]


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
    uwb_keys = _sensor_keys_of_type(frames[0], "uwb") if frames else []

    if mmwave_sensors:
        feature_names.extend([f"mm_mean_{n}" for n in MM_FEATURE_NAMES])
        feature_names.extend([f"mm_std_{n}" for n in MM_FEATURE_NAMES])
        feature_names.append("mm_path_length")
    if imu_sensors:
        imu_base = [
            "accel_x", "accel_y", "accel_z",
            "gyro_x", "gyro_y", "gyro_z",
            "gyro_mag", "accel_mag",
        ]
        feature_names.extend([f"imu_mean_{b}" for b in imu_base])
        feature_names.extend([f"imu_std_{b}" for b in imu_base])
        feature_names.extend([f"imu_rms_{b}" for b in imu_base])
        feature_names.extend([f"imu_zcr_{b}" for b in imu_base])
        n_dt = window_size - 1
        feature_names.extend([f"imu_delta_{n}_t{t}" for n in ["accel_x","accel_y","accel_z"] for t in range(n_dt)])
        feature_names.extend([f"imu_delta_{n}_t{t}" for n in ["gyro_x","gyro_y","gyro_z"] for t in range(n_dt)])
        feature_names.extend([f"imu_delta_{n}_t{t}" for n in ["gyro_mag","accel_mag"] for t in range(n_dt)])
    for uk in uwb_keys:
        feature_names.extend([
            f"{uk}_num_ranges", f"{uk}_mean_range_m", f"{uk}_std_range_m",
            f"{uk}_min_range_m", f"{uk}_max_range_m", f"{uk}_median_range_m",
        ])

    ACCEL_IDX = slice(0, 3)
    GYRO_IDX = slice(3, 6)

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
            mm_path_len = _total_path_length(window)
            features.append(mm_path_len)

        if imu_sensors:
            per_frame = np.array([extract_imu_features(f) for f in window])
            features.extend(np.mean(per_frame, axis=0).tolist())
            features.extend(np.std(per_frame, axis=0).tolist())
            features.extend(np.sqrt(np.mean(per_frame ** 2, axis=0)).tolist())
            for c in range(8):
                centered = per_frame[:, c] - np.mean(per_frame[:, c])
                if len(centered) < 2:
                    features.append(0.0)
                else:
                    features.append(float(np.sum((centered[:-1] * centered[1:]) < 0)) / window_size)
            features.extend((per_frame[1:, ACCEL_IDX] - per_frame[:-1, ACCEL_IDX]).flatten().tolist())
            features.extend((per_frame[1:, GYRO_IDX] - per_frame[:-1, GYRO_IDX]).flatten().tolist())
            features.extend((per_frame[1:, 6:8] - per_frame[:-1, 6:8]).flatten().tolist())

        for uk in uwb_keys:
            uwb_data = [f.get(uk, {}).get("data", {}) for f in window]
            uwb_feats = np.array([extract_uwb_features_from_data(d) for d in uwb_data])
            for col in range(uwb_feats.shape[1]):
                features.append(float(np.mean(uwb_feats[:, col])))

        X_list.append(features)
        y_list.append(gesture)

    return np.array(X_list), np.array(y_list), feature_names


def _load_all_frames(input_path: Path) -> list[dict]:
    """Load frames from any supported format (session dir, CSV, JSON, JSONL)."""
    frames: list[dict] = []

    if input_path.is_dir() and (input_path / "events.csv").exists():
        from src.visualize import load_session_csvs
        for trial_frames in load_session_csvs(input_path).values():
            frames.extend(trial_frames)

    elif input_path.is_dir():
        session_dirs = sorted(input_path.glob("*/events.csv"))
        if session_dirs:
            from src.visualize import load_session_csvs
            for ev_csv in session_dirs:
                for trial_frames in load_session_csvs(ev_csv.parent).values():
                    frames.extend(trial_frames)
        else:
            jsonl_files = sorted(input_path.glob("*.jsonl"))
            for f in jsonl_files:
                for line in f.read_text().strip().splitlines():
                    if line.strip():
                        frames.append(json.loads(line))

    elif input_path.suffix == ".csv":
        with open(input_path, newline="") as f:
            reader = csv.DictReader(f)
            all_cols = reader.fieldnames or []
            # Identify sensor prefixes: columns ending with _confidence (excluding base cols)
            base_cols = {"frame_index", "timestamp", "gesture", "trial", "elapsed", "dataset_source"}
            sensor_prefixes = set()
            for col in all_cols:
                if col.endswith("_confidence") and col not in base_cols:
                    sensor_prefixes.add(col[:-len("_confidence")])
            for row in reader:
                frame = {
                    "timestamp": row["timestamp"],
                    "gesture": row["gesture"],
                    "trial": int(row["trial"]),
                    "elapsed": float(row["elapsed"]),
                }
                # Copy extra non-sensor columns (e.g. collector, dataset_source)
                for col in all_cols:
                    if col in base_cols or col.endswith("_confidence"):
                        continue
                    if any(col.startswith(p + "_") for p in sensor_prefixes):
                        continue
                    val = row.get(col, "")
                    if val and val != "null":
                        frame[col] = val
                for prefix in sorted(sensor_prefixes):
                    data = {}
                    for col in all_cols:
                        if col.startswith(prefix + "_") and col != f"{prefix}_confidence":
                            field_name = col[len(prefix) + 1:]
                            val = row.get(col, "")
                            if val == "" or val == "null":
                                data[field_name] = None
                            elif val.startswith(("[", "{")):
                                try:
                                    data[field_name] = json.loads(val)
                                except json.JSONDecodeError:
                                    data[field_name] = val
                            else:
                                try:
                                    data[field_name] = int(val)
                                except ValueError:
                                    try:
                                        data[field_name] = float(val)
                                    except ValueError:
                                        data[field_name] = val
                    confidence = float(row.get(f"{prefix}_confidence", 0.0))
                    frame[prefix] = {
                        "data": data,
                        "confidence": confidence,
                        "sensor_type": prefix,
                    }
                frames.append(frame)

    elif input_path.suffix == ".json":
        data = json.loads(input_path.read_text())
        frames_list = data.get("frames", data if isinstance(data, list) else [])
        frames.extend(frames_list)

    elif input_path.suffix == ".jsonl":
        for line in input_path.read_text().strip().splitlines():
            if line.strip():
                frames.append(json.loads(line))

    else:
        raise ValueError(f"Unrecognized input: {input_path}")

    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract features from raw gesture data")
    parser.add_argument("--input", default="data/raw",
                        help="Session folder, combined CSV, combined JSON, or JSONL")
    parser.add_argument("--output", default="data/processed",
                        help="Output directory for feature matrices")
    parser.add_argument("--window", type=int, default=2,
                        help="Sliding window size in frames")
    parser.add_argument("--stride", type=int, default=5,
                        help="Window stride in frames")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Test set proportion")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--collector", default=None,
                        help="Only include frames from this collector (default: all)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found.")
        return

    print(f"Loading data from: {input_path}")
    frames = _load_all_frames(input_path)
    if not frames:
        print("No frames loaded.")
        return
    print(f"Loaded {len(frames)} frames")

    if args.collector:
        before = len(frames)
        frames = [f for f in frames if f.get("collector") == args.collector]
        print(f"  Filtered to collector '{args.collector}': {len(frames)} / {before} frames")
        if not frames:
            print("Error: no frames match the specified collector.")
            return

    mm = frames[0].get("mmwave", {})
    d = mm.get("data", {}) if isinstance(mm, dict) else {}
    pts = d.get("points", [])
    print(f"\nFirst frame mmwave data keys: {list(d.keys())}")
    print(f"  num_points: {d.get('num_points')}")
    if pts:
        print(f"  Sample point (first of {len(pts)}): {pts[0]}")
        print(f"  Point keys: {list(pts[0].keys())}")
    else:
        print(f"  WARNING: No points in first frame — all features will be zero!")

    empty_count = sum(1 for f in frames
                      if not f.get("mmwave", {}).get("data", {}).get("points", []))
    print(f"  Frames with empty points: {empty_count}/{len(frames)}")

    mm = frames[0].get("mmwave", {})
    d = mm.get("data", {}) if isinstance(mm, dict) else {}
    pts = d.get("points", [])
    print(f"\nFirst frame mmwave data keys: {list(d.keys())}")
    print(f"  num_points: {d.get('num_points')}")
    if pts:
        print(f"  Sample point (first of {len(pts)}): {pts[0]}")
        print(f"  Point keys: {list(pts[0].keys())}")
    else:
        print(f"  WARNING: No points in first frame — all features will be zero!")

    input_ts = None
    stem = input_path.stem if not input_path.is_dir() else input_path.name
    if stem.startswith("session_"):
        input_ts = stem.removeprefix("session_")
    elif stem.startswith("combined_"):
        input_ts = stem.removeprefix("combined_")
    elif stem.startswith("features_"):
        input_ts = stem.removeprefix("features_")
    else:
        from datetime import datetime, timezone
        input_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    X, y, feature_names = extract_window_features(
        frames, window_size=args.window, stride=args.stride,
    )
    print(f"\nExtracted {len(X)} windows with {len(feature_names)} features each")
    print(f"Feature names: {feature_names}")

    unique_gestures, counts = np.unique(y, return_counts=True)
    print(f"\nClass distribution:")
    for g, c in zip(unique_gestures, counts):
        idx = y == g
        print(f"  {g} ({c} windows): "
              f"feat0 mean={X[idx, 0].mean():.3f} std={X[idx, 0].std():.3f} — "
              f"feat1 mean={X[idx, 1].mean():.3f} std={X[idx, 1].std():.3f}")

    gestures = sorted(set(y))
    label_map = {g: i for i, g in enumerate(gestures)}
    y_int = np.array([label_map[g] for g in y])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_int, test_size=args.test_size,
        random_state=args.random_state, stratify=y_int,
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = out_dir / f"features_{input_ts}.npz"
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
