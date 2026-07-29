from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


def load_jsonl(path: Path) -> list[dict]:
    frames = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                frames.append(json.loads(line))
    return frames


def load_session(session_dir: Path) -> list[dict]:
    events_path = session_dir / "events.csv"
    imu_path = session_dir / "imu.csv"
    if not events_path.exists() or not imu_path.exists():
        raise FileNotFoundError(f"Missing events.csv or imu.csv in {session_dir}")

    with open(events_path) as f:
        events = list(csv.DictReader(f))
    with open(imu_path) as f:
        imu_rows = list(csv.DictReader(f))

    frames = []
    for ev, im in zip(events, imu_rows):
        frames.append({
            "timestamp": ev["timestamp"],
            "gesture": ev["gesture"],
            "trial": int(ev["trial"]),
            "elapsed": float(ev["elapsed"]),
            "imu": {
                "data": {
                    "accel": [
                        float(im.get("accel_x", 0)),
                        float(im.get("accel_y", 0)),
                        float(im.get("accel_z", 0)),
                    ],
                    "gyro": [
                        float(im.get("gyro_x", 0)),
                        float(im.get("gyro_y", 0)),
                        float(im.get("gyro_z", 0)),
                    ],
                },
                "confidence": float(im.get("confidence", 0.95)),
                "sensor_type": "imu",
            },
        })
    return frames


def extract_window_features(
    frames: list[dict],
    window_size: int = 5,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Extract features from a list of IMU frames using a sliding window.

    Features per window:
      - Mean of each channel (accel x/y/z, gyro x/y/z, gyro_mag, accel_mag)
      - Std  of each channel (same 8)
      - Accel deltas  (frame-to-frame diff for accel x/y/z → 3*(window-1) values)
      - Gyro deltas  (frame-to-frame diff for gyro x/y/z → 3*(window-1) values)
      - Magnitude deltas (frame-to-frame diff for gyro_mag, accel_mag → 2*(window-1) values)
    """
    X_list, y_list = [], []
    base_names = [
        "accel_x", "accel_y", "accel_z",
        "gyro_x", "gyro_y", "gyro_z",
        "gyro_mag", "accel_mag",
    ]
    # Indices within the 8-column per-frame array
    ACCEL_IDX = slice(0, 3)
    GYRO_IDX = slice(3, 6)
    GYRO_MAG_IDX = 6
    ACCEL_MAG_IDX = 7

    for i in range(0, len(frames) - window_size + 1, stride):
        window = frames[i:i + window_size]
        gesture = window[0].get("gesture", "unknown")

        per_frame = []
        for f in window:
            imu = f.get("imu", {}).get("data", {})
            accel = imu.get("accel", [0, 0, 0])
            gyro = imu.get("gyro", [0, 0, 0])
            ax = float(accel[0]) if len(accel) > 0 else 0.0
            ay = float(accel[1]) if len(accel) > 1 else 0.0
            az = float(accel[2]) if len(accel) > 2 else 0.0
            gx = float(gyro[0]) if len(gyro) > 0 else 0.0
            gy = float(gyro[1]) if len(gyro) > 1 else 0.0
            gz = float(gyro[2]) if len(gyro) > 2 else 0.0
            per_frame.append([
                ax, ay, az,
                gx, gy, gz,
                np.sqrt(gx*gx + gy*gy + gz*gz),   # gyro magnitude
                np.sqrt(ax*ax + ay*ay + az*az),   # accel magnitude
            ])

        feats = np.array(per_frame)
        # Mean and std for all 8 channels
        features = list(np.mean(feats, axis=0))
        features.extend(np.std(feats, axis=0).tolist())

        # RMS (root-mean-square) — captures signal energy regardless of direction
        features.extend(np.sqrt(np.mean(feats ** 2, axis=0)).tolist())

        # Zero-crossing rate — counts how often each channel oscillates
        # Soli (finger rub) produces rapid oscillations → high ZCR
        # T-arm (static) produces no oscillation → near-zero ZCR
        for c in range(8):
            centered = feats[:, c] - np.mean(feats[:, c])
            if len(centered) < 2:
                features.append(0.0)
            else:
                crossings = np.sum((centered[:-1] * centered[1:]) < 0)
                features.append(float(crossings) / window_size)

        # Accel deltas (3 channels, window-1 time steps)
        features.extend((feats[1:, ACCEL_IDX] - feats[:-1, ACCEL_IDX]).flatten().tolist())
        # Gyro deltas (3 channels, window-1 time steps)
        features.extend((feats[1:, GYRO_IDX] - feats[:-1, GYRO_IDX]).flatten().tolist())
        # Magnitude deltas (2 channels, window-1 time steps)
        features.extend((feats[1:, 6:8] - feats[:-1, 6:8]).flatten().tolist())

        X_list.append(features)
        y_list.append(gesture)

    n_dt = window_size - 1
    accel_delta_names = [f"delta_{n}_t{t}" for n in ["accel_x","accel_y","accel_z"] for t in range(n_dt)]
    gyro_delta_names = [f"delta_{n}_t{t}" for n in ["gyro_x","gyro_y","gyro_z"] for t in range(n_dt)]
    mag_delta_names = [f"delta_{n}_t{t}" for n in ["gyro_mag","accel_mag"] for t in range(n_dt)]
    feature_names = (
        [f"mean_{n}" for n in base_names]
        + [f"std_{n}" for n in base_names]
        + [f"rms_{n}" for n in base_names]
        + [f"zcr_{n}" for n in base_names]
        + accel_delta_names
        + gyro_delta_names
        + mag_delta_names
    )
    return np.array(X_list), np.array(y_list), feature_names


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract IMU features from JSONL recordings")
    parser.add_argument("--input", nargs="+", default=["data/raw"],
                        help="JSONL files or directories")
    parser.add_argument("--output", default="data/processed",
                        help="Output directory for features.npz")
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    all_frames: list[dict] = []
    for path_str in args.input:
        p = Path(path_str)
        if p.is_dir():
            # Scan subdirectories for session folders with imu.csv
            session_dirs = sorted([d for d in p.iterdir() if d.is_dir()])
            found_any = False
            for session_dir in session_dirs:
                if (session_dir / "imu.csv").exists():
                    frames = load_session(session_dir)
                    print(f"  Loaded session {session_dir.name}: {len(frames)} frames")
                    all_frames.extend(frames)
                    found_any = True
            if not found_any:
                # Fallback: check if the directory itself has imu.csv
                if (p / "imu.csv").exists():
                    frames = load_session(p)
                    print(f"  Loaded session {p.name}: {len(frames)} frames")
                    all_frames.extend(frames)
                else:
                    # Try JSONL files as last fallback
                    for f in sorted(p.glob("*.jsonl")):
                        all_frames.extend(load_jsonl(f))
        elif p.suffix == ".jsonl":
            all_frames.extend(load_jsonl(p))

    print(f"\nTotal: {len(all_frames)} frames")

    X, y, feature_names = extract_window_features(all_frames, args.window, args.stride)
    print(f"Extracted {len(X)} windows with {len(feature_names)} features")
    print(f"Feature names: {feature_names}")

    gestures = sorted(set(y))
    label_map = {g: i for i, g in enumerate(gestures)}
    y_int = np.array([label_map[g] for g in y])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_int, test_size=args.test_size,
        random_state=args.random_state, stratify=y_int,
    )

    output_path = Path(args.output)
    if output_path.is_dir() or output_path.suffix != ".npz":
        output_path = output_path / "imu_features.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path,
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        feature_names=feature_names,
        gestures=gestures,
        label_map=label_map,
        window_size=args.window,
        stride=args.stride,
    )
    print(f"\nSaved: {output_path}")
    print(f"  Train: {len(X_train)}  Test: {len(X_test)}  Classes: {len(gestures)} ({', '.join(gestures)})")


if __name__ == "__main__":
    main()
