"""Merge ALL available multi-sensor CSVs and extract features."""

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from src.extract_features import extract_window_features


def load_csv_frames(path: Path) -> list[dict]:
    frames = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        all_cols = reader.fieldnames or []
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
    return frames


def main():
    print("=== Merging ALL multi-sensor data ===\n")
    all_frames = []

    csv_paths = [
        Path("data/multi_combined/combined_20260729_220712.csv"),
        Path("data/multi_combined/combined_20260728_235002.csv"),
    ]
    for p in csv_paths:
        frames = load_csv_frames(p)
        print(f"  {p.name}: {len(frames)} frames")
        all_frames.extend(frames)

    print(f"\n  Total: {len(all_frames)} frames")

    if not all_frames:
        print("No frames loaded!")
        return

    # Extract features
    X, y, feature_names = extract_window_features(all_frames, window_size=10, stride=5)
    print(f"\n  Extracted {len(X)} windows with {len(feature_names)} features each")

    gestures = sorted(set(y))
    label_map = {g: i for i, g in enumerate(gestures)}
    y_int = np.array([label_map[g] for g in y])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_int, test_size=0.2, random_state=42, stratify=y_int,
    )

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "features_all_data.npz"
    np.savez_compressed(
        out_path,
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        feature_names=feature_names,
        gestures=gestures,
        label_map=label_map,
        window_size=10,
        stride=5,
    )
    print(f"\n  Saved: {out_path}")
    print(f"  Train: {len(X_train)} windows, Test: {len(X_test)} windows")
    print(f"  Classes: {len(gestures)} ({', '.join(gestures)})")


if __name__ == "__main__":
    main()
