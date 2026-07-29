from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from src.visualize import load_session_csvs


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine raw gesture recordings")
    parser.add_argument("--input", nargs="+", help="List of session folders to combine", required=True)
    parser.add_argument("--output", default="data/processed",
                        help="Output directory for combined dataset")
    parser.add_argument("--min-frames", type=int, default=1,
                        help="Minimum frames per trial to keep")
    args = parser.parse_args()

    input_paths = [Path(p) for p in args.input]

    all_frames: list[dict] = []
    gesture_counts: Counter = Counter()
    skipped = 0
    latest_ts: str | None = None

    def _update_latest_ts(name: str) -> None:
        nonlocal latest_ts
        for prefix in ("session_", "combined_", "features_"):
            if name.startswith(prefix):
                ts = name.removeprefix(prefix)
                if latest_ts is None or ts > latest_ts:
                    latest_ts = ts
                return

    for input_path in input_paths:
        if not input_path.exists():
            print(f"Skipping {input_path}: does not exist")
            skipped += 1
            continue

        if input_path.is_dir() and (input_path / "events.csv").exists():
            try:
                trials = load_session_csvs(input_path)
            except Exception as e:
                print(f"  Skipping {input_path.name}: {e}")
                skipped += 1
                continue
            source = input_path.name
            for trial_key, trial_frames in trials.items():
                if len(trial_frames) < args.min_frames:
                    print(f"  Skipping {input_path.name}/{trial_key}: {len(trial_frames)} frames < {args.min_frames}")
                    skipped += 1
                    continue
                for f in trial_frames:
                    f["dataset_source"] = source
                gesture = trial_frames[0].get("gesture", "unknown")
                gesture_counts[gesture] += len(trial_frames)
                all_frames.extend(trial_frames)
            print(f"  Loaded {input_path.name}: {sum(len(v) for v in trials.values())} frames")
            _update_latest_ts(input_path.name)

        elif input_path.is_dir():
            session_dirs = sorted(input_path.glob("*/events.csv"))
            if session_dirs:
                for ev_csv in session_dirs:
                    try:
                        trials = load_session_csvs(ev_csv.parent)
                    except Exception as e:
                        print(f"  Skipping {ev_csv.parent.name}: {e}")
                        skipped += 1
                        continue
                    source = ev_csv.parent.name
                    for trial_key, trial_frames in trials.items():
                        if len(trial_frames) < args.min_frames:
                            print(f"  Skipping {ev_csv.parent.name}/{trial_key}: {len(trial_frames)} frames < {args.min_frames}")
                            skipped += 1
                            continue
                        for f in trial_frames:
                            f["dataset_source"] = source
                        gesture = trial_frames[0].get("gesture", "unknown")
                        gesture_counts[gesture] += len(trial_frames)
                        all_frames.extend(trial_frames)
                    print(f"  Loaded {ev_csv.parent.name}: {sum(len(v) for v in trials.values())} frames")
                    _update_latest_ts(ev_csv.parent.name)
            else:
                jsonl_files = sorted(input_path.glob("*.jsonl"))
                if not jsonl_files:
                    print(f"  No session folders or .jsonl files found in {input_path}")
                    skipped += 1
                    continue
                print(f"  Found {len(jsonl_files)} .jsonl files in {input_path}")
                for fpath in jsonl_files:
                    frames = [json.loads(line) for line in fpath.read_text().strip().splitlines() if line.strip()]
                    if len(frames) < args.min_frames:
                        print(f"    Skipping {fpath.name}: {len(frames)} frames < {args.min_frames}")
                        skipped += 1
                        continue
                    for frame in frames:
                        frame["dataset_source"] = fpath.stem
                        gesture = frame.get("gesture", "unknown")
                        gesture_counts[gesture] += 1
                    all_frames.extend(frames)

        elif input_path.suffix in (".json", ".jsonl"):
            if input_path.suffix == ".json":
                data = json.loads(input_path.read_text())
                frames_list = data.get("frames", data if isinstance(data, list) else [])
            else:
                frames_list = [json.loads(line) for line in input_path.read_text().strip().splitlines() if line.strip()]
            if len(frames_list) < args.min_frames:
                print(f"  Skipping {input_path.name}: {len(frames_list)} frames < {args.min_frames}")
                skipped += 1
                continue
            for frame in frames_list:
                frame.setdefault("dataset_source", input_path.stem)
                gesture = frame.get("gesture", "unknown")
                gesture_counts[gesture] += 1
            all_frames.extend(frames_list)
            print(f"  Loaded {input_path.name}: {len(frames_list)} frames")
            _update_latest_ts(input_path.stem)

        else:
            print(f"  Skipping {input_path}: unrecognized format")
            skipped += 1

    if not all_frames:
        print("No frames loaded.")
        return

    print("Gesture distribution:")
    for gesture, count in sorted(gesture_counts.items()):
        print(f"  {gesture}: {count} frames")

    gestures_list = sorted(set(f.get("gesture", "unknown") for f in all_frames))
    label_to_int = {g: i for i, g in enumerate(gestures_list)}
    int_to_label = {i: g for g, i in label_to_int.items()}

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = latest_ts or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"combined_{timestamp}.csv"
    suffix = 1
    while csv_path.exists():
        csv_path = out_dir / f"combined_{timestamp}_{suffix}.csv"
        suffix += 1

    # Discover all sensor keys across frames
    sensor_keys = set()
    for frame in all_frames:
        for k, v in frame.items():
            if k in ("timestamp", "gesture", "trial", "elapsed", "dataset_source", "frame_index"):
                continue
            if isinstance(v, dict) and "data" in v:
                sensor_keys.add(k)
    sensor_keys = sorted(sensor_keys)

    # Build dynamic fieldnames
    fieldnames = ["frame_index", "timestamp", "gesture", "trial", "elapsed", "dataset_source"]
    for sk in sensor_keys:
        fieldnames.append(f"{sk}_confidence")
        sample_data = {}
        for f in all_frames:
            if sk in f:
                sample_data = f[sk].get("data", {})
                break
        for dk in sample_data:
            if dk in ("frame_index", "confidence"):
                continue
            fieldnames.append(f"{sk}_{dk}")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, frame in enumerate(all_frames):
            row = {
                "frame_index": idx,
                "timestamp": frame.get("timestamp", ""),
                "gesture": frame.get("gesture", ""),
                "trial": frame.get("trial", 0),
                "elapsed": round(frame.get("elapsed", 0.0), 6),
                "dataset_source": frame.get("dataset_source", ""),
            }
            for sk in sensor_keys:
                sensor = frame.get(sk, {})
                row[f"{sk}_confidence"] = sensor.get("confidence", 0.0) if isinstance(sensor, dict) else 0.0
                data = sensor.get("data", {}) if isinstance(sensor, dict) else {}
                for dk, dv in data.items():
                    if dk in ("frame_index", "confidence"):
                        continue
                    if isinstance(dv, (list, dict)):
                        row[f"{sk}_{dk}"] = json.dumps(dv)
                    elif dv is None:
                        row[f"{sk}_{dk}"] = ""
                    else:
                        row[f"{sk}_{dk}"] = dv
            writer.writerow(row)
    print(f"\nSaved: {csv_path} ({len(all_frames)} rows)")

    label_path = csv_path.with_stem(csv_path.stem + "_labels").with_suffix(".json")
    with open(label_path, "w") as f:
        json.dump({"label_to_int": label_to_int, "int_to_label": int_to_label}, f, indent=2)
    print(f"Saved: {label_path}")


if __name__ == "__main__":
    main()