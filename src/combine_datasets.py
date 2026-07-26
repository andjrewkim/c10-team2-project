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
            for trial_key, trial_frames in trials.items():
                if len(trial_frames) < args.min_frames:
                    print(f"  Skipping {input_path.name}/{trial_key}: {len(trial_frames)} frames < {args.min_frames}")
                    skipped += 1
                    continue
                gesture = trial_frames[0].get("gesture", "unknown")
                gesture_counts[gesture] += len(trial_frames)
                all_frames.extend(trial_frames)
            print(f"  Loaded {input_path.name}: {sum(len(v) for v in trials.values())} frames")

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
                    for trial_key, trial_frames in trials.items():
                        if len(trial_frames) < args.min_frames:
                            print(f"  Skipping {ev_csv.parent.name}/{trial_key}: {len(trial_frames)} frames < {args.min_frames}")
                            skipped += 1
                            continue
                        gesture = trial_frames[0].get("gesture", "unknown")
                        gesture_counts[gesture] += len(trial_frames)
                        all_frames.extend(trial_frames)
                    print(f"  Loaded {ev_csv.parent.name}: {sum(len(v) for v in trials.values())} frames")
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
                gesture = frame.get("gesture", "unknown")
                gesture_counts[gesture] += 1
            all_frames.extend(frames_list)
            print(f"  Loaded {input_path.name}: {len(frames_list)} frames")

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

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"combined_{timestamp}.csv"

    fieldnames = ["frame_index", "timestamp", "gesture", "trial", "elapsed",
                  "confidence", "num_points", "points", "range_profile", "motion_score"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, frame in enumerate(all_frames):
            mm = frame.get("mmwave", {})
            d = mm.get("data", {}) if isinstance(mm, dict) else {}
            row = {
                "frame_index": idx,
                "timestamp": frame.get("timestamp", ""),
                "gesture": frame.get("gesture", ""),
                "trial": frame.get("trial", 0),
                "elapsed": round(frame.get("elapsed", 0.0), 6),
                "confidence": mm.get("confidence", 0.0) if isinstance(mm, dict) else 0.0,
                "num_points": d.get("num_points", len(d.get("points", []))),
                "points": json.dumps(d.get("points", [])),
                "range_profile": json.dumps(d.get("range_profile")),
                "motion_score": d.get("motion_score", 0.0),
            }
            writer.writerow(row)
    print(f"\nSaved: {csv_path} ({len(all_frames)} rows)")

    label_path = out_dir / f"combined_{timestamp}_labels.json"
    with open(label_path, "w") as f:
        json.dump({"label_to_int": label_to_int, "int_to_label": int_to_label}, f, indent=2)
    print(f"Saved: {label_path}")


if __name__ == "__main__":
    main()