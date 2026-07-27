from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def convert_session(session_dir: Path, output_dir: Path) -> list[Path]:
    events_path = session_dir / "events.csv"
    mmwave_path = session_dir / "mmwave.csv"
    metadata_path = session_dir / "session_metadata.json"

    if not all(p.exists() for p in [events_path, mmwave_path]):
        print(f"  Skipping {session_dir.name}: missing events.csv or mmwave.csv")
        return []

    with open(metadata_path) as f:
        meta = json.load(f)
    session_id = meta["session"]

    # Load events: frame_index -> {timestamp, gesture, trial, elapsed}
    events: list[dict] = []
    with open(events_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            events.append(row)

    # Load mmwave frames: frame_index -> {confidence, num_points, points_json, ...}
    mmwave_frames: list[dict] = []
    with open(mmwave_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            mmwave_frames.append(row)

    if len(events) != len(mmwave_frames):
        print(f"  Warning: {session_dir.name}: {len(events)} events != {len(mmwave_frames)} mmwave frames")

    # Group by (gesture, trial) and write one JSONL per trial
    from collections import defaultdict
    trial_groups: dict[tuple[str, str], list[tuple[int, dict, dict]]] = defaultdict(list)

    for i in range(min(len(events), len(mmwave_frames))):
        ev = events[i]
        mm = mmwave_frames[i]
        key = (ev["gesture"], ev["trial"])
        trial_groups[key].append((int(ev["frame_index"]), ev, mm))

    output_dir.mkdir(parents=True, exist_ok=True)
    out_paths: list[Path] = []

    for (gesture, trial), frames in sorted(trial_groups.items()):
        frames.sort(key=lambda x: x[0])  # sort by frame_index
        trial_num = int(trial)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = output_dir / f"{gesture}_t{trial_num:03d}_{session_id}_{timestamp}.jsonl"

        with open(out_path, "w") as f:
            for frame_idx, ev, mm in frames:
                try:
                    points = json.loads(mm["points"])
                except (json.JSONDecodeError, KeyError):
                    points = []

                frame = {
                    "timestamp": ev["timestamp"],
                    "gesture": ev["gesture"],
                    "trial": int(ev["trial"]),
                    "elapsed": float(ev["elapsed"]),
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
                        "confidence": float(mm.get("confidence", 0.8)),
                        "sensor_type": "mmwave",
                    },
                }
                f.write(json.dumps(frame) + "\n")

        num_frames = len(frames)
        print(f"  Wrote {num_frames} frames -> {out_path.name}")
        out_paths.append(out_path)

    return out_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert session CSV data to JSONL format")
    parser.add_argument("--input", default="data/raw",
                        help="Directory containing session_* folders")
    parser.add_argument("--output", default="data/raw",
                        help="Output directory for JSONL files (same dir as manifest)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    session_dirs = sorted(input_dir.glob("session_*"))
    if not session_dirs:
        print(f"No session_* directories found in {input_dir}")
        return

    all_files: list[Path] = []
    for sd in session_dirs:
        print(f"\nConverting {sd.name}...")
        files = convert_session(sd, output_dir)
        all_files.extend(files)

    # Build manifest
    from collections import Counter
    gesture_counts: Counter = Counter()
    total_frames = 0
    for p in all_files:
        with open(p) as f:
            for line in f:
                if line.strip():
                    frame = json.loads(line)
                    gesture_counts[frame.get("gesture", "unknown")] += 1
                    total_frames += 1

    manifest = {
        "source": "session_csv_conversion",
        "files": [str(p.relative_to(output_dir)) for p in sorted(all_files)],
        "total_frames": total_frames,
        "gestures": list(gesture_counts.keys()),
        "gesture_distribution": dict(gesture_counts),
    }

    manifest_path = output_dir / "manifest_from_sessions.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. {len(all_files)} files, {total_frames} total frames.")
    print(f"Manifest saved to {manifest_path}")
    print(f"\nNow run: python -m src.combine_datasets && python -m src.extract_features && python -m src.train")


if __name__ == "__main__":
    main()
