from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def load_jsonl(path: Path) -> list[dict]:
    frames = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                frames.append(json.loads(line))
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine raw gesture recordings")
    parser.add_argument("--input", default="data/raw",
                        help="Input raw data directory")
    parser.add_argument("--output", default="data/processed",
                        help="Output directory for combined dataset")
    parser.add_argument("--min-frames", type=int, default=10,
                        help="Minimum frames per recording to keep")
    args = parser.parse_args()

    raw_dir = Path(args.input)
    if not raw_dir.exists():
        print(f"Error: {raw_dir} does not exist")
        return

    jsonl_files = sorted(raw_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"No .jsonl files found in {raw_dir}")
        return

    print(f"Found {len(jsonl_files)} recording files")

    all_frames: list[dict] = []
    gesture_counts: Counter = Counter()
    skipped = 0

    for fpath in jsonl_files:
        frames = load_jsonl(fpath)
        if len(frames) < args.min_frames:
            print(f"  Skipping {fpath.name}: {len(frames)} frames < {args.min_frames}")
            skipped += 1
            continue
        for frame in frames:
            gesture = frame.get("gesture", "unknown")
            gesture_counts[gesture] += 1
        all_frames.extend(frames)

    print(f"\nLoaded {len(all_frames)} frames from {len(jsonl_files) - skipped} files")
    print("Gesture distribution:")
    for gesture, count in sorted(gesture_counts.items()):
        print(f"  {gesture}: {count} frames")

    gestures_list = sorted(set(f.get("gesture", "unknown") for f in all_frames))
    label_to_int = {g: i for i, g in enumerate(gestures_list)}
    int_to_label = {i: g for g, i in label_to_int.items()}

    combined = {
        "metadata": {
            "num_frames": len(all_frames),
            "num_gestures": len(gestures_list),
            "gestures": gestures_list,
            "label_to_int": label_to_int,
            "num_files": len(jsonl_files) - skipped,
        },
        "frames": all_frames,
    }

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = out_dir / "combined_dataset.npz"
    np.savez_compressed(npz_path, data=all_frames,
                        metadata=combined["metadata"])
    print(f"\nSaved dataset: {npz_path}")

    json_path = out_dir / "combined_dataset.json"
    with open(json_path, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    print(f"Saved JSON: {json_path}")

    label_path = out_dir / "labels.json"
    with open(label_path, "w") as f:
        json.dump({"label_to_int": label_to_int, "int_to_label": int_to_label}, f, indent=2)
    print(f"Saved labels: {label_path}")


if __name__ == "__main__":
    main()
