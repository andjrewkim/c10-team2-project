from __future__ import annotations

import argparse
import json
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


def print_stats(frames: list[dict]) -> None:
    gestures = {}
    for f in frames:
        g = f.get("gesture", "?")
        gestures[g] = gestures.get(g, 0) + 1

    print(f"\nTotal frames: {len(frames)}")
    print(f"Gestures: {gestures}")
    sensors_in_frame = [k for k in frames[0].keys() if k not in ("timestamp", "gesture", "trial", "elapsed")]
    print(f"Sensors: {sensors_in_frame}")
    print()


def show_mmwave(frames: list[dict], max_frames: int = 200) -> None:
    import matplotlib.pyplot as plt

    n = min(len(frames), max_frames)
    ts = np.arange(n)
    xs, ys, zs, vs, npts = [], [], [], [], []
    for f in frames[:n]:
        mm = f.get("mmwave", {})
        d = mm.get("data", {})
        pts = d.get("points", [])
        npts.append(len(pts))
        if pts:
            xs.append(np.mean([p["x"] for p in pts]))
            ys.append(np.mean([p["y"] for p in pts]))
            zs.append(np.mean([p["z"] for p in pts]))
            vs.append(np.mean([abs(p.get("velocity", 0)) for p in pts]))
        else:
            xs.append(0)
            ys.append(0)
            zs.append(0)
            vs.append(0)

    fig, axs = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    axs[0].plot(ts, xs, label="x", alpha=0.7)
    axs[0].plot(ts, ys, label="y", alpha=0.7)
    axs[0].plot(ts, zs, label="z", alpha=0.7)
    axs[0].set_ylabel("Centroid (m)")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(ts, vs, color="orange")
    axs[1].set_ylabel("Mean |velocity| (m/s)")
    axs[1].grid(True, alpha=0.3)

    axs[2].plot(ts, npts, color="green")
    axs[2].set_ylabel("Point count")
    axs[2].set_xlabel("Frame")
    axs[2].grid(True, alpha=0.3)

    gesture = frames[0].get("gesture", "?")
    fig.suptitle(f"mmWave — {gesture} (trial {frames[0].get('trial', 0)})")
    plt.tight_layout()
    plt.show()


def show_imu(frames: list[dict], max_frames: int = 200) -> None:
    import matplotlib.pyplot as plt

    n = min(len(frames), max_frames)
    ts = np.arange(n)
    axs, ays, azs = [], [], []
    gxs, gys, gzs = [], [], []
    for f in frames[:n]:
        im = f.get("imu", {})
        d = im.get("data", {})
        acc = d.get("accel", [0, 0, 0])
        gyr = d.get("gyro", [0, 0, 0])
        axs.append(acc[0])
        ays.append(acc[1])
        azs.append(acc[2])
        gxs.append(gyr[0])
        gys.append(gyr[1])
        gzs.append(gyr[2])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax1.plot(ts, axs, label="accel x")
    ax1.plot(ts, ays, label="accel y")
    ax1.plot(ts, azs, label="accel z")
    ax1.set_ylabel("Accel (m/s²)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(ts, gxs, label="gyro x")
    ax2.plot(ts, gys, label="gyro y")
    ax2.plot(ts, gzs, label="gyro z")
    ax2.set_ylabel("Gyro (rad/s)")
    ax2.set_xlabel("Frame")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    gesture = frames[0].get("gesture", "?")
    fig.suptitle(f"IMU — {gesture} (trial {frames[0].get('trial', 0)})")
    plt.tight_layout()
    plt.show()


def overlay_comparison(frames_by_gesture: dict[str, list[dict]], sensor: str, field: str, max_frames: int = 100) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    for gesture, flist in frames_by_gesture.items():
        n = min(len(flist), max_frames)
        vals = []
        for f in flist[:n]:
            s = f.get(sensor, {})
            d = s.get("data", {})
            pts = d.get("points", [])
            if field == "num_points":
                vals.append(len(pts))
            elif field == "mean_x" and pts:
                vals.append(np.mean([p["x"] for p in pts]))
            elif field == "mean_y" and pts:
                vals.append(np.mean([p["y"] for p in pts]))
            elif field == "mean_vel" and pts:
                vals.append(np.mean([abs(p.get("velocity", 0)) for p in pts]))
            else:
                vals.append(0)
        ax.plot(vals, label=gesture, alpha=0.8)
    ax.set_ylabel(field)
    ax.set_xlabel("Frame")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.suptitle(f"Overlay: {sensor}.{field} by gesture")
    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize recorded gesture data")
    parser.add_argument("--input", default="data/raw",
                        help="Path to JSONL file or directory of JSONL files")
    parser.add_argument("--mode", choices=["stats", "mmwave", "imu", "overlay"],
                        default="stats",
                        help="What to show")
    parser.add_argument("--field", default="mean_y",
                        help="Field for overlay mode (num_points, mean_x, mean_y, mean_vel)")
    parser.add_argument("--gesture", default=None,
                        help="Filter to one gesture (for mmwave/imu modes)")
    args = parser.parse_args()

    path = Path(args.input)
    if path.is_dir():
        files = sorted(path.glob("*.jsonl"))
        if not files:
            print(f"No .jsonl files in {path}")
            return
    else:
        files = [path]

    frames_by_file = {}
    for f in files:
        frames = load_jsonl(f)
        gesture = frames[0].get("gesture", "?") if frames else "?"
        trial = frames[0].get("trial", 0) if frames else 0
        key = f"{gesture}_t{trial}"
        frames_by_file[key] = frames

    print(f"Loaded {len(files)} files")
    for name, frames in frames_by_file.items():
        print(f"  {name}: {len(frames)} frames")

    if args.mode == "stats":
        for name, frames in frames_by_file.items():
            print(f"\n=== {name} ===")
            print_stats(frames)
        return

    if args.mode == "overlay":
        by_gesture: dict[str, list[dict]] = {}
        for name, frames in frames_by_file.items():
            g = frames[0].get("gesture", "?")
            if g not in by_gesture:
                by_gesture[g] = []
            by_gesture[g].extend(frames)
        sensor = "mmwave"
        for f in list(by_gesture.values())[0]:
            if "imu" in f:
                sensor = "imu"
                break
        overlay_comparison(by_gesture, sensor, args.field)
        return

    for name, frames in frames_by_file.items():
        gesture = frames[0].get("gesture", "?")
        if args.gesture and gesture != args.gesture:
            continue
        print(f"\n=== {name} ({gesture}) ===")
        if args.mode == "mmwave":
            has_mm = any("mmwave" in f for f in frames)
            if has_mm:
                show_mmwave(frames)
            else:
                print("No mmWave data in this file")
        elif args.mode == "imu":
            has_imu = any("imu" in f for f in frames)
            if has_imu:
                show_imu(frames)
            else:
                print("No IMU data in this file")


if __name__ == "__main__":
    main()
