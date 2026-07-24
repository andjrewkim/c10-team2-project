from __future__ import annotations

# Make `from src.*` imports resolve no matter how the script is launched:
# `python src/collect.py` from the project root, `python -m src.collect`,
# or `cd src && python collect.py` should all just work — without requiring
# the user to remember to set PYTHONPATH or `cd` to the right directory.
import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from src.sensors.imu_reader import ImuReader
from src.sensors.mmwave_reader import MmWaveReader
from src.sensors.rfid_reader import RfidReader
from src.sensors.uwb_reader import UwbReader
from src.sensors.wifi_reader import WiFiReader

SENSOR_REGISTRY = {
    "mmwave": MmWaveReader,
    "imu": ImuReader,
    "uwb": UwbReader,
    "wifi": WiFiReader,
    "rfid": RfidReader,
}

ALL_GESTURES = [
    "pull", "push", "clockwise", "anticlockwise",
    "right", "left", "bye-bye", "one-arm-boxing",
    "clapping", "two-arm-boxing", "t-arm",
    "raise-arms", "soli", "making-fist-open",
    "palm-up-down",
]


def collect_gesture(
    reader_map: dict[str, any],
    gesture: str,
    duration_s: float,
    fps: float,
    output_dir: Path,
    trial: int = 0,
    prompt: bool = True,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"{gesture}_t{trial:03d}_{timestamp}.jsonl"

    interval = 1.0 / max(fps, 1)
    frames: list[dict] = []
    start_time = time.monotonic()
    deadline = start_time + duration_s

    print(f"  Recording '{gesture}' trial {trial} for {duration_s}s...")
    if gesture != "none" and prompt:
        input(f"  Press Enter when ready to perform '{gesture}'...")

    while time.monotonic() < deadline:
        frame: dict[str, any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "gesture": gesture,
            "trial": trial,
            "elapsed": time.monotonic() - start_time,
        }
        for name, reader in reader_map.items():
            try:
                reading = reader.read()
                frame[name] = {
                    "data": reading.data,
                    "confidence": reading.confidence,
                    "sensor_type": reading.sensor_type,
                }
            except Exception as e:
                frame[name] = {"error": str(e)}
        frames.append(frame)
        remaining = deadline - time.monotonic()
        if remaining < interval:
            break
        time.sleep(max(0, interval - (time.monotonic() - start_time - frame["elapsed"])))

    with open(out_path, "w") as f:
        for frame in frames:
            f.write(json.dumps(frame) + "\n")

    print(f"  Saved {len(frames)} frames to {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect gesture data")
    parser.add_argument("--gestures", nargs="+", default=["push", "pull"],
                        help="Gestures to collect")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Seconds per trial")
    parser.add_argument("--trials", type=int, default=3,
                        help="Trials per gesture")
    parser.add_argument("--fps", type=float, default=10,
                        help="Target frame rate")
    parser.add_argument("--sensors", nargs="+", default=["mmwave"],
                        choices=list(SENSOR_REGISTRY.keys()),
                        help="Sensors to use")
    parser.add_argument("--output", default="data/raw",
                        help="Output directory")
    parser.add_argument("--mode", default="mock",
                        choices=["mock", "serial"],
                        help="Sensor mode")
    parser.add_argument("--no-prompt", action="store_true",
                        help="Skip 'press enter' prompts (for automation)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    reader_map: dict[str, any] = {}
    for name in args.sensors:
        cls = SENSOR_REGISTRY[name]
        if name == "mmwave":
            reader = cls(mode=args.mode)
        elif name == "imu":
            reader = cls(mode=args.mode)
        elif name == "uwb":
            reader = cls(mode=args.mode)
        elif name == "wifi":
            reader = cls(mode=args.mode)
        elif name == "rfid":
            reader = cls(mode=args.mode)
        else:
            reader = cls()
        reader.start()
        reader_map[name] = reader
        print(f"  Started {name} reader ({args.mode} mode)")

    all_files: list[Path] = []
    try:
        for gesture in args.gestures:
            if gesture not in ALL_GESTURES:
                print(f"  Warning: '{gesture}' not in standard gesture list")
            for trial in range(args.trials):
                path = collect_gesture(
                    reader_map, gesture, args.duration,
                    args.fps, output_dir, trial,
                    prompt=not args.no_prompt,
                )
                all_files.append(path)
    finally:
        for name, reader in reader_map.items():
            try:
                reader.stop()
            except Exception:
                pass

    manifest_path = output_dir / "manifest.json"
    files_rel = [str(p.relative_to(output_dir)) for p in all_files]
    total = 0
    for fname in files_rel:
        with open(output_dir / fname) as f:
            total += sum(1 for _ in f)
    manifest = {
        "gestures": args.gestures,
        "trials_per_gesture": args.trials,
        "duration_s": args.duration,
        "fps": args.fps,
        "sensors": args.sensors,
        "mode": args.mode,
        "files": files_rel,
        "total_frames": total,
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to {manifest_path}")
    print(f"Total: {total} frames across {len(all_files)} files")


if __name__ == "__main__":
    main()
