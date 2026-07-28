from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.sensors.base_reader import Reading
from src.sensors.imu_reader import ImuReader
from src.sensors.mmWave.mmwave_reader import MmWaveReader
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

SENSOR_CSV_FIELDS: dict[str, list[str]] = {
    "mmwave": [
        "frame_index", "confidence", "num_points", "points",
        "range_profile", "motion_score",
    ],
    "imu": [
        "frame_index", "confidence",
        "accel_x", "accel_y", "accel_z",
        "gyro_x", "gyro_y", "gyro_z",
        "quat", "trajectory",
    ],
    "uwb": [
        "frame_index", "confidence",
        "ranges_cm", "position", "raw_ranges",
    ],
    "rfid": [
        "frame_index", "confidence",
        "tags", "touch", "antenna",
    ],
    "wifi": [
        "frame_index", "confidence",
        "rssi", "csi", "amplitudes",
    ],
}

EVENTS_FIELDS = ["frame_index", "timestamp", "gesture", "trial", "elapsed"]
TRIALS_FIELDS = ["trial_index", "gesture", "trial_num", "start_timestamp", "end_timestamp", "num_frames"]


def _flatten_reading(reading: Reading, frame_index: int) -> dict[str, Any]:
    row: dict[str, Any] = {"frame_index": frame_index, "confidence": reading.confidence}
    d = reading.data

    if reading.sensor_type == "mmwave":
        row["num_points"] = d.get("num_points", 0)
        row["points"] = json.dumps(d.get("points", []))
        row["range_profile"] = json.dumps(d.get("range_profile"))
        row["motion_score"] = d.get("motion_score", 0.0)

    elif reading.sensor_type == "imu":
        accel = d.get("accel", [0, 0, 0])
        row["accel_x"], row["accel_y"], row["accel_z"] = accel
        gyro = d.get("gyro", [0, 0, 0])
        row["gyro_x"], row["gyro_y"], row["gyro_z"] = gyro
        row["quat"] = json.dumps(d.get("quat"))
        row["trajectory"] = json.dumps(d.get("trajectory"))

    elif reading.sensor_type == "uwb":
        row["ranges_cm"] = json.dumps(d.get("ranges_cm", []))
        row["position"] = json.dumps(d.get("position"))
        row["raw_ranges"] = json.dumps(d.get("raw_ranges", []))

    elif reading.sensor_type == "rfid":
        row["tags"] = json.dumps(d.get("tags", []))
        row["touch"] = d.get("touch", False)
        row["antenna"] = d.get("antenna")

    elif reading.sensor_type == "wifi":
        row["rssi"] = json.dumps(d.get("rssi", {}))
        row["csi"] = json.dumps(d.get("csi"))
        row["amplitudes"] = json.dumps(d.get("amplitudes"))

    return row


def _open_csv(path: Path, fieldnames: list[str]) -> tuple[csv.DictWriter, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "w", newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    f.flush()
    return writer, f


def collect_gesture(
    reader_map: dict[str, Any],
    gesture: str,
    duration_s: float,
    fps: float,
    trial: int = 0,
    total_trials: int = 1,
    prompt: bool = True,
) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []

    print(f"  Recording '{gesture}' trial {trial + 1}/{total_trials} for {duration_s}s...")
    if gesture != "none" and prompt:
        input(f"  Press Enter when ready to perform '{gesture}'...")

    interval = 1.0 / max(fps, 1)
    start_time = time.monotonic()
    deadline = start_time + duration_s
    frame_idx = 0

    while time.monotonic() < deadline:
        frame: dict[str, Any] = {
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
        frame["_frame_index"] = frame_idx
        frames.append(frame)
        frame_idx += 1

        remaining = deadline - time.monotonic()
        if remaining < interval:
            break
        time.sleep(max(0, interval - (time.monotonic() - start_time - frame["elapsed"])))

    print(f"  Collected {len(frames)} frames")
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect gesture data")
    parser.add_argument("--gestures", nargs="+", default=["push", "pull"],
                        help="Gestures to collect")
    parser.add_argument("--duration", type=float, default=2.0,
                        help="Seconds per trial")
    parser.add_argument("--trials", type=int, default=3,
                        help="Trials per gesture")
    parser.add_argument("--fps", type=float, default=10,
                        help="Target frame rate")
    parser.add_argument("--sensors", nargs="+", default=["mmwave"],
                        choices=list(SENSOR_REGISTRY.keys()),
                        help="Sensors to use")
    parser.add_argument("--output", default="data/raw",
                        help="Output base directory")
    parser.add_argument("--mode", default="mock",
                        choices=["mock", "serial"],
                        help="Sensor mode")
    parser.add_argument("--imu_port", default="com13", help="IMU serial port")
    parser.add_argument("--mmwave_port", default="com12", help="mmWave serial port")
    parser.add_argument("--uwb-ports", nargs="+", default=["/dev/ttyACM0"],
                        help="Serial ports for UWB devices")
    parser.add_argument("--no-prompt", action="store_true",
                        help="Skip 'press enter' prompts (for automation)")
    args = parser.parse_args()

    session_time = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    session_dir = Path(args.output) / f"session_{session_time}"
    session_dir.mkdir(parents=True, exist_ok=True)
    print(f"Session directory: {session_dir}")

    writers: dict[str, csv.DictWriter] = {}
    file_handles: list[Any] = []
    for name in args.sensors:
        w, fh = _open_csv(session_dir / f"{name}.csv", SENSOR_CSV_FIELDS[name])
        writers[name] = w
        file_handles.append(fh)

    events_writer, ev_fh = _open_csv(session_dir / "events.csv", EVENTS_FIELDS)
    trials_writer, tr_fh = _open_csv(session_dir / "trials.csv", TRIALS_FIELDS)
    file_handles += [ev_fh, tr_fh]

    reader_map: dict[str, Any] = {}
    for name in args.sensors:
        cls = SENSOR_REGISTRY[name]
        if name == "mmwave":
            reader = cls(mode=args.mode, serial_port=args.mmwave_port, cfg_path="config/point_cloud.cfg")
        elif name == "imu":
            reader = cls(mode=args.mode, serial_port=args.imu_port)
        else:
            reader = cls(mode=args.mode)
        reader.start()
        reader_map[name] = reader
        print(f"  Started {name} reader ({args.mode} mode)")

    trial_index = 0
    total_frames = 0
    try:
        for gesture in args.gestures:
            if gesture not in ALL_GESTURES:
                print(f"  Warning: '{gesture}' not in standard gesture list")
            for trial_num in range(args.trials):
                trial_start_ts = datetime.now(timezone.utc).isoformat()
                frames = collect_gesture(
                    reader_map, gesture, args.duration,
                    args.fps, trial_num, total_trials=args.trials,
                    prompt=not args.no_prompt,
                )
                trial_end_ts = datetime.now(timezone.utc).isoformat()

                keep = input(f"  Keep this '{gesture}' trial {trial_num} ({len(frames)} frames)? (y/n): ").strip().lower()
                if keep != "y":
                    print(f"  Discarded trial")
                    continue

                for f in frames:
                    events_writer.writerow({
                        "frame_index": f["_frame_index"],
                        "timestamp": f["timestamp"],
                        "gesture": f["gesture"],
                        "trial": f["trial"],
                        "elapsed": round(f["elapsed"], 6),
                    })

                for name in args.sensors:
                    for f in frames:
                        entry = f.get(name, {})
                        if "error" in entry:
                            continue
                        reading = Reading(
                            sensor_id="",
                            sensor_type=entry.get("sensor_type", name),
                            data=entry.get("data", {}),
                            confidence=entry.get("confidence", 0.0),
                        )
                        row = _flatten_reading(reading, f["_frame_index"])
                        writers[name].writerow(row)

                trials_writer.writerow({
                    "trial_index": trial_index,
                    "gesture": gesture,
                    "trial_num": trial_num,
                    "start_timestamp": trial_start_ts,
                    "end_timestamp": trial_end_ts,
                    "num_frames": len(frames),
                })

                total_frames += len(frames)
                trial_index += 1

        print("=" * 60)

    finally:
        for name, reader in reader_map.items():
            try:
                reader.stop()
            except Exception:
                pass
        for fh in file_handles:
            try:
                fh.close()
            except Exception:
                pass

    metadata = {
        "session": session_time,
        "gestures": args.gestures,
        "trials_per_gesture": args.trials,
        "duration_s": args.duration,
        "fps": args.fps,
        "sensors": args.sensors,
        "mode": args.mode,
        "total_frames": total_frames,
        "num_trials": trial_index,
    }
    with open(session_dir / "session_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSession metadata saved to {session_dir / 'session_metadata.json'}")

    print(f"Total: {total_frames} frames across {trial_index} trials")


if __name__ == "__main__":
    main()