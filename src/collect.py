from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from typing import Any

import numpy as np

from src.sensors.base_reader import BaseReader, Reading
from src.sensors.imu_reader import ImuReader
from src.sensors.mmWave.mmwave_reader import MmWaveReader
from src.sensors.rfid_reader import RfidReader
from src.sensors.wifi_reader import WiFiReader

# ---------------------------------------------------------------------------
# UWB – UCI library import (lazy)
# ---------------------------------------------------------------------------
_UCI = None
_UCI_IMPORTED = False


def _ensure_uci(lib_path: str | None = None) -> bool:
    global _UCI, _UCI_IMPORTED
    if _UCI_IMPORTED:
        return True
    try:
        if lib_path:
            sys.path.insert(0, os.path.expanduser(lib_path))
        import uci as m
        _UCI = m
        _UCI_IMPORTED = True
        return True
    except ImportError:
        return False


_ENG_URSK_PREFIX = "ed07a80d2beb00f785af2627"


def _ursk(session_id: int) -> bytes:
    return bytes.fromhex(_ENG_URSK_PREFIX + session_id.to_bytes(4, "big").hex())


# ---------------------------------------------------------------------------
# UWB reader — wraps the 3-board UCI ranging session
# ---------------------------------------------------------------------------
class UwbReader(BaseReader):
    """Reader for a 1-anchor + 2-tag UWB ranging session (Qorvo UCI)."""

    def __init__(
        self,
        sensor_id: str = "uwb-0",
        mode: str = "mock",
        anchor_port: str = "/dev/ttyACM0",
        left_tag_port: str = "/dev/ttyACM1",
        right_tag_port: str = "/dev/ttyACM2",
        anchor_mac: int = 0x0,
        left_tag_mac: int = 0x1,
        right_tag_mac: int = 0x2,
        session_id: int = 0x42,
        channel: int = 5,
        preamble_idx: int = 10,
        sfd: int = 2,
        slot_duration: int = 2400,
        ranging_interval: int = 200,
        slots_per_rr: int = 25,
        vendor_id: int = 0x0708,
        static_sts: int = 0x060504030201,
        uci_lib_path: str | None = None,
    ):
        super().__init__(sensor_id=sensor_id, sensor_type="uwb")
        self._mode = mode
        self._anchor_port = anchor_port
        self._tag_ports = [left_tag_port, right_tag_port]
        self._anchor_mac = anchor_mac
        self._tag_macs = [left_tag_mac, right_tag_mac]
        self._session_id = session_id
        self._uci_lib_path = uci_lib_path
        self._radio = dict(
            channel=channel, preamble_idx=preamble_idx, sfd=sfd,
            slot_duration=slot_duration, ranging_interval=ranging_interval,
            slots_per_rr=slots_per_rr, vendor_id=vendor_id, static_sts=static_sts,
        )
        self._range_queue: Queue = Queue()
        self._clients: list[tuple[str, Any]] = []
        self._session_handles: dict[str, int] = {}
        self._started = False

    def start(self) -> None:
        if self._mode == "mock":
            self._started = True
            return

        if not _ensure_uci(self._uci_lib_path):
            raise ImportError(
                "Qorvo UCI library not found. Install ~/UWB_lab/uwb-qorvo-tools/lib "
                "or use --mode mock"
            )

        notif_handlers = {
            (_UCI.Gid.Ranging, _UCI.OidRanging.Start): lambda p: self._range_queue.put(p),
            ("default", "default"): lambda gid, oid, x: None,
        }

        ctrl = _UCI.Client(port=self._anchor_port, notif_handlers=notif_handlers)
        self._clients.append(("anchor", ctrl))
        self._configure_controller(ctrl)

        for i, (port, mac) in enumerate(zip(self._tag_ports, self._tag_macs)):
            tag_client = _UCI.Client(port=port)
            self._clients.append((f"tag_{i}", tag_client))
            self._configure_controlee(tag_client, mac)

        for cname, client in self._clients:
            if cname.startswith("tag_"):
                sh = self._session_handles.get(cname)
                if sh is not None:
                    client.ranging_start(sh)

        anchor_sh = self._session_handles.get("anchor")
        if anchor_sh is not None:
            ctrl.ranging_start(anchor_sh)

        self._started = True

    def read(self) -> Reading:
        if self._mode == "mock":
            return Reading(
                sensor_id=self.sensor_id,
                sensor_type=self.sensor_type,
                data={
                    "ranges_cm": [
                        round(random.uniform(30, 300), 1),
                        round(random.uniform(30, 300), 1),
                    ],
                    "position": None,
                    "raw_ranges": [],
                },
                confidence=0.8,
            )

        try:
            payload = self._range_queue.get(timeout=0.05)
        except Empty:
            return Reading(
                sensor_id=self.sensor_id,
                sensor_type=self.sensor_type,
                data={"ranges_cm": [], "position": None, "raw_ranges": []},
                confidence=0.0,
            )

        try:
            rd = _UCI.RangingData(payload)
        except Exception:
            return Reading(
                sensor_id=self.sensor_id,
                sensor_type=self.sensor_type,
                data={"ranges_cm": [], "position": None, "raw_ranges": []},
                confidence=0.0,
            )

        ranges_cm: list[float] = []
        confidences: list[float] = []

        for m in rd.meas:
            if not hasattr(m, "distance"):
                continue
            status_ok = m.status == _UCI.Status.Ok
            ranges_cm.append(float(m.distance) if status_ok else 0.0)
            confidence = (
                0.9 if status_ok and not getattr(m, "nlos", False) else
                0.5 if status_ok else 0.0
            )
            confidences.append(confidence)

        return Reading(
            sensor_id=self.sensor_id,
            sensor_type=self.sensor_type,
            data={"ranges_cm": ranges_cm, "position": None, "raw_ranges": []},
            confidence=float(np.mean(confidences)) if confidences else 0.0,
        )

    def stop(self) -> None:
        if not self._started:
            return
        for cname, client in reversed(self._clients):
            sh = self._session_handles.get(cname)
            if sh is not None:
                try:
                    client.ranging_stop(sh)
                except Exception:
                    pass
                try:
                    client.session_deinit(sh)
                except Exception:
                    pass
            try:
                client.close()
            except Exception:
                pass
        self._clients.clear()
        self._session_handles.clear()
        self._started = False

    # -- UCI configuration helpers -----------------------------------------

    def _configure_controller(self, client: Any) -> None:
        uci = _UCI
        rts, sh = client.session_init(self._session_id, uci.SessionType.Ranging)
        if rts != uci.Status.Ok:
            raise RuntimeError(f"session_init (controller): {rts}")
        if sh is None:
            sh = self._session_id
        self._session_handles["anchor"] = sh

        cfg = [
            (uci.App.DeviceType, uci.DeviceType.Controller),
            (uci.App.DeviceRole, uci.DeviceRole.Initiator),
            (uci.App.MultiNodeMode, uci.Node.OneToMAny),
            (uci.App.RangingRoundUsage, uci.RangingRound.DsTwrDeferred),
            (uci.App.DeviceMacAddress, self._anchor_mac),
            (uci.App.ChannelNumber, self._radio["channel"]),
            (uci.App.ScheduleMode, 1),
            (uci.App.StsConfig, uci.StsConfig.Static),
            (uci.App.RframeConfig, uci.RfFrame.Qp3),
            (uci.App.ResultReportConfig, 9),
            (uci.App.VendorId, self._radio["vendor_id"]),
            (uci.App.StaticStsIv, self._radio["static_sts"]),
            (uci.App.AoaResultReq, 0),
            (uci.App.UwbInitiationTime, 0),
            (uci.App.PreambleCodeIndex, self._radio["preamble_idx"]),
            (uci.App.SfdId, self._radio["sfd"]),
            (uci.App.SlotDuration, self._radio["slot_duration"]),
            (uci.App.RangingInterval, self._radio["ranging_interval"]),
            (uci.App.SlotsPerRr, self._radio["slots_per_rr"]),
            (uci.App.MaxNumberOfMeasurements, 0),
            (uci.App.HoppingMode, 0),
            (uci.App.RssiReporting, 1),
            (uci.App.BlockStrideLength, 0),
            (uci.App.NumberOfControlees, len(self._tag_macs)),
            (uci.App.DstMacAddress, self._tag_macs),
            (uci.App.SessionKey, _ursk(self._session_id)),
        ]
        rts, msg = client.session_set_app_config(sh, cfg)
        if rts != uci.Status.Ok:
            raise RuntimeError(f"session_set_app_config (controller): {rts} {msg}")

    def _configure_controlee(self, client: Any, tag_mac: int) -> None:
        uci = _UCI
        rts, sh = client.session_init(self._session_id, uci.SessionType.Ranging)
        if rts != uci.Status.Ok:
            raise RuntimeError(f"session_init (tag 0x{tag_mac:04x}): {rts}")
        if sh is None:
            sh = self._session_id
        tag_key = f"tag_{self._tag_macs.index(tag_mac)}"
        self._session_handles[tag_key] = sh

        cfg = [
            (uci.App.DeviceType, uci.DeviceType.Controlee),
            (uci.App.DeviceRole, uci.DeviceRole.Responder),
            (uci.App.MultiNodeMode, uci.Node.OneToMAny),
            (uci.App.RangingRoundUsage, uci.RangingRound.DsTwrDeferred),
            (uci.App.DeviceMacAddress, tag_mac),
            (uci.App.ChannelNumber, self._radio["channel"]),
            (uci.App.ScheduleMode, 1),
            (uci.App.StsConfig, uci.StsConfig.Static),
            (uci.App.RframeConfig, uci.RfFrame.Qp3),
            (uci.App.ResultReportConfig, 9),
            (uci.App.VendorId, self._radio["vendor_id"]),
            (uci.App.StaticStsIv, self._radio["static_sts"]),
            (uci.App.AoaResultReq, 0),
            (uci.App.UwbInitiationTime, 0),
            (uci.App.PreambleCodeIndex, self._radio["preamble_idx"]),
            (uci.App.SfdId, self._radio["sfd"]),
            (uci.App.SlotDuration, self._radio["slot_duration"]),
            (uci.App.RangingInterval, self._radio["ranging_interval"]),
            (uci.App.SlotsPerRr, self._radio["slots_per_rr"]),
            (uci.App.MaxNumberOfMeasurements, 0),
            (uci.App.HoppingMode, 0),
            (uci.App.RssiReporting, 1),
            (uci.App.DstMacAddress, [self._anchor_mac]),
            (uci.App.SessionKey, _ursk(self._session_id)),
        ]
        rts, msg = client.session_set_app_config(sh, cfg)
        if rts != uci.Status.Ok:
            raise RuntimeError(f"session_set_app_config (tag 0x{tag_mac:04x}): {rts} {msg}")


# ---------------------------------------------------------------------------
# Sensor registry
# ---------------------------------------------------------------------------
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
    prompt: bool = True,
) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []

    print(f"  Recording '{gesture}' trial {trial} for {duration_s}s...")
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

    # UWB 3-board ports
    parser.add_argument("--uwb-anchor-port", default="/dev/ttyACM0",
                        help="Serial port for the anchor board")
    parser.add_argument("--uwb-left-tag-port", default="/dev/ttyACM1",
                        help="Serial port for the left wrist tag")
    parser.add_argument("--uwb-right-tag-port", default="/dev/ttyACM2",
                        help="Serial port for the right wrist tag")
    # UWB MACs & session
    parser.add_argument("--uwb-anchor-mac", default="0x0")
    parser.add_argument("--uwb-left-tag-mac", default="0x1")
    parser.add_argument("--uwb-right-tag-mac", default="0x2")
    parser.add_argument("--uwb-session-id", default="0x42")
    # UWB radio config
    parser.add_argument("--uwb-channel", type=int, default=5)
    parser.add_argument("--uwb-preamble-idx", type=int, default=10)
    parser.add_argument("--uwb-sfd", type=int, default=2)
    parser.add_argument("--uwb-slot-duration", type=int, default=2400)
    parser.add_argument("--uwb-ranging-interval", type=int, default=200)
    parser.add_argument("--uwb-slots-per-rr", type=int, default=25)
    parser.add_argument("--uwb-vendor-id", type=int, default=0x0708)
    parser.add_argument("--uwb-static-sts", type=int, default=0x060504030201)
    parser.add_argument("--uwb-lib-path", default=None,
                        help="Path to Qorvo UCI library (e.g. ~/UWB_lab/uwb-qorvo-tools/lib)")
    parser.add_argument("--countdown", type=float, default=0,
                        help="Seconds to wait (countdown) before each prompted trial")

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
            reader = cls(mode=args.mode, serial_port=args.mmwave_port,
                         cfg_path="config/point_cloud.cfg")
        elif name == "imu":
            reader = cls(mode=args.mode, serial_port=args.imu_port)
        elif name == "uwb":
            reader = cls(
                mode=args.mode,
                anchor_port=args.uwb_anchor_port,
                left_tag_port=args.uwb_left_tag_port,
                right_tag_port=args.uwb_right_tag_port,
                anchor_mac=int(args.uwb_anchor_mac, 0),
                left_tag_mac=int(args.uwb_left_tag_mac, 0),
                right_tag_mac=int(args.uwb_right_tag_mac, 0),
                session_id=int(args.uwb_session_id, 0),
                channel=args.uwb_channel,
                preamble_idx=args.uwb_preamble_idx,
                sfd=args.uwb_sfd,
                slot_duration=args.uwb_slot_duration,
                ranging_interval=args.uwb_ranging_interval,
                slots_per_rr=args.uwb_slots_per_rr,
                vendor_id=args.uwb_vendor_id,
                static_sts=args.uwb_static_sts,
                uci_lib_path=args.uwb_lib_path,
            )
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
                if args.countdown > 0 and args.no_prompt:
                    print(f"  Starting in {args.countdown}s...")
                    time.sleep(args.countdown)

                trial_start_ts = datetime.now(timezone.utc).isoformat()
                frames = collect_gesture(
                    reader_map, gesture, args.duration,
                    args.fps, trial_num,
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
