#!/usr/bin/env python3
"""
Collect labeled gesture data from 3-board UWB setup (1 anchor + 2 wrist tags).

Writes JSONL files to ``data/raw/`` in the format expected by the pipeline
(combine_datasets.py -> extract_features.py -> train.py).

Usage
-----
  conda run -n py39 python scripts/collect_uwb_gestures.py \\
    --anchor-port /dev/cu.usbmodemE89A5C6EB9A71 \\
    --left-tag-port /dev/cu.usbmodemFDADDB2EC1651 \\
    --right-tag-port /dev/cu.usbmodemD46FFE3655DD1 \\
    --duration 5 --trials 5
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from typing import Optional

sys.path.insert(0, os.path.expanduser("~/UWB_lab/uwb-qorvo-tools/lib"))
from uci import *

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
log = logging.getLogger("collect_uwb")

ENG_URSK_PREFIX = "ed07a80d2beb00f785af2627"

TARGET_GESTURES = ["none", "clapping", "two-arm-boxing"]


def _ursk(session_id: int) -> bytes:
    return bytes.fromhex(
        ENG_URSK_PREFIX + session_id.to_bytes(4, "big").hex()
    )


def configure_controller(
    client: Client, session_id: int, anchor_mac: int,
    tag_macs: list[int], args,
) -> int:
    rts, sh = client.session_init(session_id, SessionType.Ranging)
    if rts != Status.Ok:
        raise RuntimeError(f"session_init (controller): {rts}")
    if sh is None:
        sh = session_id

    cfg = [
        (App.DeviceType, DeviceType.Controller),
        (App.DeviceRole, DeviceRole.Initiator),
        (App.MultiNodeMode, Node.OneToMAny),
        (App.RangingRoundUsage, RangingRound.DsTwrDeferred),
        (App.DeviceMacAddress, anchor_mac),
        (App.ChannelNumber, args.channel),
        (App.ScheduleMode, 1),
        (App.StsConfig, StsConfig.Static),
        (App.RframeConfig, RfFrame.Qp3),
        (App.ResultReportConfig, 9),
        (App.VendorId, args.vendor_id),
        (App.StaticStsIv, args.static_sts),
        (App.AoaResultReq, 0),
        (App.UwbInitiationTime, 0),
        (App.PreambleCodeIndex, args.preamble_idx),
        (App.SfdId, args.sfd),
        (App.SlotDuration, args.slot_duration),
        (App.RangingInterval, args.ranging_interval),
        (App.SlotsPerRr, args.slots_per_rr),
        (App.MaxNumberOfMeasurements, 0),
        (App.HoppingMode, 0),
        (App.RssiReporting, 1),
        (App.BlockStrideLength, 0),
        (App.NumberOfControlees, len(tag_macs)),
        (App.DstMacAddress, tag_macs),
        (App.SessionKey, _ursk(session_id)),
    ]
    rts, msg = client.session_set_app_config(sh, cfg)
    if rts != Status.Ok:
        raise RuntimeError(f"session_set_app_config (controller): {rts} {msg}")
    print(f"  Controller configured (anchor=0x{anchor_mac:04x}, "
          f"tags={[hex(m) for m in tag_macs]})", file=sys.stderr)
    return sh


def configure_controlee(
    client: Client, session_id: int, tag_mac: int,
    anchor_mac: int, args,
) -> int:
    rts, sh = client.session_init(session_id, SessionType.Ranging)
    if rts != Status.Ok:
        raise RuntimeError(f"session_init (tag 0x{tag_mac:04x}): {rts}")
    if sh is None:
        sh = session_id

    cfg = [
        (App.DeviceType, DeviceType.Controlee),
        (App.DeviceRole, DeviceRole.Responder),
        (App.MultiNodeMode, Node.OneToMAny),
        (App.RangingRoundUsage, RangingRound.DsTwrDeferred),
        (App.DeviceMacAddress, tag_mac),
        (App.ChannelNumber, args.channel),
        (App.ScheduleMode, 1),
        (App.StsConfig, StsConfig.Static),
        (App.RframeConfig, RfFrame.Qp3),
        (App.ResultReportConfig, 9),
        (App.VendorId, args.vendor_id),
        (App.StaticStsIv, args.static_sts),
        (App.AoaResultReq, 0),
        (App.UwbInitiationTime, 0),
        (App.PreambleCodeIndex, args.preamble_idx),
        (App.SfdId, args.sfd),
        (App.SlotDuration, args.slot_duration),
        (App.RangingInterval, args.ranging_interval),
        (App.SlotsPerRr, args.slots_per_rr),
        (App.MaxNumberOfMeasurements, 0),
        (App.HoppingMode, 0),
        (App.RssiReporting, 1),
        (App.DstMacAddress, [anchor_mac]),
        (App.SessionKey, _ursk(session_id)),
    ]
    rts, msg = client.session_set_app_config(sh, cfg)
    if rts != Status.Ok:
        raise RuntimeError(f"session_set_app_config (tag 0x{tag_mac:04x}): {rts} {msg}")
    print(f"  Tag 0x{tag_mac:04x} configured", file=sys.stderr)
    return sh


def record_trial(
    range_queue: Queue,
    gesture: str,
    trial: int,
    duration_s: float,
    output_dir: Path,
    no_prompt: bool = False,
    countdown_s: float = 2.0,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"{gesture}_t{trial:03d}_{timestamp}.jsonl"

    print(f"  Recording '{gesture}' trial {trial} for {duration_s}s...", file=sys.stderr)
    if gesture != "none":
        if no_prompt:
            print(f"  Starting in {countdown_s} seconds...", file=sys.stderr)
            time.sleep(countdown_s)
        else:
            try:
                input("  Press Enter when ready to perform gesture...")
            except (EOFError, OSError):
                print("  (stdin unavailable, starting immediately)", file=sys.stderr)

    deadline = time.monotonic() + duration_s
    frames = []
    start_time = time.monotonic()

    while time.monotonic() < deadline:
        try:
            payload = range_queue.get(timeout=0.05)
        except Empty:
            continue

        try:
            rd = RangingData(payload)
        except Exception:
            continue

        distances_cm = []
        confidences = []
        for m in rd.meas:
            if not hasattr(m, "distance"):
                continue
            status_ok = m.status == Status.Ok
            if status_ok:
                distances_cm.append(float(m.distance))
            else:
                distances_cm.append(0.0)
            confidence = 0.9 if status_ok and not m.nlos else (
                0.5 if status_ok else 0.0
            )
            confidences.append(confidence)

        frame = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "gesture": gesture,
            "trial": trial,
            "elapsed": time.monotonic() - start_time,
            "uwb": {
                "data": {
                    "ranges_cm": distances_cm,
                },
                "confidence": float(
                    sum(confidences) / len(confidences)
                ) if confidences else 0.0,
                "sensor_type": "uwb",
            },
        }
        frames.append(frame)

    with open(out_path, "w") as f:
        for frame in frames:
            f.write(json.dumps(frame) + "\n")

    print(f"  Saved {len(frames)} frames to {out_path}", file=sys.stderr)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Collect labeled UWB gesture data from 3-board setup"
    )
    parser.add_argument("--anchor-port", required=True,
                        help="Serial port for the anchor board")
    parser.add_argument("--left-tag-port", required=True,
                        help="Serial port for the left wrist tag")
    parser.add_argument("--right-tag-port", required=True,
                        help="Serial port for the right wrist tag")
    parser.add_argument("--anchor-mac", default="0x0")
    parser.add_argument("--left-tag-mac", default="0x1")
    parser.add_argument("--right-tag-mac", default="0x2")
    parser.add_argument("--session-id", default="0x42")
    parser.add_argument("--gestures", nargs="+", default=TARGET_GESTURES,
                        help=f"Gestures to collect (default: {TARGET_GESTURES})")
    parser.add_argument("--trials", type=int, default=5,
                        help="Trials per gesture (default: 5)")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Seconds per trial (default: 5)")
    parser.add_argument("--output", default="data/raw",
                        help="Output directory (default: data/raw)")
    parser.add_argument("--no-prompt", action="store_true",
                        help="Skip 'press enter' prompts (use with conda run)")
    parser.add_argument("--countdown", type=float, default=2.0,
                        help="Countdown seconds before each prompted trial (default: 2)")

    parser.add_argument("--channel", type=int, default=5)
    parser.add_argument("--preamble-idx", type=int, default=10)
    parser.add_argument("--sfd", type=int, default=2)
    parser.add_argument("--slot-duration", type=int, default=2400)
    parser.add_argument("--ranging-interval", type=int, default=200)
    parser.add_argument("--slots-per-rr", type=int, default=25)
    parser.add_argument("--vendor-id", type=int, default=0x0708)
    parser.add_argument("--static-sts", type=int, default=0x060504030201)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    session_id = int(args.session_id, 0)
    anchor_mac = int(args.anchor_mac, 0)
    tag_macs = [int(args.left_tag_mac, 0), int(args.right_tag_mac, 0)]
    tag_ports = [args.left_tag_port, args.right_tag_port]

    range_queue: Queue = Queue()

    notif_handlers = {
        (Gid.Ranging, OidRanging.Start): lambda payload: range_queue.put(payload),
        ("default", "default"): lambda gid, oid, x: None,
    }

    clients = []
    session_handles = {}

    try:
        print("Connecting to anchor (controller)...", file=sys.stderr)
        ctrl = Client(port=args.anchor_port, notif_handlers=notif_handlers)
        clients.append(("anchor", ctrl))
        session_handles["anchor"] = configure_controller(
            ctrl, session_id, anchor_mac, tag_macs, args
        )

        tag_names = ["left", "right"]
        for tag_name, port, mac in zip(tag_names, tag_ports, tag_macs):
            print(f"Connecting to {tag_name} tag 0x{mac:04x} on {port}...",
                  file=sys.stderr)
            tag_client = Client(port=port)
            clients.append((f"tag_{tag_name}", tag_client))
            session_handles[f"tag_{tag_name}"] = configure_controlee(
                tag_client, session_id, mac, anchor_mac, args
            )

        for cname, client in clients:
            if cname.startswith("tag_"):
                sh = session_handles[cname]
                print(f"  Starting ranging on {cname}...", file=sys.stderr)
                rts = client.ranging_start(sh)
                if rts != Status.Ok:
                    raise RuntimeError(f"ranging_start ({cname}): {rts}")

        ctrl_sh = session_handles["anchor"]
        print("  Starting ranging on anchor...", file=sys.stderr)
        rts = ctrl.ranging_start(ctrl_sh)
        if rts != Status.Ok:
            raise RuntimeError(f"ranging_start (anchor): {rts}")

        print("\nRanging active. Ready to collect.\n", file=sys.stderr)

        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        all_files = []
        for gesture in args.gestures:
            if gesture not in TARGET_GESTURES:
                print(f"  Warning: '{gesture}' not in standard list: {TARGET_GESTURES}",
                      file=sys.stderr)
            for trial in range(args.trials):
                path = record_trial(
                    range_queue, gesture, trial,
                    args.duration, output_dir,
                    no_prompt=args.no_prompt,
                    countdown_s=args.countdown,
                )
                all_files.append(path)

        manifest = {
            "gestures": args.gestures,
            "trials_per_gesture": args.trials,
            "duration_s": args.duration,
            "sensors": ["uwb"],
            "anchor_port": args.anchor_port,
            "left_tag_port": args.left_tag_port,
            "right_tag_port": args.right_tag_port,
            "files": [str(p.relative_to(output_dir)) for p in all_files],
            "total_files": len(all_files),
        }
        manifest_path = output_dir / "manifest_uwb.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\nManifest saved: {manifest_path}", file=sys.stderr)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise
    finally:
        print("\nStopping ranging...", file=sys.stderr)
        for cname, client in reversed(clients):
            if cname in session_handles:
                sh = session_handles[cname]
                try:
                    client.ranging_stop(sh)
                except Exception as e:
                    log.warning(f"ranging_stop ({cname}): {e}")
        for cname, client in reversed(clients):
            if cname in session_handles:
                sh = session_handles[cname]
                try:
                    client.session_deinit(sh)
                except Exception as e:
                    log.warning(f"session_deinit ({cname}): {e}")
        for cname, client in reversed(clients):
            try:
                client.close()
            except Exception:
                pass
        print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
