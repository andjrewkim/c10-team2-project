#!/usr/bin/env python3
"""
Multi-controlee FIRA TWR ranging: 1 anchor (controller) + N tags (controlees).

Connects to each DWM3001CDK board over UCI, configures sessions, starts
ranging, and streams range data as JSONL lines to stdout.

Usage
-----
  conda run -n py39 python scripts/run_uwb_multitag.py \\
    --anchor-port /dev/cu.usbmodemFDADDB2EC1651 \\
    --tag-ports /dev/cu.usbmodemE89A5C6EB9A71 /dev/cu.usbmodemD46FFE3655DD1 \\
    --duration -1

Each output line::
  {"timestamp": "2025-01-01T00:00:00.123456+00:00",
   "sequence": 42,
   "measurements": [
     {"mac": "00:01", "distance_cm": 123.4, "status": "Ok",
      "nlos": false, "rssi": -65.3, "confidence": 0.9},
     {"mac": "00:02", "distance_cm": 234.5, "status": "Ok",
      "nlos": true, "rssi": -72.1, "confidence": 0.7}
   ]}
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from queue import Empty, Queue
from typing import Optional

sys.path.insert(0, os.path.expanduser("~/UWB_lab/uwb-qorvo-tools/lib"))
from uci import *

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
log = logging.getLogger("uwb_multitag")


ENG_URSK_PREFIX = "ed07a80d2beb00f785af2627"


def _ursk(session_id: int) -> bytes:
    return bytes.fromhex(
        ENG_URSK_PREFIX + session_id.to_bytes(4, "big").hex()
    )


def make_client(port: str, notif_handlers: Optional[dict] = None) -> Client:
    return Client(port=port, notif_handlers=notif_handlers or {})


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
    print(f"  Controller session {sh} configured (anchor=0x{anchor_mac:04x}, "
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
        (App.BlockStrideLength, 0),
        (App.DstMacAddress, [anchor_mac]),
        (App.SessionKey, _ursk(session_id)),
    ]
    rts, msg = client.session_set_app_config(sh, cfg)
    if rts != Status.Ok:
        raise RuntimeError(f"session_set_app_config (tag 0x{tag_mac:04x}): {rts} {msg}")
    print(f"  Tag 0x{tag_mac:04x} session {sh} configured", file=sys.stderr)
    return sh


def run_ranging(args):
    session_id = int(args.session_id, 0)
    anchor_mac = int(args.anchor_mac, 0)
    tag_macs = [int(m, 0) for m in args.tag_macs]

    range_queue: Queue = Queue()

    notif_handlers = {
        (Gid.Ranging, OidRanging.Start): lambda payload: range_queue.put(payload),
        ("default", "default"): lambda gid, oid, x: None,
    }

    clients = []
    session_handles = {}

    try:
        print(f"Connecting to anchor (controller) on {args.anchor_port}...",
              file=sys.stderr)
        ctrl = make_client(args.anchor_port, notif_handlers)
        clients.append(("anchor", ctrl))
        session_handles["anchor"] = configure_controller(
            ctrl, session_id, anchor_mac, tag_macs, args
        )

        for i, (port, mac) in enumerate(zip(args.tag_ports, tag_macs)):
            print(f"Connecting to tag 0x{mac:04x} on {port}...", file=sys.stderr)
            tag_client = make_client(port)
            clients.append((f"tag_{i}", tag_client))
            session_handles[f"tag_{i}"] = configure_controlee(
                tag_client, session_id, mac, anchor_mac, args
            )

        for name in clients:
            if name[0].startswith("tag"):
                client = name[1]
                sh = session_handles[name[0]]
                print(f"  Starting ranging on {name[0]}...", file=sys.stderr)
                rts = client.ranging_start(sh)
                if rts != Status.Ok:
                    raise RuntimeError(f"ranging_start ({name[0]}): {rts}")

        ctrl_sh = session_handles["anchor"]
        print(f"  Starting ranging on anchor...", file=sys.stderr)
        rts = ctrl.ranging_start(ctrl_sh)
        if rts != Status.Ok:
            raise RuntimeError(f"ranging_start (anchor): {rts}")

        print(f"Ranging active. Streaming JSONL to stdout...", file=sys.stderr)

        deadline = None
        if args.duration and args.duration > 0:
            deadline = time.monotonic() + args.duration

        running = True

        def _handle_sigint(sig, frame):
            nonlocal running
            running = False
        signal.signal(signal.SIGINT, _handle_sigint)

        while running:
            if deadline and time.monotonic() >= deadline:
                break
            try:
                payload = range_queue.get(timeout=0.1)
            except Empty:
                continue

            try:
                rd = RangingData(payload)
            except Exception as e:
                log.warning(f"Failed to decode RangingData: {e}")
                continue

            measurements = []
            for m in rd.meas:
                if not hasattr(m, "distance"):
                    continue
                status_ok = m.status == Status.Ok
                confidence = 0.9 if status_ok and not m.nlos else (
                    0.5 if status_ok else 0.0
                )
                measurements.append({
                    "mac": m.mac_add,
                    "distance_cm": float(m.distance),
                    "status": m.status.name,
                    "nlos": bool(m.nlos),
                    "rssi": float(m.rssi),
                    "confidence": confidence,
                })

            line = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "sequence": rd.idx,
                "session_handle": rd.session_handle,
                "measurements": measurements,
            }
            sys.stdout.write(json.dumps(line) + "\n")
            sys.stdout.flush()

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


def main():
    parser = argparse.ArgumentParser(
        description="Multi-tag UWB ranging via FIRA TWR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--anchor-port", required=True,
                        help="Serial port for the anchor (controller) board")
    parser.add_argument("--tag-ports", required=True, nargs="+",
                        help="Serial ports for the tag (controlee) boards")
    parser.add_argument("--anchor-mac", default="0x0",
                        help="Anchor short MAC address (default: 0x0)")
    parser.add_argument("--tag-macs", nargs="+",
                        default=["0x1", "0x2"],
                        help="Tag short MAC addresses (default: 0x1 0x2)")
    parser.add_argument("--session-id", default="0x42",
                        help="FIRA session ID (default: 0x42)")
    parser.add_argument("--duration", type=float, default=-1,
                        help="Ranging duration in seconds (-1 = forever)")
    parser.add_argument("--channel", type=int, default=5,
                        help="UWB channel number (default: 9)")
    parser.add_argument("--preamble-idx", type=int, default=10,
                        help="Preamble code index (default: 10)")
    parser.add_argument("--sfd", type=int, default=2,
                        help="SFD ID (default: 2)")
    parser.add_argument("--slot-duration", type=int, default=2400,
                        help="Slot duration in RSTU units (default: 2400)")
    parser.add_argument("--ranging-interval", type=int, default=200,
                        help="Ranging interval in ms (default: 200)")
    parser.add_argument("--slots-per-rr", type=int, default=25,
                        help="Slots per ranging round (default: 25)")
    parser.add_argument("--vendor-id", type=int, default=0x0708,
                        help="Vendor ID (default: 0x0708)")
    parser.add_argument("--static-sts", type=int, default=0x060504030201,
                        help="Static STS IV (default: 0x060504030201)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable debug logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if len(args.tag_ports) != len(args.tag_macs):
        parser.error("--tag-ports and --tag-macs must have the same count")

    run_ranging(args)


if __name__ == "__main__":
    main()
