#!/usr/bin/env python3
import argparse
import json
import logging
import os
import pickle
import signal
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from typing import Optional

sys.path.insert(0, os.path.expanduser("~/UWB_lab/uwb-qorvo-tools/lib"))
from uci import *

import numpy as np

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
log = logging.getLogger("realtime_uwb_demo")

ENG_URSK_PREFIX = "ed07a80d2beb00f785af2627"

WINDOW_SIZE = 10
STRIDE = 5


def _ursk(session_id: int) -> bytes:
    return bytes.fromhex(ENG_URSK_PREFIX + session_id.to_bytes(4, "big").hex())


def configure_controller(client, session_id, anchor_mac, tag_macs, args):
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
    print(f"  Controller configured (anchor=0x{anchor_mac:04x}, tags={[hex(m) for m in tag_macs]})", file=sys.stderr)
    return sh


def configure_controlee(client, session_id, tag_mac, anchor_mac, args):
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
    print(f"  Tag 0x{tag_mac:04x} configured", file=sys.stderr)
    return sh


MAX_VALID_CM = 5000

def extract_features(ranges_m: list[float]) -> list[float]:
    if not ranges_m:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    return [
        float(len(ranges_m)),
        float(np.mean(ranges_m)),
        float(np.std(ranges_m)) if len(ranges_m) > 1 else 0.0,
        float(np.min(ranges_m)),
        float(np.max(ranges_m)),
        float(np.median(ranges_m)),
    ]


def main():
    parser = argparse.ArgumentParser(description="Real-time UWB gesture detection")
    parser.add_argument("--anchor-port", required=True)
    parser.add_argument("--left-tag-port", required=True)
    parser.add_argument("--right-tag-port", required=True)
    parser.add_argument("--anchor-mac", default="0x0")
    parser.add_argument("--left-tag-mac", default="0x1")
    parser.add_argument("--right-tag-mac", default="0x2")
    parser.add_argument("--session-id", default="0x42")
    parser.add_argument("--model", default="models/best_model.pkl")
    parser.add_argument("--features", default="data/processed/features.npz")

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

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)

    features_path = Path(args.features)
    gestures = []
    if features_path.exists():
        data = dict(np.load(features_path, allow_pickle=True))
        if "gestures" in data:
            gestures = list(data["gestures"])

    print(f"Loaded model: {model_path}", file=sys.stderr)
    if gestures:
        print(f"Gestures: {', '.join(gestures)}", file=sys.stderr)

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
        print("Connecting to anchor...", file=sys.stderr)
        ctrl = Client(port=args.anchor_port, notif_handlers=notif_handlers)
        clients.append(("anchor", ctrl))
        session_handles["anchor"] = configure_controller(ctrl, session_id, anchor_mac, tag_macs, args)

        for tag_name, port, mac in zip(["left", "right"], tag_ports, tag_macs):
            print(f"Connecting to {tag_name} tag 0x{mac:04x} on {port}...", file=sys.stderr)
            tag_client = Client(port=port)
            clients.append((f"tag_{tag_name}", tag_client))
            session_handles[f"tag_{tag_name}"] = configure_controlee(tag_client, session_id, mac, anchor_mac, args)

        for cname, client in clients:
            if cname.startswith("tag_"):
                sh = session_handles[cname]
                rts = client.ranging_start(sh)
                if rts != Status.Ok:
                    raise RuntimeError(f"ranging_start ({cname}): {rts}")

        ctrl_sh = session_handles["anchor"]
        rts = ctrl.ranging_start(ctrl_sh)
        if rts != Status.Ok:
            raise RuntimeError(f"ranging_start (anchor): {rts}")

        print("\nRanging active. Real-time demo running...\n", file=sys.stderr)
        print("Make a gesture (clap or box) and watch the prediction.\n", file=sys.stderr)
        print(f"{'Prediction':>15s}  {'Conf.':>5s}  {'Distances (L/R cm)':>25s}", file=sys.stderr)

        frame_buffer: deque = deque(maxlen=WINDOW_SIZE)
        frames_since_pred = 0
        pred_count = 0
        running = True

        def _handle_sigint(sig, frame):
            nonlocal running
            running = False
        signal.signal(signal.SIGINT, _handle_sigint)

        while running:
            try:
                payload = range_queue.get(timeout=0.05)
            except Empty:
                continue

            try:
                rd = RangingData(payload)
            except Exception:
                continue

            distances_cm = []
            for m in rd.meas:
                if hasattr(m, "distance"):
                    d = float(m.distance)
                    if 0 < d < MAX_VALID_CM:
                        distances_cm.append(d)

            frame_buffer.append(distances_cm)

            if len(frame_buffer) >= WINDOW_SIZE:
                frames_since_pred += 1
                if frames_since_pred < STRIDE:
                    continue
                frames_since_pred = 0

                all_ranges = [r for frame in frame_buffer for r in frame]
                features = extract_features([r / 100.0 for r in all_ranges])

                X = np.array(features).reshape(1, -1)
                pred = pipeline.predict(X)[0]

                conf = 0.0
                if hasattr(pipeline, "predict_proba"):
                    proba = pipeline.predict_proba(X)[0]
                    conf = float(max(proba))

                label = gestures[pred] if pred < len(gestures) else str(pred)
                d1 = int(distances_cm[0]) if len(distances_cm) > 0 else 0
                d2 = int(distances_cm[1]) if len(distances_cm) > 1 else 0
                print(f"  {label:>15s}  {conf:.3f}  {d1:>5.0f} / {d2:<5.0f}")
                sys.stdout.flush()

            time.sleep(0.01)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
    finally:
        print("\nStopping...", file=sys.stderr)
        for cname, client in reversed(clients):
            if cname in session_handles:
                try:
                    client.ranging_stop(session_handles[cname])
                except Exception:
                    pass
        for cname, client in reversed(clients):
            if cname in session_handles:
                try:
                    client.session_deinit(session_handles[cname])
                except Exception:
                    pass
        for cname, client in reversed(clients):
            try:
                client.close()
            except Exception:
                pass
        print(f"Done. Predictions made: {pred_count}", file=sys.stderr)


if __name__ == "__main__":
    main()
