#!/usr/bin/env python3
"""Command-line tool for running a recording session.

Usage
-----
    python -m recording.cli --label "walking" --participant alice --duration 30

The script will:
    1. Print a countdown so the subject and operator stay in sync.
    2. Start recording observations from all configured sensors.
    3. Stop after the specified duration.
    4. Write the session file to ``data/raw/`` and print the path.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from recording.session import RecordingSession
from transport.mqtt_client import MqttConfig

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]


def _load_mqtt_config(path: str | None) -> MqttConfig:
    if path is not None and yaml is not None:
        with open(path) as f:
            raw = yaml.safe_load(f)
        return MqttConfig(
            host=raw.get("host", "localhost"),
            port=raw.get("port", 1883),
            keepalive=raw.get("keepalive", 60),
            username=raw.get("username"),
            password=raw.get("password"),
            tls_enabled=raw.get("tls_enabled", False),
            client_id=raw.get("client_id", "recording-cli"),
            topic_prefix=raw.get("topic_prefix", ""),
            qos=raw.get("qos", 1),
        )
    # Fall back to environment variables
    return MqttConfig(
        host=os.getenv("MQTT_HOST", "localhost"),
        port=int(os.getenv("MQTT_PORT", "1883")),
        keepalive=int(os.getenv("MQTT_KEEPALIVE", "60")),
        username=os.getenv("MQTT_USERNAME"),
        password=os.getenv("MQTT_PASSWORD"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record a labeled sensor-data session.",
    )
    parser.add_argument("--label", required=True, help="Activity label (e.g. 'walking')")
    parser.add_argument("--participant", required=True, dest="participant_id", help="Participant ID (e.g. 'alice')")
    parser.add_argument("--duration", type=int, default=30, help="Recording duration in seconds (default: 30)")
    parser.add_argument("--mqtt-config", help="Path to MQTT YAML config (optional; falls back to env vars)")
    parser.add_argument("--output-dir", default="data/raw", help="Output directory (default: data/raw)")
    args = parser.parse_args()

    mqtt_config = _load_mqtt_config(args.mqtt_config)
    session = RecordingSession(mqtt_config=mqtt_config, output_dir=args.output_dir)

    print("=" * 50)
    print(f"Session config:")
    print(f"  Label:        {args.label}")
    print(f"  Participant:  {args.participant_id}")
    print(f"  Duration:     {args.duration}s")
    print(f"  MQTT broker:  {mqtt_config.host}:{mqtt_config.port}")
    print("=" * 50)

    # Countdown
    print("\nStarting in...")
    for i in range(5, 0, -1):
        print(f"  {i}...")
        time.sleep(1)

    print("\n[RECORDING] Started — perform the action now.")
    session.start(label=args.label, participant_id=args.participant_id)

    try:
        for remaining in range(args.duration, 0, -1):
            sys.stdout.write(f"\r  {remaining}s remaining  ")
            sys.stdout.flush()
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")

    print("\n[RECORDING] Stopping...")
    output_path = session.stop()

    print(f"\nDone. Observations saved to: {output_path}")


if __name__ == "__main__":
    main()
