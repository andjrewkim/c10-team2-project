"""UHF RFID reader driver (TCP/IP, many passive tags per reader).

Supports two modes:

- **mock** (default): synthetic tag inventory for development/testing.
- **tcp**: real TCP stream to an Impinj / ThingMagic / Zebra reader,
  parsed with the lab's RFID log utilities and touch detection.

The TCP reader and touch-detection logic live in
``sensors/lab_integration.rfid`` and were ported from the COSMOS
``lab06-rfid-lab`` tools.
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timezone
from typing import Any

from src.sensors.base import BaseSensor, SensorObservation


class RfidReaderSensor(BaseSensor):
    """Driver for a UHF RFID reader.

    Parameters
    ----------
    sensor_id : str
        Unique identifier for the reader (e.g. ``"rfid-gate-1"``).
    host : str
        IP address or hostname of the reader.
    port : int
        TCP port for the reader's API or LLRP interface.
    read_power : int
        Transmit power in dBm.
    mode : str
        ``"mock"`` or ``"tcp"``.
    selected_epcs : list[str] | None
        EPCs to filter (empty = all visible).
    """

    def __init__(
        self,
        sensor_id: str,
        host: str = "192.168.1.100",
        port: int = 5084,
        read_power: int = 30,
        mode: str = "mock",
        selected_epcs: list[str] | None = None,
    ) -> None:
        super().__init__(sensor_id=sensor_id, sensor_type="rfid")
        self.host = host
        self.port = port
        self.read_power = read_power
        self.mode = mode
        self.selected_epcs = selected_epcs or []

    def read(self) -> list[SensorObservation]:
        if self.mode == "tcp":
            return self._read_tcp()
        return self._read_mock()

    # ------------------------------------------------------------------
    # Mock mode
    # ------------------------------------------------------------------

    def _read_mock(self) -> list[SensorObservation]:
        tags: list[dict[str, Any]] = [
            {"epc": "E280116060000204", "antenna_port": 1,
             "rssi": random.uniform(-70, -40),
             "phase_angle": random.uniform(0, 6.28),
             "read_count": random.randint(1, 15)},
            {"epc": "E280116060000205", "antenna_port": 1,
             "rssi": random.uniform(-80, -50),
             "phase_angle": random.uniform(0, 6.28),
             "read_count": random.randint(1, 10)},
            {"epc": "E280116060000206", "antenna_port": 1,
             "rssi": random.uniform(-75, -45),
             "phase_angle": random.uniform(0, 6.28),
             "read_count": random.randint(1, 8)},
        ]
        return [
            SensorObservation(
                sensor_id=self.sensor_id,
                sensor_type=self.sensor_type,
                timestamp=datetime.now(timezone.utc),
                observation=t,
                confidence=0.85,
                tag_id=t["epc"],
                metadata={"mode": "mock", "host": self.host, "port": self.port,
                          "read_power": self.read_power},
            )
            for t in tags
        ]

    # ------------------------------------------------------------------
    # TCP mode (real reader)
    # ------------------------------------------------------------------

    def _read_tcp(self) -> list[SensorObservation]:
        try:
            from sensors.lab_integration.rfid import read_tcp_stream
        except ImportError:
            return self._read_mock()

        try:
            records = read_tcp_stream(self.host, self.port, duration=2.0)
        except Exception:
            return self._read_mock()

        # Filter to selected EPCs if configured.
        if self.selected_epcs:
            selected_set = {e.upper() for e in self.selected_epcs}
            records = [r for r in records if r.epc.upper() in selected_set]

        # Group by EPC and take the most recent reading per tag.
        seen: dict[str, Any] = {}
        for rec in records:
            seen[rec.epc] = rec

        return [
            SensorObservation(
                sensor_id=self.sensor_id,
                sensor_type=self.sensor_type,
                timestamp=datetime.now(timezone.utc),
                observation={
                    "epc": rec.epc,
                    "rssi": rec.rssi,
                    "read_count": rec.read_count,
                    "antenna_port": 1,
                },
                confidence=0.85,
                tag_id=rec.epc,
                metadata={"mode": "tcp", "host": self.host, "port": self.port,
                          "read_power": self.read_power},
            )
            for rec in seen.values()
        ]


__all__ = ["RfidReaderSensor"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RFID reader self-test")
    parser.add_argument("--host", default="192.168.1.100")
    parser.add_argument("--port", type=int, default=5084)
    parser.add_argument("--mode", choices=["mock", "tcp"], default="mock")
    args = parser.parse_args()
    sensor = RfidReaderSensor(sensor_id="rfid-self-test", host=args.host,
                               port=args.port, mode=args.mode)
    for o in sensor.read():
        print(f"  tag={o.tag_id}  rssi={o.observation['rssi']:.1f} dBm  "
              f"mode={o.metadata['mode']}")
