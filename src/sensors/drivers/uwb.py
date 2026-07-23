"""UWB anchor driver (time-of-flight ranging, serial/UART).

Supports two modes:

- **mock** (default): synthetic ranging values for development/testing.
- **serial**: real serial stream from a DW1000 / DWM1001 / Qorvo UWB
  device, parsed with the lab's ranging log parser and outlier filtering.

The ranging parser, distance filter, and feature extraction live in
``sensors/lab_integration.uwb`` and were ported from the COSMOS
``lab09-uwb-lab`` tools.
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timezone

from src.sensors.base import BaseSensor, SensorObservation


class UwbAnchorSensor(BaseSensor):
    """Driver for a single UWB ranging anchor.

    Parameters
    ----------
    sensor_id : str
        Unique identifier for this anchor.
    serial_port : str
        UART device path (e.g. ``"/dev/ttyACM0"``).
    baudrate : int
        Serial baud rate.
    position : dict[str, float] | None
        Fixed 3D position as ``{"x": ..., "y": ..., "z": ...}`` in metres.
    mode : str
        ``"mock"`` or ``"serial"``.
    """

    def __init__(
        self,
        sensor_id: str,
        serial_port: str = "/dev/ttyACM0",
        baudrate: int = 115200,
        position: dict[str, float] | None = None,
        mode: str = "mock",
    ) -> None:
        super().__init__(sensor_id=sensor_id, sensor_type="uwb")
        self.serial_port = serial_port
        self.baudrate = baudrate
        self.position = position or {"x": 0.0, "y": 0.0, "z": 0.0}
        self.mode = mode

    def read(self) -> list[SensorObservation]:
        if self.mode == "serial":
            return self._read_serial()
        return self._read_mock()

    # ------------------------------------------------------------------
    # Mock mode
    # ------------------------------------------------------------------

    def _read_mock(self) -> list[SensorObservation]:
        tags = [
            {"tag_id": "tag-001", "range_m": random.uniform(0.5, 8.0)},
            {"tag_id": "tag-002", "range_m": random.uniform(0.5, 8.0)},
        ]
        results: list[SensorObservation] = []
        for t in tags:
            obs_data = {
                "anchor_id": self.sensor_id,
                "tag_id": t["tag_id"],
                "range_m": t["range_m"],
                "rssi": random.uniform(-90.0, -30.0),
                "nlos": random.random() < 0.2,
            }
            results.append(
                SensorObservation(
                    sensor_id=self.sensor_id,
                    sensor_type=self.sensor_type,
                    timestamp=datetime.now(timezone.utc),
                    observation=obs_data,
                    confidence=0.9 if not obs_data["nlos"] else 0.5,
                    position=dict(self.position),
                    tag_id=t["tag_id"],
                    metadata={"mode": "mock", "serial_port": self.serial_port,
                              "baudrate": self.baudrate,
                              "position": self.position},
                )
            )
        return results

    # ------------------------------------------------------------------
    # Serial mode (real UWB device)
    # ------------------------------------------------------------------

    def _read_serial(self) -> list[SensorObservation]:
        try:
            from sensors.lab_integration.uwb import (
                RangeLogParser,
                filter_distances_cm,
            )
        except ImportError:
            return self._read_mock()

        try:
            import serial as pyserial
        except ImportError:
            return self._read_mock()

        observations: list[SensorObservation] = []
        try:
            ser = pyserial.Serial(self.serial_port, baudrate=self.baudrate,
                                  timeout=0.5)
            parser = RangeLogParser()
            raw_distances: list[float] = []
            for _ in range(30):
                raw = ser.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").strip()
                sample = parser.feed(line)
                if sample and sample.get("status") == "Ok":
                    dist_cm = float(sample.get("distance_cm", 0.0))
                    if 0.0 < dist_cm < 60000.0:
                        raw_distances.append(dist_cm)
            ser.close()

            if raw_distances:
                filtered = filter_distances_cm(raw_distances)
                if filtered:
                    mean_dist_m = sum(filtered) / len(filtered) / 100.0
                    nlos_flag = len(filtered) < len(raw_distances) * 0.5
                else:
                    mean_dist_m = sum(raw_distances) / len(raw_distances) / 100.0
                    nlos_flag = True

                obs_data = {
                    "anchor_id": self.sensor_id,
                    "tag_id": "tag-001",
                    "range_m": mean_dist_m,
                    "rssi": -60.0,
                    "nlos": nlos_flag,
                    "raw_samples": len(raw_distances),
                    "filtered_samples": len(filtered) if filtered else 0,
                }
                observations.append(
                    SensorObservation(
                        sensor_id=self.sensor_id,
                        sensor_type=self.sensor_type,
                        timestamp=datetime.now(timezone.utc),
                        observation=obs_data,
                        confidence=0.9 if not nlos_flag else 0.5,
                        position=dict(self.position),
                        tag_id="tag-001",
                        metadata={"mode": "serial",
                                  "serial_port": self.serial_port,
                                  "baudrate": self.baudrate,
                                  "position": self.position},
                    )
                )
        except Exception:
            return self._read_mock()
        return observations


__all__ = ["UwbAnchorSensor"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UWB anchor self-test")
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--mode", choices=["mock", "serial"], default="mock")
    args = parser.parse_args()
    sensor = UwbAnchorSensor(
        sensor_id="uwb-self-test", serial_port=args.port, baudrate=args.baud,
        position={"x": 1.0, "y": 2.0, "z": 0.0}, mode=args.mode,
    )
    for o in sensor.read():
        obs = o.observation
        print(f"[{o.timestamp.isoformat()}] {o.sensor_type}/{o.sensor_id}  "
              f"tag={o.tag_id}  range={obs.get('range_m', 0):.3f} m  "
              f"nlos={obs.get('nlos', False)}  conf={o.confidence}")
