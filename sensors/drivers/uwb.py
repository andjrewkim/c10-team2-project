"""UWB anchor driver (time-of-flight ranging, serial/UART).

Expected hardware: DW1000, DWM1001, or Decawave-based UWB anchor.
Each anchor has a fixed known position in the environment.

TODO: Replace stub body with real serial reads from the UWB module.
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timezone

from sensors.base import BaseSensor, SensorObservation


class UwbAnchorSensor(BaseSensor):
    """Driver for a single UWB ranging anchor.

    Parameters
    ----------
    sensor_id : str
        Unique identifier for this anchor.
    serial_port : str
        UART device path (e.g. "/dev/ttyACM0").
    baudrate : int
        Serial baud rate.
    position : dict[str, float] | None
        Fixed 3D position as {"x": ..., "y": ..., "z": ...} in metres.
    """

    def __init__(
        self,
        sensor_id: str,
        serial_port: str = "/dev/ttyACM0",
        baudrate: int = 115200,
        position: dict[str, float] | None = None,
    ) -> None:
        super().__init__(sensor_id=sensor_id, sensor_type="uwb")
        self.serial_port = serial_port
        self.baudrate = baudrate
        self.position = position or {"x": 0.0, "y": 0.0, "z": 0.0}

    def read(self) -> list[SensorObservation]:
        """Blocking read: request range from this anchor's UWB module.

        Expected observation dict:
            anchor_id  : str
            tag_id     : str       (the tag being ranged)
            range_m    : float     (estimated distance in metres)
            rssi       : float     (dBm)
            nlos       : bool      (non-line-of-sight flag)
        """
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
                    metadata={"serial_port": self.serial_port, "baudrate": self.baudrate, "position": self.position},
                )
            )
        return results


__all__ = ["UwbAnchorSensor"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UWB anchor self-test")
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=115200)
    args = parser.parse_args()
    sensor = UwbAnchorSensor(
        sensor_id="uwb-self-test", serial_port=args.port, baudrate=args.baud,
        position={"x": 1.0, "y": 2.0, "z": 0.0},
    )
    for o in sensor.read():
        print(f"[{o.timestamp.isoformat()}] {o.sensor_type}/{o.sensor_id}  tag={o.tag_id}  pos={o.position}  conf={o.confidence}")
