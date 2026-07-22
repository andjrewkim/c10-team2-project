"""UWB anchor driver stub (time-of-flight ranging, serial/UART).

TODO: Replace the stub body with real serial reads from a UWB module
      such as DW1000, DWM1001, or Decawave-based anchor.
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timezone
from typing import Any

from sensors.base import BaseSensor, SensorObservation


class UwbAnchorSensor(BaseSensor):
    """Driver for a single UWB ranging anchor.

    Each anchor has a fixed known position in the environment, set via
    the ``position`` constructor arg (which populates
    ``SensorObservation.position``).

    Parameters
    ----------
    sensor_id : str
        Unique identifier for this anchor (e.g. "uwb-anchor-1").
    serial_port : str
        UART device path (e.g. "/dev/ttyACM0" on Linux, "COM3" on Windows).
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
        # TODO: Open serial port — e.g.
        #   import serial
        #   self._ser = serial.Serial(self.serial_port, self.baudrate, timeout=1.0)

    def read(self) -> list[SensorObservation]:
        """Blocking read: request range from this anchor's UWB module.

        Expected observation format (dict):
            anchor_id    : str
            tag_id       : str        (the tag being ranged)
            range_m      : float      (estimated distance in metres)
            rssi         : float      (signal strength, dBm)
            nlos         : bool       (non-line-of-sight flag, if available)

        Returns
        -------
        list[SensorObservation]
            One observation per tag detected in this cycle.
            Empty list if no tags are in range.
        """
        # TODO: Replace with real serial command/response.
        #   self._ser.write(b"get_range\n")
        #   response = self._ser.readline().decode().strip()
        #   tag_id, range_m, rssi = response.split(",")
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
                    position={"x": self.position["x"], "y": self.position["y"], "z": self.position["z"]},
                    tag_id=t["tag_id"],
                    metadata={
                        "serial_port": self.serial_port,
                        "baudrate": self.baudrate,
                        "position": self.position,
                    },
                )
            )
        return results

    # ------------------------------------------------------------------
    # TODO: add private helpers for serial communication protocol
    # ------------------------------------------------------------------


def _test_main() -> None:
    parser = argparse.ArgumentParser(description="UWB anchor self-test")
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=115200)
    args = parser.parse_args()

    sensor = UwbAnchorSensor(
        sensor_id="uwb-self-test",
        serial_port=args.port,
        baudrate=args.baud,
        position={"x": 1.0, "y": 2.0, "z": 0.0},
    )
    obs_list = sensor.read()
    for obs in obs_list:
        print(f"[{obs.timestamp.isoformat()}] {obs.sensor_type}/{obs.sensor_id}")
        print(f"  tag:     {obs.tag_id}")
        print(f"  data:    {obs.observation}")
        print(f"  pos:     {obs.position}")
        print(f"  conf:    {obs.confidence}")


if __name__ == "__main__":
    _test_main()
