"""UHF RFID reader driver (TCP/IP, many passive tags per reader).

Expected hardware: Impinj Speedway / ThingMagic / Zebra UHF reader.
The reader is the "sensor"; individual tags appear as separate observations
with unique tag_id.

TODO: Replace stub body with real SDK/socket communication (LLRP or Octane SDK).
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timezone
from typing import Any

from sensors.base import BaseSensor, SensorObservation


class RfidReaderSensor(BaseSensor):
    """Driver for a UHF RFID reader.

    Parameters
    ----------
    sensor_id : str
        Unique identifier for the reader (e.g. "rfid-gate-1").
    host : str
        IP address or hostname of the reader.
    port : int
        TCP port for the reader's API or LLRP interface.
    read_power : int
        Transmit power in dBm.
    """

    def __init__(
        self,
        sensor_id: str,
        host: str = "192.168.1.100",
        port: int = 5084,
        read_power: int = 30,
    ) -> None:
        super().__init__(sensor_id=sensor_id, sensor_type="rfid")
        self.host = host
        self.port = port
        self.read_power = read_power

    def read(self) -> list[SensorObservation]:
        """Blocking read: inventory all tags in the field.

        Returns one observation per unique tag seen. Each carries a tag_id.
        """
        tags: list[dict[str, Any]] = [
            {"epc": "E280116060000204", "antenna_port": 1, "rssi": random.uniform(-70, -40),
             "phase_angle": random.uniform(0, 6.28), "read_count": random.randint(1, 15)},
            {"epc": "E280116060000205", "antenna_port": 1, "rssi": random.uniform(-80, -50),
             "phase_angle": random.uniform(0, 6.28), "read_count": random.randint(1, 10)},
            {"epc": "E280116060000206", "antenna_port": 1, "rssi": random.uniform(-75, -45),
             "phase_angle": random.uniform(0, 6.28), "read_count": random.randint(1, 8)},
        ]
        results: list[SensorObservation] = []
        for t in tags:
            results.append(
                SensorObservation(
                    sensor_id=self.sensor_id,
                    sensor_type=self.sensor_type,
                    timestamp=datetime.now(timezone.utc),
                    observation=t,
                    confidence=0.85,
                    tag_id=t["epc"],
                    metadata={"host": self.host, "port": self.port, "read_power": self.read_power},
                )
            )
        return results


__all__ = ["RfidReaderSensor"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RFID reader self-test")
    parser.add_argument("--host", default="192.168.1.100")
    parser.add_argument("--port", type=int, default=5084)
    args = parser.parse_args()
    sensor = RfidReaderSensor(sensor_id="rfid-self-test", host=args.host, port=args.port)
    for o in sensor.read():
        print(f"  tag={o.tag_id}  rssi={o.observation['rssi']:.1f} dBm")
