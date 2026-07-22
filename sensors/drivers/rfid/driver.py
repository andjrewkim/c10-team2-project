"""RFID reader driver stub (UHF reader with many passive tags, TCP/IP).

TODO: Replace the stub body with real SDK or socket communication to
      an RFID reader (Impinj, ThingMagic, Zebra, etc.).
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timezone
from typing import Any

from sensors.base import BaseSensor, SensorObservation


class RfidReaderSensor(BaseSensor):
    """Driver for a UHF RFID reader.

    The reader is the "sensor"; individual tags appear as separate
    observations sharing the same ``sensor_id``/``sensor_type`` but
    each carrying a unique ``tag_id``.

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
        # TODO: Open TCP socket or initialise SDK client — e.g.
        #   import socket
        #   self._sock = socket.create_connection((self.host, self.port), timeout=5.0)
        #   self._sdk = ImpinjReader(self._sock)

    def read(self) -> list[SensorObservation]:
        """Blocking read: inventory all tags in the field.

        Expected observation format (dict):
            epc            : str       (Electronic Product Code, hex)
            antenna_port   : int       (reader antenna port)
            rssi           : float     (dBm)
            phase_angle    : float     (radians, if available)
            doppler_freq   : float     (Hz, if available)
            read_count     : int       (how many times seen)

        Returns
        -------
        list[SensorObservation]
            One observation per unique tag seen. May be empty.
        """
        # TODO: Send inventory command and parse the response.
        #   For LLRP: RO_ACCESS_SPEC → RO_REPORT.
        #   For Impinj Speedway: `.query()` via Octane SDK.
        tags: list[dict[str, Any]] = [
            {
                "epc": "E280116060000204",
                "antenna_port": 1,
                "rssi": random.uniform(-70, -40),
                "phase_angle": random.uniform(0, 6.28),
                "read_count": random.randint(1, 15),
            },
            {
                "epc": "E280116060000205",
                "antenna_port": 1,
                "rssi": random.uniform(-80, -50),
                "phase_angle": random.uniform(0, 6.28),
                "read_count": random.randint(1, 10),
            },
            {
                "epc": "E280116060000206",
                "antenna_port": 1,
                "rssi": random.uniform(-75, -45),
                "phase_angle": random.uniform(0, 6.28),
                "read_count": random.randint(1, 8),
            },
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
                    metadata={
                        "host": self.host,
                        "port": self.port,
                        "read_power": self.read_power,
                    },
                )
            )
        return results

    # ------------------------------------------------------------------
    # TODO: add helpers for:
    #   - Reader connection state management
    #   - LLRP message encoding/decoding
    #   - Tag-report filtering (duplicate suppression, timeout)
    # ------------------------------------------------------------------


def _test_main() -> None:
    parser = argparse.ArgumentParser(description="RFID reader self-test")
    parser.add_argument("--host", default="192.168.1.100")
    parser.add_argument("--port", type=int, default=5084)
    args = parser.parse_args()

    sensor = RfidReaderSensor(sensor_id="rfid-self-test", host=args.host, port=args.port)
    obs_list = sensor.read()
    print(f"Read {len(obs_list)} tags:")
    for obs in obs_list:
        print(f"  tag={obs.tag_id}  rssi={obs.observation['rssi']:.1f} dBm")


if __name__ == "__main__":
    _test_main()
